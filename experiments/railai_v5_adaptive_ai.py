"""
RailAI Adaptive Intelligence v5
--------------------------------
A laptop-friendly SIH prototype for:
- Real-time train telemetry simulation
- Weather-aware speed prediction
- Dynamic ETA prediction
- Online/adaptive learning from prediction errors
- Concept-drift detection
- Ensemble weighting based on recent performance
- Replay buffer for continual learning
- Sensor-fusion confidence
- Anomaly detection
- Incident/risk scoring
- Optional YOLO camera detection
- Explainable predictions
- Persistent model state + event logs

IMPORTANT:
This is a research/demo decision-support prototype.
It must NOT directly control railway signalling, interlocking,
movement authority, braking, or other safety-critical equipment.
"""

import os
import json
import math
import time
import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest


# ================================================================
# CONFIG
# ================================================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

MODEL_DIR = "railai_models"
EVENT_LOG = "railai_events.csv"
PREDICTION_LOG = "railai_predictions.csv"

os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# UTILITIES
# ================================================================

def clamp(x, low, high):
    return max(low, min(high, x))


def now():
    return datetime.now().isoformat(timespec="seconds")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# ================================================================
# WEATHER
# ================================================================

class WeatherEngine:
    """
    Uses Open-Meteo when internet is available.
    Falls back to simulated weather if not.
    """

    def get(self, lat, lon):
        try:
            import requests

            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "rain,"
                "wind_speed_10m,"
                "visibility"
            )

            r = requests.get(url, timeout=4)
            r.raise_for_status()

            c = r.json()["current"]

            return {
                "temperature": float(c.get("temperature_2m", 28)),
                "humidity": float(c.get("relative_humidity_2m", 60)),
                "precipitation": float(c.get("precipitation", 0)),
                "rain": float(c.get("rain", 0)),
                "wind": float(c.get("wind_speed_10m", 10)),
                "visibility": float(c.get("visibility", 10000)),
                "source": "live"
            }

        except Exception:
            # Stable offline fallback
            return {
                "temperature": random.uniform(22, 35),
                "humidity": random.uniform(40, 90),
                "precipitation": max(0, np.random.normal(1.0, 2.0)),
                "rain": max(0, np.random.normal(0.7, 2.0)),
                "wind": max(0, np.random.normal(12, 7)),
                "visibility": max(1000, np.random.normal(9000, 1800)),
                "source": "simulation"
            }

    @staticmethod
    def operating_factor(w):
        factor = 1.0

        if w["rain"] > 2:
            factor -= 0.04
        if w["rain"] > 10:
            factor -= 0.10
        if w["wind"] > 30:
            factor -= 0.04
        if w["wind"] > 50:
            factor -= 0.10
        if w["visibility"] < 5000:
            factor -= 0.07
        if w["visibility"] < 2000:
            factor -= 0.12

        return clamp(factor, 0.55, 1.0)


# ================================================================
# ROUTE / TRACK KNOWLEDGE
# ================================================================

ROUTES = {
    "LKO_BARABANKI": {
        "name": "Lucknow - Barabanki",
        "speed_limit": 110,
        "track_quality": 0.94,
        "congestion": 0.45,
        "gradient": 0.04,
        "maintenance_risk": 0.12
    },
    "BARABANKI_AYODHYA": {
        "name": "Barabanki - Ayodhya",
        "speed_limit": 100,
        "track_quality": 0.91,
        "congestion": 0.55,
        "gradient": 0.08,
        "maintenance_risk": 0.18
    },
    "AYODHYA_SULTANPUR": {
        "name": "Ayodhya - Sultanpur",
        "speed_limit": 95,
        "track_quality": 0.88,
        "congestion": 0.40,
        "gradient": 0.12,
        "maintenance_risk": 0.22
    },
    "LKO_KANPUR": {
        "name": "Lucknow - Kanpur",
        "speed_limit": 120,
        "track_quality": 0.96,
        "congestion": 0.70,
        "gradient": 0.03,
        "maintenance_risk": 0.08
    }
}


# ================================================================
# ONLINE STANDARDIZER
# ================================================================

class OnlineStandardizer:
    """
    Running mean/std without needing the entire dataset in RAM.
    """

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=float)
        self.m2 = np.ones(n_features, dtype=float)

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        for row in X:
            self.n += 1

            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.n < 2:
            return X

        var = self.m2 / max(self.n - 1, 1)
        std = np.sqrt(np.maximum(var, 1e-8))

        return (X - self.mean) / std

    def state(self):
        return {
            "n": int(self.n),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist()
        }

    def load_state(self, s):
        self.n = int(s["n"])
        self.mean = np.array(s["mean"], dtype=float)
        self.m2 = np.array(s["m2"], dtype=float)


# ================================================================
# PAGE-HINKLEY CONCEPT DRIFT DETECTOR
# ================================================================

class PageHinkley:
    """
    Detects sustained changes in prediction error.

    This lets the system react when railway conditions change,
    instead of assuming yesterday's patterns remain valid forever.
    """

    def __init__(self, delta=0.01, threshold=8.0):
        self.delta = delta
        self.threshold = threshold

        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.n = 0

    def update(self, value):
        value = abs(float(value))

        self.n += 1
        self.mean += (value - self.mean) / self.n

        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)

        drift = (
            self.cumulative - self.minimum
            > self.threshold
        )

        if drift:
            self.cumulative = 0.0
            self.minimum = 0.0

        return drift


# ================================================================
# ADAPTIVE ENSEMBLE
# ================================================================

class AdaptiveEnsemble:
    """
    Three-model ensemble:
      1. Online SGD model
      2. Online SGD model with a different learning rate
      3. Tree model periodically refreshed from replay memory

    The ensemble weights are continuously changed using recent
    prediction error. Better models receive more weight.
    """

    def __init__(self, n_features, task_name):
        self.n_features = n_features
        self.task_name = task_name

        self.scaler = OnlineStandardizer(n_features)

        self.models = [
            SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=0.0005,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=SEED
            ),
            SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.003,
                random_state=SEED + 1
            ),
            ExtraTreesRegressor(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1
            )
        ]

        self.initialized = [False, False, False]

        self.weights = np.array(
            [0.40, 0.35, 0.25],
            dtype=float
        )

        self.errors = [
            deque(maxlen=100),
            deque(maxlen=100),
            deque(maxlen=100)
        ]

        self.replay_X = deque(maxlen=2500)
        self.replay_y = deque(maxlen=2500)

        self.drift = PageHinkley(
            delta=0.01,
            threshold=8.0
        )

        self.total_updates = 0
        self.drift_events = 0

    def _initialize_sgd(self, model, X, y, index):
        model.partial_fit(X, y)
        self.initialized[index] = True

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.scaler.partial_fit(X)
        Z = self.scaler.transform(X)

        preds = []

        # SGD models
        for i in [0, 1]:
            if not self.initialized[i]:
                preds.append(np.array([np.mean(self.replay_y)] if self.replay_y else [0.0]))
            else:
                preds.append(self.models[i].predict(Z))

        # Tree model
        if self.initialized[2]:
            preds.append(self.models[2].predict(X))
        else:
            preds.append(
                np.array([np.mean(self.replay_y)] if self.replay_y else [0.0])
            )

        pred_matrix = np.vstack(preds)

        # Safety against numerical explosions
        pred_matrix = np.nan_to_num(
            pred_matrix,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        )

        p = float(np.dot(self.weights, pred_matrix[:, 0]))

        uncertainty = float(np.std(pred_matrix[:, 0]))

        return p, uncertainty, pred_matrix[:, 0]

    def update(self, X, y, sample_weight=1.0):
        X = np.asarray(X, dtype=float).reshape(1, -1)
        y = float(y)

        self.scaler.partial_fit(X)
        Z = self.scaler.transform(X)

        self.replay_X.append(X[0].copy())
        self.replay_y.append(y)

        # Online models learn immediately.
        for i in [0, 1]:
            self.models[i].partial_fit(
                Z,
                np.array([y]),
                sample_weight=np.array([sample_weight])
            )
            self.initialized[i] = True

        self.total_updates += 1

        # Refresh tree periodically from replay memory.
        if (
            len(self.replay_y) >= 100
            and self.total_updates % 25 == 0
        ):
            self.models[2].fit(
                np.array(self.replay_X),
                np.array(self.replay_y)
            )
            self.initialized[2] = True

    def learn_from_error(self, predictions, actual):
        """
        Change ensemble weights based on recent errors.

        A model with lower recent error gets more weight.
        """

        actual = float(actual)

        for i, p in enumerate(predictions):
            error = abs(float(p) - actual)
            self.errors[i].append(error)

        recent_mae = np.array([
            np.mean(e) if e else 1e6
            for e in self.errors
        ])

        # Lower error -> higher score.
        scores = 1.0 / (recent_mae + 1e-4)
        scores = scores / scores.sum()

        # Smooth update prevents violent weight oscillations.
        self.weights = (
            0.85 * self.weights
            + 0.15 * scores
        )

        self.weights /= self.weights.sum()

        drift = self.drift.update(
            abs(np.mean(recent_mae))
        )

        if drift:
            self.drift_events += 1

            # Temporarily flatten the ensemble when the environment
            # changes, giving multiple models a chance to relearn.
            self.weights = np.array(
                [0.34, 0.33, 0.33]
            )

        return drift, recent_mae

    def save_state(self, path):
        state = {
            "task_name": self.task_name,
            "weights": self.weights.tolist(),
            "total_updates": self.total_updates,
            "drift_events": self.drift_events,
            "scaler": self.scaler.state(),
            "replay_X": np.array(self.replay_X).tolist(),
            "replay_y": list(self.replay_y)
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def load_state(self, path):
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.weights = np.array(
                state["weights"],
                dtype=float
            )

            self.total_updates = int(
                state["total_updates"]
            )

            self.drift_events = int(
                state["drift_events"]
            )

            self.scaler.load_state(
                state["scaler"]
            )

            self.replay_X = deque(
                [np.array(x, dtype=float)
                 for x in state["replay_X"]],
                maxlen=2500
            )

            self.replay_y = deque(
                state["replay_y"],
                maxlen=2500
            )

            # Reconstruct models from replay memory.
            if len(self.replay_y) >= 10:
                X = np.array(self.replay_X)
                y = np.array(self.replay_y)

                Z = self.scaler.transform(X)

                for i in [0, 1]:
                    self.models[i].partial_fit(Z, y)
                    self.initialized[i] = True

                if len(y) >= 100:
                    self.models[2].fit(X, y)
                    self.initialized[2] = True

            return True

        except Exception:
            return False


# ================================================================
# FEATURE ENGINE
# ================================================================

FEATURES = [
    "speed_limit",
    "track_quality",
    "congestion",
    "maintenance_risk",
    "gradient",
    "rain",
    "wind",
    "visibility",
    "temperature",
    "humidity",
    "scheduled_speed",
    "actual_speed",
    "acceleration",
    "current_delay",
    "remaining_distance",
    "remaining_stops",
    "station_congestion",
    "hour",
    "minute_sin",
    "minute_cos",
    "animal_risk",
    "obstacle_risk",
    "sensor_confidence"
]


def make_features(train, route, weather, actual_speed,
                  acceleration, animal_risk=0.0,
                  obstacle_risk=0.0,
                  sensor_confidence=1.0):

    minute = datetime.now().minute
    angle = 2 * math.pi * minute / 60

    return np.array([
        route["speed_limit"],
        route["track_quality"],
        route["congestion"],
        route["maintenance_risk"],
        route["gradient"],
        weather["rain"],
        weather["wind"],
        weather["visibility"],
        weather["temperature"],
        weather["humidity"],
        train["scheduled_speed"],
        actual_speed,
        acceleration,
        train["current_delay"],
        train["remaining_distance"],
        train["remaining_stops"],
        train["station_congestion"],
        datetime.now().hour,
        math.sin(angle),
        math.cos(angle),
        animal_risk,
        obstacle_risk,
        sensor_confidence
    ], dtype=float)


# ================================================================
# SENSOR FUSION
# ================================================================

class SensorFusion:
    """
    Compares independent sources.
    In a real deployment these could be GPS, RTIS, track sensors,
    CCTV and other authorized telemetry.
    """

    def fuse_speed(self, gps_speed, track_speed=None,
                   vision_speed=None):

        values = [gps_speed]
        reliabilities = [0.92]

        if track_speed is not None:
            values.append(track_speed)
            reliabilities.append(0.98)

        if vision_speed is not None:
            values.append(vision_speed)
            reliabilities.append(0.80)

        values = np.array(values, dtype=float)
        reliabilities = np.array(reliabilities, dtype=float)

        fused = np.average(
            values,
            weights=reliabilities
        )

        disagreement = np.std(values)

        confidence = clamp(
            100 - disagreement * 5,
            0,
            100
        )

        return float(fused), float(confidence)


# ================================================================
# VISION
# ================================================================

class VisionEngine:
    """
    Optional YOLO module.

    General-purpose YOLO is NOT a railway-certified hazard model.
    For SIH, train/fine-tune a railway-specific dataset later.
    """

    def __init__(self):
        self.model = None

        try:
            from ultralytics import YOLO
            self.model = YOLO("yolo11n.pt")
            print("[VISION] YOLO enabled.")
        except Exception:
            print(
                "[VISION] YOLO not installed. "
                "Camera detection disabled."
            )

    def analyze(self, frame):
        if self.model is None:
            return []

        results = self.model(
            frame,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    "object": result.names[cls],
                    "confidence": conf,
                    "box": box.xyxy[0].tolist()
                })

        return detections


def risk_from_detections(detections):
    animal_classes = {
        "elephant": 1.0,
        "horse": 0.7,
        "cow": 0.8,
        "dog": 0.4,
        "sheep": 0.5
    }

    obstacle_classes = {
        "person": 0.9,
        "car": 0.9,
        "truck": 1.0,
        "bus": 1.0,
        "motorcycle": 0.7,
        "bicycle": 0.6
    }

    animal = 0.0
    obstacle = 0.0

    for d in detections:
        obj = d["object"]
        conf = d["confidence"]

        animal = max(
            animal,
            animal_classes.get(obj, 0) * conf
        )

        obstacle = max(
            obstacle,
            obstacle_classes.get(obj, 0) * conf
        )

    return animal, obstacle


# ================================================================
# INCIDENT / RISK ENGINE
# ================================================================

class RiskEngine:

    def evaluate(
        self,
        speed_error,
        acceleration,
        weather,
        anomaly,
        animal_risk,
        obstacle_risk,
        sensor_confidence,
        maintenance_risk
    ):
        score = 0.0
        reasons = []

        if animal_risk > 0.7:
            score += 60
            reasons.append("High-confidence animal hazard")

        elif animal_risk > 0.35:
            score += 30
            reasons.append("Possible animal near track")

        if obstacle_risk > 0.7:
            score += 60
            reasons.append("High-confidence track obstruction")

        elif obstacle_risk > 0.35:
            score += 30
            reasons.append("Possible track obstruction")

        if abs(speed_error) > 20:
            score += 20
            reasons.append("Large speed prediction deviation")

        if acceleration < -2.0:
            score += 25
            reasons.append("Abnormal deceleration")

        if weather["rain"] > 10:
            score += 12
            reasons.append("Heavy rainfall")

        if weather["visibility"] < 2000:
            score += 15
            reasons.append("Poor visibility")

        if anomaly:
            score += 20
            reasons.append("Telemetry anomaly")

        if sensor_confidence < 60:
            score += 12
            reasons.append("Low sensor agreement")

        if maintenance_risk > 0.25:
            score += 10
            reasons.append("Elevated infrastructure risk")

        score = clamp(score, 0, 100)

        if score >= 70:
            level = "CRITICAL"
        elif score >= 40:
            level = "HIGH"
        elif score >= 20:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "score": score,
            "level": level,
            "reasons": reasons
        }


# ================================================================
# BLOCK / CONFLICT RECOMMENDER
# ================================================================

class BlockPlanner:

    def recommend(self, trains):
        """
        Decision-support only.
        Does NOT issue movement authority.
        """

        recommendations = []

        sorted_trains = sorted(
            trains,
            key=lambda t: (
                t["predicted_arrival_timestamp"],
                -t["priority"]
            )
        )

        for i, train in enumerate(sorted_trains):
            action = "NORMAL"
            reason = "No immediate conflict"

            for other in sorted_trains[i + 1:]:
                time_gap = (
                    other["predicted_arrival_timestamp"]
                    - train["predicted_arrival_timestamp"]
                )

                same_junction = (
                    train["junction"]
                    == other["junction"]
                )

                if same_junction and time_gap < 180:
                    action = "RESEQUENCE"
                    reason = (
                        f"Potential junction conflict with "
                        f"{other['train_id']}"
                    )
                    break

            recommendations.append({
                "train_id": train["train_id"],
                "action": action,
                "reason": reason
            })

        return recommendations


# ================================================================
# ADAPTIVE RAIL AI
# ================================================================

class RailAI:

    def __init__(self):
        self.weather = WeatherEngine()
        self.sensor = SensorFusion()
        self.vision = VisionEngine()
        self.risk = RiskEngine()
        self.block = BlockPlanner()

        self.speed_ai = AdaptiveEnsemble(
            len(FEATURES),
            "speed"
        )

        self.delay_ai = AdaptiveEnsemble(
            len(FEATURES),
            "delay"
        )

        self.state_path_speed = os.path.join(
            MODEL_DIR,
            "speed_state.json"
        )

        self.state_path_delay = os.path.join(
            MODEL_DIR,
            "delay_state.json"
        )

        loaded_speed = self.speed_ai.load_state(
            self.state_path_speed
        )

        loaded_delay = self.delay_ai.load_state(
            self.state_path_delay
        )

        print(
            f"[AI] Speed memory loaded: {loaded_speed}"
        )

        print(
            f"[AI] Delay memory loaded: {loaded_delay}"
        )

    # ------------------------------------------------------------
    # INITIAL BOOTSTRAP
    # ------------------------------------------------------------

    def bootstrap(self, samples=2500):
        """
        Creates a synthetic prior so the prototype works immediately.

        Replace this with historical railway data for a real project.
        """

        if self.speed_ai.total_updates > 50:
            return

        print("[AI] Bootstrapping adaptive models...")

        for _ in range(samples):
            route = random.choice(
                list(ROUTES.values())
            )

            train = {
                "scheduled_speed": random.uniform(80, 120),
                "current_delay": max(
                    0, np.random.normal(8, 6)
                ),
                "remaining_distance": random.uniform(10, 400),
                "remaining_stops": random.randint(0, 10),
                "station_congestion": random.random()
            }

            weather = {
                "rain": max(0, np.random.normal(2, 4)),
                "wind": max(0, np.random.normal(12, 10)),
                "visibility": max(
                    1200,
                    np.random.normal(9000, 2500)
                ),
                "temperature": random.uniform(15, 38),
                "humidity": random.uniform(30, 95)
            }

            actual_speed = random.uniform(
                40,
                route["speed_limit"]
            )

            acceleration = np.random.normal(
                0,
                0.7
            )

            X = make_features(
                train,
                route,
                weather,
                actual_speed,
                acceleration,
                random.random() * 0.1,
                random.random() * 0.1,
                0.9
            )

            # Synthetic but physically sensible target.
            weather_factor = (
                1.0
                - min(weather["rain"], 20) * 0.006
                - max(weather["wind"] - 30, 0) * 0.001
                - max(5000 - weather["visibility"], 0) / 100000
            )

            target_speed = (
                route["speed_limit"]
                * weather_factor
                * (0.84 - 0.18 * route["congestion"])
                * route["track_quality"]
            )

            target_speed += np.random.normal(0, 4)
            target_speed = clamp(
                target_speed,
                20,
                route["speed_limit"]
            )

            delay = (
                train["current_delay"] * 0.55
                + route["congestion"] * 12
                + (1 - weather_factor) * 40
                + train["remaining_stops"] * 1.5
                + np.random.normal(0, 2)
            )

            delay = max(0, delay)

            self.speed_ai.update(X, target_speed)
            self.delay_ai.update(X, delay)

        self.speed_ai.save_state(
            self.state_path_speed
        )

        self.delay_ai.save_state(
            self.state_path_delay
        )

        print(
            "[AI] Bootstrap complete."
        )

    # ------------------------------------------------------------
    # PROCESS ONE TRAIN
    # ------------------------------------------------------------

    def process_train(
        self,
        train,
        actual_speed=None,
        track_speed=None,
        vision_detections=None
    ):
        route = ROUTES[train["route"]]

        weather = self.weather.get(
            train["latitude"],
            train["longitude"]
        )

        # --------------------------------------------------------
        # SPEED SENSOR FUSION
        # --------------------------------------------------------

        if actual_speed is None:
            actual_speed = random.uniform(
                50,
                route["speed_limit"]
            )

        fused_speed, sensor_confidence = (
            self.sensor.fuse_speed(
                actual_speed,
                track_speed=track_speed
            )
        )

        acceleration = (
            fused_speed - train.get(
                "previous_speed",
                fused_speed
            )
        ) / max(
            train.get("update_seconds", 10),
            1
        )

        # --------------------------------------------------------
        # VISION RISK
        # --------------------------------------------------------

        animal_risk = 0.0
        obstacle_risk = 0.0

        if vision_detections:
            animal_risk, obstacle_risk = (
                risk_from_detections(
                    vision_detections
                )
            )

        # --------------------------------------------------------
        # FEATURES
        # --------------------------------------------------------

        X = make_features(
            train,
            route,
            weather,
            fused_speed,
            acceleration,
            animal_risk,
            obstacle_risk,
            sensor_confidence / 100
        )

        # --------------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------------

        predicted_speed, speed_uncertainty, speed_models = (
            self.speed_ai.predict(X)
        )

        predicted_delay, delay_uncertainty, delay_models = (
            self.delay_ai.predict(X)
        )

        weather_limit = (
            route["speed_limit"]
            * self.weather.operating_factor(weather)
        )

        predicted_speed = clamp(
            predicted_speed,
            10,
            weather_limit
        )

        speed_error = (
            fused_speed - predicted_speed
        )

        anomaly = self._anomaly_check(
            X,
            speed_error
        )

        risk = self.risk.evaluate(
            speed_error,
            acceleration,
            weather,
            anomaly,
            animal_risk,
            obstacle_risk,
            sensor_confidence,
            route["maintenance_risk"]
        )

        # --------------------------------------------------------
        # INCIDENT DELAY
        # --------------------------------------------------------

        incident_delay = {
            "LOW": 0,
            "MEDIUM": 3,
            "HIGH": 10,
            "CRITICAL": 25
        }[risk["level"]]

        # --------------------------------------------------------
        # ETA
        # --------------------------------------------------------

        distance = max(
            train["remaining_distance"],
            0.1
        )

        travel_minutes = (
            distance / max(predicted_speed, 10)
        ) * 60

        total_delay = (
            predicted_delay
            + incident_delay
        )

        eta_minutes = (
            travel_minutes
            + total_delay
        )

        arrival = (
            datetime.now()
            + timedelta(
                minutes=eta_minutes
            )
        )

        # --------------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------------

        explanation = self.explain(
            route,
            weather,
            fused_speed,
            predicted_speed,
            sensor_confidence,
            risk
        )

        return {
            "timestamp": now(),
            "train_id": train["train_id"],
            "route": route["name"],
            "actual_speed": fused_speed,
            "predicted_speed": predicted_speed,
            "speed_error": speed_error,
            "speed_uncertainty": speed_uncertainty,
            "speed_confidence": clamp(
                100 - speed_uncertainty * 3,
                0,
                99
            ),
            "predicted_delay": predicted_delay,
            "delay_uncertainty": delay_uncertainty,
            "eta_minutes": eta_minutes,
            "arrival": arrival.isoformat(
                timespec="seconds"
            ),
            "weather": weather,
            "weather_limit": weather_limit,
            "sensor_confidence": sensor_confidence,
            "animal_risk": animal_risk,
            "obstacle_risk": obstacle_risk,
            "anomaly": anomaly,
            "risk": risk,
            "explanation": explanation,
            "speed_models": speed_models.tolist(),
            "speed_weights": self.speed_ai.weights.tolist(),
            "delay_weights": self.delay_ai.weights.tolist()
        }

    # ------------------------------------------------------------
    # ANOMALY
    # ------------------------------------------------------------

    def _anomaly_check(self, X, speed_error):
        """
        Lightweight adaptive anomaly rule.
        In the production version, train a dedicated anomaly model
        using historical telemetry.
        """

        if abs(speed_error) > 30:
            return True

        if np.any(~np.isfinite(X)):
            return True

        return False

    # ------------------------------------------------------------
    # DELAYED FEEDBACK
    # ------------------------------------------------------------

    def learn(
        self,
        X,
        predicted_speed,
        actual_speed,
        predicted_delay,
        actual_delay,
        sample_weight=1.0
    ):
        """
        This is the critical adaptive-learning loop.

        Prediction
            ->
        Real outcome
            ->
        Error
            ->
        Model update
            ->
        Ensemble reweighting
            ->
        Drift detection
        """

        # Get model-level predictions before updating.
        _, _, speed_models = self.speed_ai.predict(X)
        _, _, delay_models = self.delay_ai.predict(X)

        speed_drift, speed_errors = (
            self.speed_ai.learn_from_error(
                speed_models,
                actual_speed
            )
        )

        delay_drift, delay_errors = (
            self.delay_ai.learn_from_error(
                delay_models,
                actual_delay
            )
        )

        # Higher-confidence observations should influence learning more.
        self.speed_ai.update(
            X,
            actual_speed,
            sample_weight=sample_weight
        )

        self.delay_ai.update(
            X,
            actual_delay,
            sample_weight=sample_weight
        )

        self.speed_ai.save_state(
            self.state_path_speed
        )

        self.delay_ai.save_state(
            self.state_path_delay
        )

        return {
            "speed_error": actual_speed - predicted_speed,
            "delay_error": actual_delay - predicted_delay,
            "speed_drift": speed_drift,
            "delay_drift": delay_drift,
            "speed_model_mae": speed_errors.tolist(),
            "delay_model_mae": delay_errors.tolist(),
            "speed_weights": self.speed_ai.weights.tolist(),
            "delay_weights": self.delay_ai.weights.tolist()
        }

    # ------------------------------------------------------------
    # EXPLAINABLE AI
    # ------------------------------------------------------------

    def explain(
        self,
        route,
        weather,
        actual_speed,
        predicted_speed,
        sensor_confidence,
        risk
    ):
        factors = []

        if weather["rain"] > 2:
            factors.append(
                f"rain={weather['rain']:.1f}mm"
            )

        if weather["wind"] > 30:
            factors.append(
                f"wind={weather['wind']:.1f}km/h"
            )

        if weather["visibility"] < 5000:
            factors.append(
                f"visibility={weather['visibility']:.0f}m"
            )

        if route["congestion"] > 0.6:
            factors.append(
                "high route congestion"
            )

        if route["maintenance_risk"] > 0.2:
            factors.append(
                "elevated infrastructure risk"
            )

        if abs(actual_speed - predicted_speed) > 15:
            factors.append(
                "large speed deviation"
            )

        if sensor_confidence < 70:
            factors.append(
                "sensor disagreement"
            )

        if not factors:
            factors.append(
                "normal operating conditions"
            )

        return factors


# ================================================================
# SYNTHETIC LIVE TRAIN
# ================================================================

def create_train():
    return {
        "train_id": "RAILAI-101",
        "route": "LKO_BARABANKI",

        "latitude": 26.8467,
        "longitude": 80.9462,

        "scheduled_speed": 105,
        "current_delay": 8,

        "remaining_distance": 120,
        "remaining_stops": 4,

        "station_congestion": 0.45,

        "update_seconds": 10,

        "previous_speed": 70
    }


# ================================================================
# DEMO
# ================================================================

def run_demo():
    print("\n" + "=" * 80)
    print("RAILAI v5 - ADAPTIVE MULTIMODAL RAILWAY AI")
    print("=" * 80)

    ai = RailAI()
    ai.bootstrap()

    train = create_train()

    print(
        "\nThe model will intentionally encounter changing conditions."
    )

    print(
        "Watch the ensemble weights and prediction errors adapt."
    )

    for step in range(1, 31):
        print("\n" + "-" * 80)
        print(f"LIVE CYCLE {step}/30")
        print("-" * 80)

        # Simulated environment regime changes.
        if step < 10:
            train["current_delay"] += random.uniform(-0.5, 1.5)
            actual_speed = random.uniform(80, 100)

        elif step < 20:
            # Heavy weather regime
            train["current_delay"] += random.uniform(1, 4)
            actual_speed = random.uniform(55, 80)

        else:
            # Incident-like regime
            train["current_delay"] += random.uniform(2, 6)
            actual_speed = random.uniform(35, 65)

        train["current_delay"] = max(
            0,
            train["current_delay"]
        )

        result = ai.process_train(
            train,
            actual_speed=actual_speed
        )

        print(
            f"Actual speed       : "
            f"{result['actual_speed']:.1f} km/h"
        )

        print(
            f"AI predicted speed : "
            f"{result['predicted_speed']:.1f} km/h"
        )

        print(
            f"Speed error        : "
            f"{result['speed_error']:+.1f} km/h"
        )

        print(
            f"AI confidence      : "
            f"{result['speed_confidence']:.1f}%"
        )

        print(
            f"Predicted delay    : "
            f"{result['predicted_delay']:.1f} min"
        )

        print(
            f"Dynamic ETA        : "
            f"{result['eta_minutes']:.1f} min"
        )

        print(
            f"Risk               : "
            f"{result['risk']['level']} "
            f"({result['risk']['score']:.0f}/100)"
        )

        print(
            f"Sensor confidence  : "
            f"{result['sensor_confidence']:.1f}%"
        )

        print(
            f"Speed model weights: "
            f"{np.round(result['speed_weights'], 3)}"
        )

        print(
            "Why:",
            ", ".join(result["explanation"])
        )

        # --------------------------------------------------------
        # Delayed ground truth
        # --------------------------------------------------------
        #
        # In a real system, actual_delay comes later from actual
        # train movement/arrival records. Here we simulate it.

        actual_delay = max(
            0,
            train["current_delay"]
            + random.uniform(-2, 5)
        )

        X = make_features(
            train,
            ROUTES[train["route"]],
            result["weather"],
            result["actual_speed"],
            0,
            result["animal_risk"],
            result["obstacle_risk"],
            result["sensor_confidence"] / 100
        )

        learning = ai.learn(
            X,
            result["predicted_speed"],
            result["actual_speed"],
            result["predicted_delay"],
            actual_delay,
            sample_weight=1.0
        )

        print(
            f"Learning -> "
            f"speed error={learning['speed_error']:+.1f}, "
            f"delay error={learning['delay_error']:+.1f}"
        )

        if learning["speed_drift"]:
            print(
                "!!! CONCEPT DRIFT DETECTED: "
                "speed environment changed."
            )

        if learning["delay_drift"]:
            print(
                "!!! CONCEPT DRIFT DETECTED: "
                "delay environment changed."
            )

        # Move train.
        train["previous_speed"] = (
            result["actual_speed"]
        )
        train["remaining_distance"] = max(
            0,
            train["remaining_distance"]
            - result["actual_speed"]
            * train["update_seconds"]
            / 3600
        )

        time.sleep(0.35)

    print("\n" + "=" * 80)
    print("ADAPTIVE DEMO COMPLETE")
    print("=" * 80)

    print(
        f"Speed updates: "
        f"{ai.speed_ai.total_updates}"
    )

    print(
        f"Delay updates: "
        f"{ai.delay_ai.total_updates}"
    )

    print(
        f"Speed drift events: "
        f"{ai.speed_ai.drift_events}"
    )

    print(
        f"Delay drift events: "
        f"{ai.delay_ai.drift_events}"
    )

    print(
        "\nModel state saved in:",
        MODEL_DIR
    )


# ================================================================
# OPTIONAL CAMERA DEMO
# ================================================================

def run_camera():
    try:
        import cv2
    except ImportError:
        print(
            "Install OpenCV with: "
            "pip install opencv-python"
        )
        return

    ai = RailAI()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    print(
        "Camera mode active. Press Q to exit."
    )

    while True:
        ok, frame = camera.read()

        if not ok:
            break

        detections = ai.vision.analyze(frame)

        animal_risk, obstacle_risk = (
            risk_from_detections(
                detections
            )
        )

        for d in detections:
            x1, y1, x2, y2 = [
                int(v)
                for v in d["box"]
            ]

            label = (
                f"{d['object']} "
                f"{d['confidence']:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if animal_risk > 0.6:
            cv2.putText(
                frame,
                "ANIMAL HAZARD",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        if obstacle_risk > 0.6:
            cv2.putText(
                frame,
                "OBSTACLE HAZARD",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        cv2.imshow(
            "RailAI Vision",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


# ================================================================
# MENU
# ================================================================

def main():
    print(
        """
============================================================

RAILAI v5

1  -> Adaptive railway simulation
2  -> Camera/object detection
3  -> Exit

============================================================
"""
    )

    choice = input(
        "Choose: "
    ).strip()

    if choice == "1":
        run_demo()

    elif choice == "2":
        run_camera()

    else:
        print("Exiting RailAI.")


if __name__ == "__main__":
    main()
