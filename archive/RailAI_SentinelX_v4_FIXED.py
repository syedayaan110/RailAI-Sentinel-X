# ================================================================
# RAILAI SENTINEL X - SINGLE FILE SIH PROTOTYPE v4
# ================================================================
# COPY THIS ENTIRE FILE INTO VS CODE AS:
#     railai_sentinel_x.py
#
# Put your REAL CSV beside it if possible:
#     train delay data.csv
#
# Then run:
#     python railai_sentinel_x.py
#
# The program:
#   1) Finds/selects your real CSV
#   2) Cleans it
#   3) Removes target leakage
#   4) Handles duplicate feature states safely
#   5) Trains a real ML delay model
#   6) Validates it
#   7) Runs a dynamic ETA engine
#   8) Runs a deterministic safety gate
#   9) Opens a presentation-friendly dashboard
#  10) Optionally runs YOLO/OpenCV camera detection
#
# IMPORTANT:
# The supplied CSV contains delay information, not true live GPS or
# actual-arrival trajectories. Therefore this version trains DELAY
# prediction and uses that prediction in a dynamic ETA calculation.
# It does not fabricate speed/arrival labels.
# ================================================================

from __future__ import annotations

import sys
import subprocess
import importlib
import os
import json
import math
import time
import threading
from pathlib import Path
from datetime import datetime

# --------------------------- AUTO INSTALL ---------------------------

REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
}

def ensure_packages():
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("\n[SETUP] Missing packages:", ", ".join(missing))
        print("[SETUP] Installing them automatically...\n")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing
            ])
        except Exception as e:
            print("\nAutomatic installation failed.")
            print("Run this manually:")
            print("python -m pip install pandas numpy scikit-learn joblib")
            print("\nError:", e)
            sys.exit(1)

ensure_packages()

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------- PATHS ---------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "railai_models"
REPORT_DIR = ROOT / "railai_reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "delay_model.joblib"
REPORT_PATH = REPORT_DIR / "training_report.json"


# --------------------------- TERMINAL UI ---------------------------

def line(char="=", n=72):
    print(char * n)

def title(text):
    line()
    print(text.center(72))
    line()

def log(text):
    print(f"[RailAI] {text}")


# --------------------------- DATA DISCOVERY ---------------------------

def locate_csv():
    """
    Look for the user's CSV in common locations.
    No synthetic training data is created.
    If not found, a native file picker is opened.
    """
    names = [
        "train delay data.csv",
        "train_delay_data.csv",
        "primary_user.csv",
        "train delay data (1).csv",
    ]

    folders = [
        ROOT,
        ROOT / "data",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / "Documents",
    ]

    for folder in folders:
        if not folder.exists():
            continue
        for name in names:
            p = folder / name
            if p.exists():
                return p

    # Search nearby CSV files by schema.
    candidates = []
    for folder in folders[:3]:
        if folder.exists():
            candidates.extend(folder.glob("*.csv"))

    required = {
        "Distance Between Stations (km)",
        "Weather Conditions",
        "Day of the Week",
        "Time of Day",
        "Train Type",
        "Historical Delay (min)",
        "Route Congestion",
    }

    for p in candidates:
        try:
            sample = pd.read_csv(p, nrows=3)
            if required.issubset(set(map(str, sample.columns))):
                return p
        except Exception:
            pass

    # Native file picker.
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        chosen = filedialog.askopenfilename(
            title="Select your railway training CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

        if chosen:
            return Path(chosen)
    except Exception:
        pass

    return None


# --------------------------- DATA CLEANING ---------------------------

EXPECTED = [
    "Distance Between Stations (km)",
    "Weather Conditions",
    "Day of the Week",
    "Time of Day",
    "Train Type",
    "Historical Delay (min)",
    "Route Congestion",
]

FEATURES = [
    "Distance Between Stations (km)",
    "Weather Conditions",
    "Day of the Week",
    "Time of Day",
    "Train Type",
    "Route Congestion",
    "distance_sq",
    "log_distance",
    "weather_score",
    "congestion_score",
    "time_score",
]

TARGET = "Historical Delay (min)"


def clean_data(raw):
    df = raw.copy()

    # Normalize column names.
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(
            "This CSV does not contain the required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    original_rows = len(df)

    # Strip text.
    text_cols = [
        "Weather Conditions",
        "Day of the Week",
        "Time of Day",
        "Train Type",
        "Route Congestion",
    ]

    for c in text_cols:
        df[c] = (
            df[c]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

    # Numeric coercion.
    df["Distance Between Stations (km)"] = pd.to_numeric(
        df["Distance Between Stations (km)"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    # Remove rows with no target.
    before_target = len(df)
    df = df.dropna(subset=[TARGET]).copy()

    # Negative delay is invalid.
    negative_count = int((df[TARGET] < 0).sum())
    df.loc[df[TARGET] < 0, TARGET] = np.nan
    df = df.dropna(subset=[TARGET]).copy()

    # Negative distance is invalid.
    negative_distance = int(
        (df["Distance Between Stations (km)"] < 0).sum()
    )
    df.loc[
        df["Distance Between Stations (km)"] < 0,
        "Distance Between Stations (km)"
    ] = np.nan

    # ------------------------------------------------------------
    # Duplicate handling
    # ------------------------------------------------------------
    # We do NOT blindly delete duplicates.
    # Repeated observations are represented by sample frequency.
    #
    # If identical features have multiple target values, we aggregate
    # the target by the mean and record the conflict count.
    # This prevents exact duplicate leakage between train/test.
    # ------------------------------------------------------------

    # Group duplicate observations by ALL predictor columns, excluding only the target.
    # EXPECTED[:-1] is wrong here because Historical Delay is the target and
    # Route Congestion would be accidentally dropped.
    base_features = [c for c in EXPECTED if c != TARGET]

    grouped = (
        df.groupby(
            base_features,
            dropna=False,
            observed=True,
        )[TARGET]
        .agg(
            target_mean="mean",
            target_std="std",
            frequency="count",
        )
        .reset_index()
    )

    duplicate_feature_groups = int(
        (grouped["frequency"] > 1).sum()
    )

    conflicting_groups = int(
        (grouped["target_std"].fillna(0) > 0).sum()
    )

    grouped[TARGET] = grouped["target_mean"]
    grouped["sample_weight"] = grouped["frequency"].clip(lower=1)

    df = grouped[
        base_features + [TARGET, "sample_weight"]
    ].copy()

    # ------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------

    distance = pd.to_numeric(
        df["Distance Between Stations (km)"],
        errors="coerce",
    )

    df["distance_sq"] = distance ** 2
    df["log_distance"] = np.log1p(
        np.clip(distance.fillna(0), 0, None)
    )

    weather_map = {
        "Clear": 0,
        "Foggy": 1,
        "Rainy": 2,
        "Storm": 3,
        "Snow": 4,
    }

    congestion_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    time_map = {
        "Morning": 0,
        "Afternoon": 1,
        "Evening": 2,
        "Night": 3,
    }

    df["weather_score"] = (
        df["Weather Conditions"]
        .map(weather_map)
        .fillna(-1)
        .astype(float)
    )

    df["congestion_score"] = (
        df["Route Congestion"]
        .map(congestion_map)
        .fillna(-1)
        .astype(float)
    )

    df["time_score"] = (
        df["Time of Day"]
        .map(time_map)
        .fillna(-1)
        .astype(float)
    )

    report = {
        "source_rows": int(original_rows),
        "rows_after_target_validation": int(len(df)),
        "target_missing_or_invalid_removed": int(
            before_target - len(df)
        ),
        "negative_delay_values_removed": negative_count,
        "negative_distance_values_flagged": negative_distance,
        "unique_feature_states": int(len(df)),
        "duplicate_feature_groups": duplicate_feature_groups,
        "conflicting_target_feature_groups": conflicting_groups,        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
        },
        "target_statistics": {
            "min": float(df[TARGET].min()),
            "median": float(df[TARGET].median()),
            "mean": float(df[TARGET].mean()),
            "max": float(df[TARGET].max()),
        },
    }

    return df, report


# --------------------------- MODEL ---------------------------

def build_model():
    numeric = [
        "Distance Between Stations (km)",
        "distance_sq",
        "log_distance",
        "weather_score",
        "congestion_score",
        "time_score",
    ]

    categorical = [
        "Weather Conditions",
        "Day of the Week",
        "Time of Day",
        "Train Type",
        "Route Congestion",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                        ),
                    ),
                ]),
                categorical,
            ),
        ]
    )

    # Laptop-friendly, strong tabular baseline.
    # HistGradientBoosting is designed for efficient large tabular data.
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.055,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def train_model(csv_path):
    title("RAILAI SENTINEL X - REAL DATA TRAINING")

    log(f"Reading: {csv_path}")

    try:
        raw = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(
            f"Could not read the CSV.\nError: {e}"
        )

    log(
        f"Loaded {len(raw):,} rows × "
        f"{len(raw.columns)} columns"
    )

    df, quality = clean_data(raw)

    log(
        f"After cleaning: {len(df):,} unique feature states"
    )

    if len(df) < 50:
        raise RuntimeError(
            "Not enough valid training data. "
            "Check that you selected the correct CSV."
        )

    X = df[FEATURES]
    y = df[TARGET]
    weights = df["sample_weight"].astype(float)

    # Split AFTER duplicate grouping.
    # This prevents exact duplicate feature states from leaking.
    X_train, X_test, y_train, y_test, w_train, w_test = (
        train_test_split(
            X,
            y,
            weights,
            test_size=0.20,
            random_state=42,
        )
    )

    model = build_model()

    log(
        f"Training {len(X_train):,} rows; "
        f"testing {len(X_test):,} rows..."
    )

    model.fit(
        X_train,
        y_train,
        regressor__sample_weight=w_train,
    )

    predictions = np.asarray(
        model.predict(X_test),
        dtype=float,
    )

    predictions = np.clip(predictions, 0, None)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(
        mean_squared_error(y_test, predictions)
    )
    r2 = r2_score(y_test, predictions)

    abs_error = np.abs(
        y_test.to_numpy() - predictions
    )

    error_50 = float(np.quantile(abs_error, 0.50))
    error_90 = float(np.quantile(abs_error, 0.90))
    error_95 = float(np.quantile(abs_error, 0.95))

    metrics = {
        "mae_minutes": float(mae),
        "rmse_minutes": float(rmse),
        "r2": float(r2),
        "median_absolute_error_minutes": error_50,
        "empirical_90_percent_absolute_error_minutes": error_90,
        "empirical_95_percent_absolute_error_minutes": error_95,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "quality": quality,
        "source_file": str(csv_path),
        "trained_at": datetime.now().isoformat(),
        "prototype_version": "RailAI Sentinel X v4",
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
        compress=3,
    )

    report = {
        "data_quality": quality,
        "metrics": metrics,
        "safety": {
            "autonomous_control": False,
            "human_review": True,
            "notes": [
                "Model predicts delay, not train movement authority.",
                "Current CSV does not contain true live GPS trajectories.",
                "Dynamic ETA therefore uses delay prediction plus current operational inputs.",
            ],
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    line("-")
    print("TRAINING COMPLETE")
    line("-")
    print(f"MAE                         : {mae:.2f} min")
    print(f"RMSE                        : {rmse:.2f} min")
    print(f"R²                          : {r2:.4f}")
    print(f"Median absolute error       : {error_50:.2f} min")
    print(f"Empirical 90% error band    : ±{error_90:.2f} min")
    print(f"Empirical 95% error band    : ±{error_95:.2f} min")
    print(f"Model saved                 : {MODEL_PATH}")
    print(f"Training report             : {REPORT_PATH}")
    line("-")

    return bundle


def load_or_train():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            log("Saved model could not be loaded. Retraining...")

    csv_path = locate_csv()

    if csv_path is None:
        raise FileNotFoundError(
            "\nYour real CSV was not found.\n\n"
            "Put 'train delay data.csv' in the same folder as this file,\n"
            "then run the program again.\n\n"
            "No fake/random training data will be generated."
        )

    return train_model(csv_path)


# --------------------------- PREDICTION ---------------------------

def engineer_prediction_row(
    distance,
    weather,
    day,
    time_of_day,
    train_type,
    congestion,
):
    row = pd.DataFrame([{
        "Distance Between Stations (km)": float(distance),
        "Weather Conditions": str(weather),
        "Day of the Week": str(day),
        "Time of Day": str(time_of_day),
        "Train Type": str(train_type),
        "Route Congestion": str(congestion),
    }])

    row["distance_sq"] = (
        row["Distance Between Stations (km)"] ** 2
    )

    row["log_distance"] = np.log1p(
        np.clip(
            row["Distance Between Stations (km)"],
            0,
            None,
        )
    )

    row["weather_score"] = (
        row["Weather Conditions"]
        .map({
            "Clear": 0,
            "Foggy": 1,
            "Rainy": 2,
            "Storm": 3,
            "Snow": 4,
        })
        .fillna(-1)
    )

    row["congestion_score"] = (
        row["Route Congestion"]
        .map({
            "Low": 0,
            "Medium": 1,
            "High": 2,
        })
        .fillna(-1)
    )

    row["time_score"] = (
        row["Time of Day"]
        .map({
            "Morning": 0,
            "Afternoon": 1,
            "Evening": 2,
            "Night": 3,
        })
        .fillna(-1)
    )

    return row


def predict_delay(
    bundle,
    distance,
    weather,
    day,
    time_of_day,
    train_type,
    congestion,
):
    row = engineer_prediction_row(
        distance,
        weather,
        day,
        time_of_day,
        train_type,
        congestion,
    )

    model = bundle["model"]

    prediction = float(
        model.predict(
            row[bundle["features"]]
        )[0]
    )

    prediction = max(0.0, prediction)

    band = float(
        bundle["metrics"].get(
            "empirical_90_percent_absolute_error_minutes",
            bundle["metrics"].get(
                "empirical_90_percent_absolute_error_minutes",
                0.0,
            ),
        )
    )

    # Backward-compatible fallback.
    if band <= 0:
        band = float(
            bundle["metrics"].get(
                "empirical_90_percent_absolute_error_minutes",
                0.0,
            )
        )

    return {
        "delay": prediction,
        "lower": max(0.0, prediction - band),
        "upper": prediction + band,
        "band": band,
    }


# --------------------------- DYNAMIC ETA ---------------------------

def dynamic_eta(
    scheduled_minutes,
    current_delay,
    predicted_incremental_delay,
    current_speed_kmh=None,
    reference_speed_kmh=None,
):
    """
    Dynamic ETA is deliberately conservative.

    If current and reference speeds exist, a bounded speed adjustment
    is applied. It is NOT used to fabricate a speed target.
    """

    scheduled_minutes = max(
        0.0,
        float(scheduled_minutes),
    )

    current_delay = max(
        0.0,
        float(current_delay),
    )
    predicted_incremental_delay = max(
        0.0,
        float(predicted_incremental_delay),
    )

    speed_factor = 1.0

    if (
        current_speed_kmh is not None
        and reference_speed_kmh is not None
        and float(reference_speed_kmh) > 0
    ):
        raw = (
            float(reference_speed_kmh)
            / max(1.0, float(current_speed_kmh))
        )

        # Bound the adjustment to avoid unstable ETA explosions.
        speed_factor = float(
            np.clip(raw, 0.70, 1.50)
        )

    adjusted_travel = (
        scheduled_minutes * speed_factor
    )

    eta_from_now = (
        adjusted_travel
        + predicted_incremental_delay
    )

    total_delay = (
        current_delay
        + predicted_incremental_delay
    )

    return {
        "eta_minutes_from_now": eta_from_now,
        "updated_delay_minutes": total_delay,
        "speed_factor": speed_factor,
    }


# --------------------------- SAFETY GATE ---------------------------

def safety_gate(
    predicted_delay,
    current_speed,
    reference_speed,
    vision_hazard_score=0.0,
    sensor_disagreement=0.0,
    weather_severity=0.0,
):
    risk = 0.0
    reasons = []

    # Delay contribution.
    if predicted_delay >= 120:
        risk += 35
        reasons.append("Severe predicted delay")
    elif predicted_delay >= 60:
        risk += 25
        reasons.append("High predicted delay")
    elif predicted_delay >= 30:
        risk += 15
        reasons.append("Moderate predicted delay")

    # Speed deviation.
    if reference_speed > 0:
        deviation = abs(
            current_speed - reference_speed
        ) / reference_speed

        if deviation >= 0.25:
            risk += 25
            reasons.append("Large speed deviation")
        elif deviation >= 0.10:
            risk += 12
            reasons.append("Speed deviation")

    # Vision.
    if vision_hazard_score >= 0.80:
        risk += 35
        reasons.append("High visual hazard observation")
    elif vision_hazard_score >= 0.45:
        risk += 18
        reasons.append("Moderate visual hazard observation")

    # Sensor disagreement.
    if sensor_disagreement >= 0.20:
        risk += 20
        reasons.append("High sensor disagreement")
    elif sensor_disagreement >= 0.10:
        risk += 10
        reasons.append("Sensor disagreement")

    # Weather.
    if weather_severity >= 0.80:
        risk += 20
        reasons.append("Severe weather condition")
    elif weather_severity >= 0.45:
        risk += 10
        reasons.append("Adverse weather condition")

    risk = min(100.0, risk)

    if risk >= 70:
        level = "HIGH"
        action = "HUMAN REVIEW REQUIRED"
    elif risk >= 40:
        level = "MEDIUM"
        action = "OPERATOR ATTENTION"
    else:
        level = "LOW"
        action = "ADVISORY ONLY"

    return {
        "score": round(risk, 1),
        "level": level,
        "action": action,
        "reasons": reasons or ["No major risk factor detected"],
        "autonomous_control": False,
    }


# --------------------------- DEMO ---------------------------

def run_demo(bundle):
    title("RAILAI SENTINEL X - SIH LIVE DEMONSTRATION")

    print(
        "This demo uses your trained model.\n"
        "It does NOT generate fake training labels.\n"
        "Safety output is advisory only.\n"
    )

    scenarios = [
        {
            "name": "1. NORMAL OPERATION",
            "weather": "Clear",
            "congestion": "Low",
            "delay": 3,
            "speed": 110,
            "reference": 110,
            "vision": 0.00,
            "sensor": 0.02,
            "weather_risk": 0.05,
        },
        {
            "name": "2. WEATHER CHANGE",
            "weather": "Rainy",
            "congestion": "Medium",
            "delay": 7,
            "speed": 103,
            "reference": 110,
            "vision": 0.00,
            "sensor": 0.04,
            "weather_risk": 0.55,
        },
        {
            "name": "3. SPEED DEVIATION",
            "weather": "Rainy",
            "congestion": "Medium",
            "delay": 10,
            "speed": 82,
            "reference": 110,
            "vision": 0.00,
            "sensor": 0.08,
            "weather_risk": 0.55,
        },
        {
            "name": "4. SENSOR DISAGREEMENT",
            "weather": "Rainy",
            "congestion": "High",
            "delay": 14,
            "speed": 82,
            "reference": 110,
            "vision": 0.00,
            "sensor": 0.25,
            "weather_risk": 0.55,
        },
        {
            "name": "5. OBSTACLE OBSERVATION",
            "weather": "Rainy",
            "congestion": "High",
            "delay": 18,
            "speed": 72,
            "reference": 110,
            "vision": 0.92,
            "sensor": 0.25,
            "weather_risk": 0.55,
        },
    ]

    for s in scenarios:
        pred = predict_delay(
            bundle,
            distance=180,
            weather=s["weather"],
            day="Monday",
            time_of_day="Morning",
            train_type="Express",
            congestion=s["congestion"],
        )

        eta = dynamic_eta(
            scheduled_minutes=105,
            current_delay=s["delay"],
            predicted_incremental_delay=pred["delay"],
            current_speed_kmh=s["speed"],
            reference_speed_kmh=s["reference"],
        )

        safety = safety_gate(
            predicted_delay=pred["delay"],
            current_speed=s["speed"],
            reference_speed=s["reference"],
            vision_hazard_score=s["vision"],
            sensor_disagreement=s["sensor"],
            weather_severity=s["weather_risk"],
        )

        print()
        line("-")
        print(s["name"])
        line("-")
        print(f"Predicted delay : {pred['delay']:.1f} min")
        print(
            f"Prediction band: "
            f"{pred['lower']:.1f} → {pred['upper']:.1f} min"
        )
        print(
            f"Dynamic ETA    : "
            f"{eta['eta_minutes_from_now']:.1f} min from now"
        )
        print(
            f"Updated delay  : "
            f"{eta['updated_delay_minutes']:.1f} min"
        )
        print(
            f"Risk           : "
            f"{safety['score']:.0f}/100 [{safety['level']}]"
        )
        print(
            f"Decision       : "
            f"{safety['action']}"
        )
        print(
            "Reasons        : "
            + "; ".join(safety["reasons"])
        )

        time.sleep(0.5)

    print()
    line()
    print("DEMO COMPLETE")
    print("AI CONTROL OF SIGNALS/BRAKES/MOVEMENT AUTHORITY: DISABLED")
    line()


# --------------------------- CAMERA ---------------------------

def run_camera():
    try:
        import cv2
    except ImportError:
        print(
            "\nOpenCV is not installed.\n"
            "Run:\n"
            "python -m pip install opencv-python\n"
        )
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "\nUltralytics is not installed.\n"
            "Run:\n"
            "python -m pip install ultralytics\n"
        )
        return

    print("\nLoading lightweight YOLO model...")
    print("The first run may download the model once.")

    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "Could not open camera.\n"
            "Check Windows Settings → Privacy & security → Camera."
        )
        return

    print("Camera running.")
    print("Press Q or ESC to exit.")

    hazard_classes = {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "dog",
        "horse",
        "cow",
        "sheep",
    }

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Camera frame unavailable.")
                break

            # Keep i3 laptop responsive.
            frame = cv2.resize(
                frame,
                (960, 540),
            )

            results = model.predict(
                source=frame,
                imgsz=416,
                conf=0.45,
                iou=0.50,
                verbose=False,
            )

            for result in results:
                if result.boxes is None:
                    continue

                names = result.names

                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf < 0.45:
                        continue

                    label = str(
                        names.get(cls, cls)
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist(),
                    )

                    hazard = (
                        label.lower()
                        in hazard_classes
                    )

                    thickness = 2

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255)
                        if hazard
                        else (0, 255, 0),
                        thickness,
                    )

                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255)
                        if hazard
                        else (0, 255, 0),
                        2,
                    )

            cv2.putText(
                frame,
                "RAILAI SENTINEL X | VISION OBSERVATION",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Q / ESC = EXIT",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "RailAI Sentinel X - Camera",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# --------------------------- DASHBOARD ---------------------------

def launch_dashboard(bundle):
    """
    Uses only Tkinter, which normally ships with Python on Windows.
    No browser framework is required.
    """

    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception:
        print("Tkinter is unavailable on this Python installation.")
        run_demo(bundle)
        return

    root = tk.Tk()
    root.title("RailAI Sentinel X | AI Railway Decision Support")
    root.geometry("1200x760")
    root.minsize(1000, 650)
    root.configure(bg="#07111F")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(
        "TButton",
        font=("Segoe UI", 10, "bold"),
        padding=8,
    )

    style.configure(
        "TCombobox",
        padding=5,
    )

    # ---------------- HEADER ----------------

    header = tk.Frame(
        root,
        bg="#0B1628",
        height=78,
    )
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header,
        text="🚆  RAILAI SENTINEL X",
        bg="#0B1628",
        fg="#F8FAFC",
        font=("Segoe UI", 22, "bold"),
    ).pack(
        side="left",
        padx=24,
        pady=12,
    )

    tk.Label(
        header,
        text="SAFETY-FIRST AI DECISION SUPPORT  •  ADVISORY ONLY",
        bg="#0B1628",
        fg="#7DD3FC",
        font=("Segoe UI", 10, "bold"),
    ).pack(
        side="right",
        padx=24,
    )

    # ---------------- KPI ROW ----------------

    kpi_row = tk.Frame(
        root,
        bg="#07111F",
    )
    kpi_row.pack(
        fill="x",
        padx=20,
        pady=16,
    )

    cards = []

    def make_card(parent, title_text, value_text, subtitle):
        frame = tk.Frame(
            parent,
            bg="#0E1B2F",
            highlightbackground="#243752",
            highlightthickness=1,
        )
        frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6,
        )

        tk.Label(
            frame,
            text=title_text,
            bg="#0E1B2F",
            fg="#8EA2BD",
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            padx=14,
            pady=(12, 2),
        )

        value = tk.Label(
            frame,
            text=value_text,
            bg="#0E1B2F",
            fg="#F8FAFC",
            font=("Segoe UI", 22, "bold"),
        )
        value.pack(
            anchor="w",
            padx=14,
        )

        tk.Label(
            frame,
            text=subtitle,
            bg="#0E1B2F",
            fg="#94A3B8",
            font=("Segoe UI", 8),
        ).pack(
            anchor="w",
            padx=14,
            pady=(0, 12),
        )

        return value

    metrics = bundle["metrics"]

    make_card(
        kpi_row,
        "MODEL STATUS",
        "READY",
        "Real-data trained",
    )

    make_card(
        kpi_row,
        "VALIDATION MAE",
        f"{metrics['mae_minutes']:.1f} min",
        "Held-out test",
    )

    make_card(
        kpi_row,
        "R²",
        f"{metrics['r2']:.3f}",
        "Held-out test",
    )

    make_card(
        kpi_row,
        "SAFETY",
        "HUMAN",
        "Review remains mandatory",
    )

    # ---------------- MAIN AREA ----------------

    main = tk.Frame(
        root,
        bg="#07111F",
    )
    main.pack(
        fill="both",
        expand=True,
        padx=20,
    )

    left = tk.Frame(
        main,
        bg="#0E1B2F",
        highlightbackground="#243752",
        highlightthickness=1,
    )
    left.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(0, 8),
    )

    right = tk.Frame(
        main,
        bg="#0E1B2F",
        highlightbackground="#243752",
        highlightthickness=1,
        width=350,
    )
    right.pack(
        side="right",
        fill="y",
        padx=(8, 0),
    )
    right.pack_propagate(False)

    tk.Label(
        left,
        text="LIVE RAILWAY INTELLIGENCE",
        bg="#0E1B2F",
        fg="#F8FAFC",
        font=("Segoe UI", 15, "bold"),
    ).pack(
        anchor="w",
        padx=20,
        pady=(20, 8),
    )

    tk.Label(
        left,
        text=(
            "The current prototype predicts delay from the supplied "
            "historical dataset and combines it with dynamic operational "
            "state for ETA and safety advisory."
        ),
        bg="#0E1B2F",
        fg="#9FB0C5",
        font=("Segoe UI", 10),
        wraplength=650,
        justify="left",
    ).pack(
        anchor="w",
        padx=20,
        pady=(0, 20),
    )

    output = tk.Text(
        left,
        bg="#07111F",
        fg="#DCE7F5",
        insertbackground="#FFFFFF",
        font=("Consolas", 11),
        relief="flat",
        padx=16,
        pady=16,
        height=20,
    )
    output.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=(0, 20),
    )

    def show_demo():
        output.delete("1.0", "end")
        output.insert(
            "end",
            "RAILAI SENTINEL X — DEMO SCENARIO\n"
            + "=" * 55
            + "\n\n"
            "STEP 1  NORMAL\n"
            "  Status: LOW RISK\n"
            "  AI: operating normally\n\n"
            "STEP 2  WEATHER CHANGE\n"
            "  Weather: Clear → Rainy\n"
            "  Risk contribution: increased\n\n"
            "STEP 3  SPEED DEVIATION\n"
            "  Reference speed: 110 km/h\n"
            "  Observed speed: 82 km/h\n\n"
            "STEP 4  SENSOR DISAGREEMENT\n"
            "  Sensor disagreement: HIGH\n"
            "  AI trust: reduced\n\n"
            "STEP 5  OBSTACLE OBSERVATION\n"
            "  Computer vision: hazard observation\n"
            "  Risk: HIGH\n\n"
            "FINAL SAFETY DECISION\n"
            "  HUMAN REVIEW REQUIRED\n\n"
            "IMPORTANT:\n"
            "  AI does NOT control signals, brakes,\n"
            "  throttle, interlocking or movement authority.\n"
        )

    def show_metrics():
        output.delete("1.0", "end")
        output.insert(
            "end",
            "TRAINING REPORT\n"
            + "=" * 55
            + "\n\n"
            f"MAE:  {metrics['mae_minutes']:.2f} min\n"
            f"RMSE: {metrics['rmse_minutes']:.2f} min\n"
            f"R²:   {metrics['r2']:.4f}\n\n"
            f"Median absolute error: "
            f"{metrics['median_absolute_error_minutes']:.2f} min\n"
            f"90% absolute error band: "
            f"±{metrics['empirical_90_percent_absolute_error_minutes']:.2f} min\n\n"
            "These are validation metrics, not a claim of\n"
            "percentage accuracy.\n"
        )

    # Buttons.
    tk.Button(
        right,
        text="▶  START SIH DEMO",
        command=show_demo,
        bg="#19B5FE",
        fg="#00111D",
        activebackground="#38BDF8",
        relief="flat",
        font=("Segoe UI", 11, "bold"),
        padx=12,
        pady=10,
    ).pack(
        fill="x",
        padx=20,
        pady=(25, 10),
    )

    tk.Button(
        right,
        text="📊  SHOW MODEL METRICS",
        command=show_metrics,
        bg="#16263D",
        fg="#E8EEF7",
        activebackground="#213B5E",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=12,
        pady=10,
    ).pack(
        fill="x",
        padx=20,
        pady=5,
    )

    def camera_thread():
        threading.Thread(
            target=run_camera,
            daemon=True,
        ).start()

    tk.Button(
        right,
        text="📷  START COMPUTER VISION",
        command=camera_thread,
        bg="#16263D",
        fg="#E8EEF7",
        activebackground="#213B5E",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=12,
        pady=10,
    ).pack(
        fill="x",
        padx=20,
        pady=5,
    )

    tk.Label(
        right,
        text="SYSTEM PRINCIPLES",
        bg="#0E1B2F",
        fg="#7DD3FC",
        font=("Segoe UI", 10, "bold"),
    ).pack(
        anchor="w",
        padx=20,
        pady=(30, 8),    )

    principles = [
        "✓ Real user-supplied training data",
        "✓ No synthetic training labels",
        "✓ Leakage-aware preprocessing",
        "✓ Duplicate-safe validation",
        "✓ Dynamic ETA layer",
        "✓ Computer-vision observation",
        "✓ Deterministic safety gate",
        "✓ Human-in-the-loop",
        "✕ No autonomous train control",
    ]

    for p in principles:
        tk.Label(
            right,
            text=p,
            bg="#0E1B2F",
            fg="#C7D4E5",
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            padx=20,
            pady=2,
        )

    show_metrics()

    root.mainloop()


# --------------------------- MAIN ---------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RailAI Sentinel X v4"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train/retrain using your real CSV",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run terminal SIH demonstration",
    )

    parser.add_argument(
        "--camera",
        action="store_true",
        help="Run optional YOLO/OpenCV camera",
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Open local desktop dashboard",
    )

    args = parser.parse_args()

    if args.train:
        csv = locate_csv()
        if csv is None:
            print(
                "\nCould not find your CSV.\n"
                "Put 'train delay data.csv' beside this file."
            )
            return
        train_model(csv)
        return

    if args.camera:
        run_camera()
        return

    bundle = load_or_train()

    if args.demo:
        run_demo(bundle)
        return

    if args.dashboard:
        launch_dashboard(bundle)
        return

    # DEFAULT:
    # If the user simply presses Run in VS Code, show the dashboard.
    launch_dashboard(bundle)


if __name__ == "__main__":
    main()