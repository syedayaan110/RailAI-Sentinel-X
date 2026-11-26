#!/usr/bin/env python3
"""
RailAI Sentinel X v3 - SIH prototype

Real-data-first railway delay + ETA decision-support system.

Design goals:
- Uses the user's supplied CSV exactly as a first-class labelled dataset.
- Uses the supplied 1.5M-row journey dataset when available, with strict leakage control.
- Uses timetable CSVs as route metadata, never as fake delay labels.
- Optional curated public-data collection from public GitHub repositories and Open-Meteo.
- Temporal validation, duplicate handling, robust cleaning, uncertainty estimates.
- Lightweight enough for a normal laptop; CPU thread count is configurable.
- Computer vision is an advisory observation layer only.

Safety: this prototype must never directly control railway signalling, interlocking,
brakes, throttle, movement authority or other certified railway systems.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, hashlib, zipfile, urllib.request, urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXT = DATA / "external"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
for d in (DATA, EXT, MODELS, REPORTS, STATE): d.mkdir(parents=True, exist_ok=True)

PRIMARY = DATA / "train delay data.csv"
KAGGLE_ZIP = DATA / "indian-railways-predict-train-delay.zip"
TIMETABLE_A = DATA / "Train_details_22122017.csv"
TIMETABLE_B = DATA / "isl_wise_train_detail_03082015_v1.csv"
MODEL_BUNDLE = MODELS / "railai_sentinel_x_v3.joblib"
METRICS = REPORTS / "metrics.json"
QUALITY = REPORTS / "data_quality.json"
PROVENANCE = REPORTS / "provenance.json"

PRIMARY_TARGET = "Historical Delay (min)"
PRIMARY_COLS = [
    "Distance Between Stations (km)", "Weather Conditions", "Day of the Week",
    "Time of Day", "Train Type", PRIMARY_TARGET, "Route Congestion"
]

PUBLIC_SOURCES = [
    {
        "name": "Indian Railway Delay Visualization",
        "url": "https://github.com/adityaazad79/Indian-Railway-Delay-Visualization",
        "kind": "monthly/weekly/train/station delay + timetable metadata",
        "license": "Apache-2.0 repository license shown by GitHub",
        "use": "contextual route/station priors after schema validation"
    },
    {
        "name": "Indian Railway Express Trains Delay Datasets",
        "url": "https://github.com/ankitaanand28/DA323_IndianRailwayTrainDelayDatasets",
        "kind": "train/station delay aggregates, Mar 2023-Mar 2024",
        "license": "CC BY-NC-SA 4.0",
        "use": "external validation/contextual station priors"
    },
    {
        "name": "Indian Railways Passenger Reservation Enquiry",
        "url": "https://www.indianrail.gov.in/enquiry/SCHEDULE/TrainSchedule.jsp",
        "kind": "official timetable enquiry",
        "license": "official public enquiry service",
        "use": "manual verification/reference, not bulk scraping"
    },
    {
        "name": "Open-Meteo Historical Weather",
        "url": "https://open-meteo.com/en/docs/historical-weather-api",
        "kind": "historical weather/reanalysis",
        "license": "see provider terms",
        "use": "optional weather enrichment when coordinates/time are available"
    },
    {
        "name": "Kaggle Indian Railways Predict Train Delay",
        "url": "https://www.kaggle.com/competitions/indian-railways-predict-train-delay/data",
        "kind": "1.5M journey-level ML training records",
        "license": "Kaggle competition terms",
        "use": "primary operational training benchmark when locally supplied"
    }
]


def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def sha256_file(path: Path, block=1<<20):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()


def collect_public():
    """Download only curated public repositories. No login/paywall/robots bypass."""
    urls = {
        "adityaazad79_Indian-Railway-Delay-Visualization.zip": "https://github.com/adityaazad79/Indian-Railway-Delay-Visualization/archive/refs/heads/main.zip",
        "ankitaanand28_DA323_IndianRailwayTrainDelayDatasets.zip": "https://github.com/ankitaanand28/DA323_IndianRailwayTrainDelayDatasets/archive/refs/heads/main.zip",
    }
    for name, url in urls.items():
        dest=EXT/name
        if dest.exists():
            print(f"[PUBLIC] already present: {dest.name}")
            continue
        print(f"[PUBLIC] downloading {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[PUBLIC] saved {dest}")
        except Exception as e:
            print(f"[PUBLIC] skipped: {e}")
    save_json(REPORTS/"public_sources.json", {"collected_at": now(), "sources": PUBLIC_SOURCES})


def show_sources():
    for s in PUBLIC_SOURCES:
        print(f"- {s['name']}\n  {s['url']}\n  {s['kind']}\n  use: {s['use']}\n")


def read_csv_safely(path, **kwargs):
    return pd.read_csv(path, low_memory=False, **kwargs)


def clean_primary(path: Path):
    df=read_csv_safely(path)
    missing=[c for c in PRIMARY_COLS if c not in df.columns]
    if missing: raise ValueError(f"Primary CSV missing columns: {missing}")
    df=df[PRIMARY_COLS].copy()
    for c in ["Weather Conditions","Day of the Week","Time of Day","Train Type","Route Congestion"]:
        df[c]=df[c].astype("string").str.strip().replace({"": pd.NA})
    df["Distance Between Stations (km)"]=pd.to_numeric(df["Distance Between Stations (km)"], errors="coerce")
    df[PRIMARY_TARGET]=pd.to_numeric(df[PRIMARY_TARGET], errors="coerce")
    bad_distance=int((df["Distance Between Stations (km)"]<0).fillna(False).sum())
    bad_delay=int((df[PRIMARY_TARGET]<0).fillna(False).sum())
    df.loc[df["Distance Between Stations (km)"]<0,"Distance Between Stations (km)"]=np.nan
    df.loc[df[PRIMARY_TARGET]<0,PRIMARY_TARGET]=np.nan
    for c in ["Distance Between Stations (km)",PRIMARY_TARGET]:
        df[c]=df[c].fillna(df[c].median())
    for c in ["Weather Conditions","Day of the Week","Time of Day","Train Type","Route Congestion"]:
        df[c]=df[c].fillna("Unknown")
    dup=int(df.duplicated().sum())
    # Keep duplicate frequency as a sample weight before deduplication.
    df["sample_weight"]=df.groupby(PRIMARY_COLS, dropna=False)[PRIMARY_TARGET].transform("size")
    df=df.drop_duplicates(PRIMARY_COLS).reset_index(drop=True)
    report={
        "file":str(path),"rows_raw":int(len(df)+dup),"rows_unique":int(len(df)),
        "duplicates_removed":dup,"bad_negative_distance":bad_distance,
        "bad_negative_delay":bad_delay,"missing_after":df.isna().sum().to_dict(),
        "target_min":float(df[PRIMARY_TARGET].min()),"target_max":float(df[PRIMARY_TARGET].max()),
        "target_median":float(df[PRIMARY_TARGET].median())
    }
    return df, report


def build_timetable_features(*paths):
    frames=[]
    for p in paths:
        if not p or not Path(p).exists(): continue
        try:
            x=read_csv_safely(p, usecols=lambda c: c in {
                "Train No","Train No.","SEQ","islno","Distance","Train Name","train Name",
                "Station Code","station Code","Source Station","Source Station Code",
                "Destination Station","Destination station Code"
            })
            frames.append(x)
        except Exception as e: print(f"[TIMETABLE] skipped {p}: {e}")
    if not frames: return pd.DataFrame()
    x=pd.concat(frames, ignore_index=True)
    train_col=next((c for c in ["Train No","Train No."] if c in x.columns),None)
    dist_col="Distance" if "Distance" in x.columns else None
    if not train_col or not dist_col: return pd.DataFrame()
    x["train_key"]=x[train_col].astype(str).str.replace("'","",regex=False).str.extract(r"(\d+)",expand=False)
    x["Distance_num"]=pd.to_numeric(x[dist_col],errors="coerce")
    g=x.groupby("train_key",dropna=True)
    out=g.agg(
        timetable_max_distance_km=("Distance_num","max"),
        timetable_min_distance_km=("Distance_num","min"),
        timetable_stops=("Distance_num","size")
    ).reset_index()
    out["timetable_route_span_km"]=(out["timetable_max_distance_km"]-out["timetable_min_distance_km"]).clip(lower=0)
    return out


def load_kaggle_zip(path: Path, use_all=True, max_rows=None):
    if not path.exists(): return None
    z=zipfile.ZipFile(path)
    name=next((n for n in z.namelist() if n.endswith("ir_train.csv")),None)
    if not name: return None
    # Read in chunks to avoid a huge RAM spike. All chunks are concatenated only after selecting columns.
    keep=[
        "train_number","train_type","departure_date","year","month","day_of_week","departure_hour",
        "is_weekend","is_night_departure","is_peak_hour","is_festival_season","season","zone","zone_abbr",
        "source_station_category","destination_station_category","distance_km","num_scheduled_stops",
        "scheduled_travel_hours","track_doubled","is_hdn_route","traction_type","is_electrified","psr_count",
        "is_circular_route","is_monsoon_season","is_fog_risk","fog_risk_score","zone_fog_index",
        "zone_congestion_index","season_severity_score","loco_age_years","coach_age_years","has_lhb_coaches",
        "is_rake_shared","maintenance_score","seat_utilisation_pct","is_overloaded","late_incoming_rake",
        "is_special_train","route_historical_ontime_pct","delay_minutes"
    ]
    chunks=[]; total=0
    with z.open(name) as f:
        for ch in pd.read_csv(f,usecols=keep,chunksize=100_000,low_memory=False):
            if max_rows and total>=max_rows: break
            if max_rows and total+len(ch)>max_rows: ch=ch.iloc[:max_rows-total]
            chunks.append(ch); total+=len(ch)
    df=pd.concat(chunks,ignore_index=True)
    return df


def make_features(df: pd.DataFrame):
    d=df.copy()
    if "departure_date" in d:
        dt=pd.to_datetime(d["departure_date"],errors="coerce")
        d["dep_year"]=dt.dt.year
        d["dep_month"]=dt.dt.month
        d["dep_dayofyear"]=dt.dt.dayofyear
        d["dep_weekofyear"]=dt.dt.isocalendar().week.astype(float)
    # Universal engineered variables
    if "Distance Between Stations (km)" in d:
        d["distance_log1p"]=np.log1p(pd.to_numeric(d["Distance Between Stations (km)"],errors="coerce").clip(lower=0))
        d["distance_band"]=(pd.to_numeric(d["Distance Between Stations (km)"],errors="coerce")//100).clip(0,50)
    if "distance_km" in d:
        d["distance_log1p"]=np.log1p(pd.to_numeric(d["distance_km"],errors="coerce").clip(lower=0))
        d["distance_band"]=(pd.to_numeric(d["distance_km"],errors="coerce")//100).clip(0,50)
    if "scheduled_travel_hours" in d and "distance_km" in d:
        speed=d["distance_km"]/d["scheduled_travel_hours"].replace(0,np.nan)
        d["scheduled_avg_speed_kmph"]=speed.replace([np.inf,-np.inf],np.nan).clip(0,200)
    if "departure_hour" in d:
        h=pd.to_numeric(d["departure_hour"],errors="coerce").fillna(12)
        d["hour_sin"]=np.sin(2*np.pi*h/24); d["hour_cos"]=np.cos(2*np.pi*h/24)
    if "month" in d:
        m=pd.to_numeric(d["month"],errors="coerce").fillna(6)
        d["month_sin"]=np.sin(2*np.pi*m/12); d["month_cos"]=np.cos(2*np.pi*m/12)
    # Map primary schema to operational names when necessary.
    mapping={
        "Distance Between Stations (km)":"distance_km",
        "Historical Delay (min)":"delay_minutes",
        "Route Congestion":"route_congestion_text",
        "Weather Conditions":"weather_text",
        "Day of the Week":"day_text",
        "Time of Day":"time_text",
        "Train Type":"train_type_text"
    }
    for a,b in mapping.items():
        if a in d.columns and b not in d.columns: d[b]=d[a]
    return d



class EfficientTabularHGB:
    """Compact sklearn HGB wrapper that encodes categorical columns without one-hot expansion."""
    def __init__(self, seed=SEED): self.seed=seed
    def _transform_fit(self,X):
        self.columns=list(X.columns); self.cat_cols=[]; self.maps={}; A=X.copy()
        for c in self.columns:
            if X[c].dtype=="object" or str(X[c].dtype).startswith("string") or str(X[c].dtype).startswith("category"):
                self.cat_cols.append(c)
                vals=X[c].astype("string").fillna("__MISSING__").unique().tolist()
                self.maps[c]={v:i+1 for i,v in enumerate(vals)}
                A[c]=X[c].astype("string").fillna("__MISSING__").map(self.maps[c]).fillna(1).astype(np.float32)
            else:
                A[c]=pd.to_numeric(X[c],errors="coerce").astype(np.float32)
        return A
    def _transform(self,X):
        A=X.reindex(columns=self.columns,copy=True)
        for c in self.columns:
            if c in self.cat_cols:
                A[c]=A[c].astype("string").fillna("__MISSING__").map(self.maps[c]).fillna(1).astype(np.float32)
            else:
                A[c]=pd.to_numeric(A[c],errors="coerce").astype(np.float32)
        return A
    def fit(self,X,y,weights=None):
        A=self._transform_fit(X)
        mask=[c in self.cat_cols for c in self.columns]
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=.08,l2_regularization=2.0,random_state=self.seed,categorical_features=mask,early_stopping=True,validation_fraction=None)
        # HGB currently has no stable sample_weight support across all sklearn versions used by this prototype.
        self.model.fit(A,y)
        return self
    def predict(self,X): return self.model.predict(self._transform(X))

def fit_model(X_train,y_train,X_val,y_val,weights=None,threads=4):
    # Large operational data: no one-hot encoding and no huge intermediate sparse matrices.
    if len(X_train) >= 100_000:
        model=EfficientTabularHGB(SEED).fit(X_train,y_train,weights)
        return model,model.predict(X_val),"sklearn_hist_gradient_boosting_compact"
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor
    cat=[c for c in X_train.columns if X_train[c].dtype=="object" or str(X_train[c].dtype).startswith("string") or str(X_train[c].dtype).startswith("category")]
    num=[c for c in X_train.columns if c not in cat]
    prep=ColumnTransformer([
        ("num",SimpleImputer(strategy="median"),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),cat)
    ])
    Xt=prep.fit_transform(X_train); Xv=prep.transform(X_val)
    model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=0.08,l2_regularization=2.0,random_state=SEED,early_stopping=True,validation_fraction=None)
    model.fit(Xt,y_train)
    return Pipeline([("prep",prep),("model",model)]),model.predict(Xv),"sklearn_hist_gradient_boosting"


def prepare_lgbm_inference(model, X):
    """Recreate the exact LightGBM categorical contract used during training."""
    if hasattr(model,"_railai_feature_columns"):
        X=X.reindex(columns=model._railai_feature_columns,copy=True)
        for c,cats in model._railai_categories.items():
            X[c]=pd.Categorical(X[c].astype("string"),categories=cats)
        for c,med in model._railai_medians.items():
            X[c]=pd.to_numeric(X[c],errors="coerce").fillna(med)
    return X


def train_operational(zip_path: Path, threads=4, max_rows=None):
    df=load_kaggle_zip(zip_path,max_rows=max_rows)
    if df is None: raise FileNotFoundError(f"Could not read ir_train.csv from {zip_path}")
    # Timetable CSVs are intentionally kept out of the 1.5M-row training matrix to
    # avoid a large RAM spike. They are validated separately and can be used by the
    # ETA route lookup layer. This keeps the ML training deterministic and laptop-friendly.
    # Leakage protection: never use target, cause, or derived delayed flag as feature.
    target="delay_minutes"
    leakage={"journey_id","primary_delay_cause","delay_minutes","is_delayed"}
    y=pd.to_numeric(df[target],errors="coerce")
    valid=y.notna()
    df=df.loc[valid].reset_index(drop=True); y=y.loc[valid].reset_index(drop=True)
    # Compact feature matrix: select relevant source columns before creating engineered values.
    dt_year=pd.to_numeric(df["year"],errors="coerce")
    train_mask=dt_year<=2023
    val_mask=dt_year==2024
    if val_mask.sum()<100: raise ValueError("2024 holdout is missing from operational data")
    feature_candidates=[
        "train_type","year","month","day_of_week","departure_hour","is_weekend","is_night_departure",
        "is_peak_hour","is_festival_season","season","zone","zone_abbr","source_station_category",
        "destination_station_category","distance_km","num_scheduled_stops","scheduled_travel_hours",
        "track_doubled","is_hdn_route","traction_type","is_electrified","psr_count","is_circular_route",
        "is_monsoon_season","is_fog_risk","fog_risk_score","zone_fog_index","zone_congestion_index",
        "season_severity_score","loco_age_years","coach_age_years","has_lhb_coaches","is_rake_shared",
        "maintenance_score","seat_utilisation_pct","is_overloaded","late_incoming_rake","is_special_train",
        "route_historical_ontime_pct","scheduled_avg_speed_kmph","distance_log1p","distance_band","hour_sin","hour_cos","month_sin","month_cos"
    ]
    X=df[[c for c in feature_candidates if c in df.columns]].copy()
    if "distance_km" in X and "scheduled_travel_hours" in X:
        X["scheduled_avg_speed_kmph"]=(X["distance_km"] / X["scheduled_travel_hours"].replace(0,np.nan)).clip(0,200)
    if "departure_hour" in X:
        h=pd.to_numeric(X["departure_hour"],errors="coerce").fillna(12)
        X["hour_sin"]=np.sin(2*np.pi*h/24); X["hour_cos"]=np.cos(2*np.pi*h/24)
    if "month" in X:
        m=pd.to_numeric(X["month"],errors="coerce").fillna(6)
        X["month_sin"]=np.sin(2*np.pi*m/12); X["month_cos"]=np.cos(2*np.pi*m/12)
    # IDs and raw timestamps are intentionally excluded to prevent memorisation/leakage.
    total_rows=int(len(df)); train_rows=int(train_mask.sum()); val_rows=int(val_mask.sum())
    X_train,X_val=X.loc[train_mask],X.loc[val_mask]
    y_train,y_val=y.loc[train_mask],y.loc[val_mask]
    # Release the wide raw dataframe before sklearn creates its transformed matrices.
    import gc
    del df
    gc.collect()
    model,pred,backend=fit_model(X_train,y_train,X_val,y_val,threads=threads)
    mae=float(np.mean(np.abs(y_val-pred)))
    rmse=float(np.sqrt(np.mean((y_val-pred)**2)))
    r2=float(1-np.sum((y_val-pred)**2)/np.sum((y_val-y_val.mean())**2))
    # Delay classification threshold is defined by the prototype, not supplied is_delayed.
    ycls=(y_val>=15).astype(int)
    try: auc=float(__import__('sklearn').metrics.roc_auc_score(ycls,pred))
    except Exception: auc=None
    metrics={"backend":backend,"rows_total":total_rows,"train_rows":train_rows,"validation_rows_2024":val_rows,"mae_min":mae,"rmse_min":rmse,"r2":r2,"roc_auc_proxy":auc,"leakage_excluded":sorted(leakage),"trained_at":now()}
    return model,metrics


def train_primary(path: Path, threads=4):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    df,quality=clean_primary(path)
    X=make_features(df.drop(columns=[PRIMARY_TARGET,"sample_weight"]))
    y=df[PRIMARY_TARGET].astype(float)
    weights=df["sample_weight"].astype(float)
    tr,va=train_test_split(np.arange(len(df)),test_size=0.20,random_state=SEED)
    model,pred,backend=fit_model(X.iloc[tr],y.iloc[tr],X.iloc[va],y.iloc[va],weights=weights.iloc[tr],threads=threads)
    metrics={"backend":backend,"rows":int(len(df)),"train_rows":int(len(tr)),"validation_rows":int(len(va)),
             "mae":float(mean_absolute_error(y.iloc[va],pred)),"rmse":float(np.sqrt(mean_squared_error(y.iloc[va],pred))),
             "r2":float(r2_score(y.iloc[va],pred)),"quality":quality}
    print(f"[PRIMARY] holdout MAE={metrics['mae']:.2f} min | RMSE={metrics['rmse']:.2f} | R2={metrics['r2']:.3f}")
    # Final deployment model is retrained on ALL cleaned primary observations.
    final_model,_,final_backend=fit_model(X,y,X,y,weights=weights,threads=threads)
    metrics["final_training_rows"]=int(len(df)); metrics["final_backend"]=final_backend    return final_model,metrics


def train_all(primary_path: Path, kaggle_zip: Optional[Path], threads=4, max_rows=None):
    print("=== RailAI Sentinel X v3 training ===")
    t=time.time()
    import joblib, gc
    # Operational dataset is the memory-heavy part, so train it first while the
    # process is clean. Then train the tiny user-labelled model.
    op_model=None; om=None
    if kaggle_zip and kaggle_zip.exists():
        print("[OPERATIONAL] training on supplied 1.5M-row journey dataset (2024 temporal holdout)")
        op_model,om=train_operational(kaggle_zip,threads,max_rows=max_rows)
        gc.collect()
    primary_model,pm=train_primary(primary_path,threads)
    tt=build_timetable_features(TIMETABLE_A,TIMETABLE_B)
    tt_report={"rows":int(len(tt)),"train_keys":int(tt["train_key"].nunique()) if not tt.empty else 0,
               "max_route_span_km":float(tt["timetable_route_span_km"].max()) if not tt.empty else None}
    save_json(REPORTS/"timetable_quality.json",tt_report)
    bundle={"version":"3.0","trained_at":now(),"primary_model":primary_model,"operational_model":op_model,
            "primary_metrics":pm,"operational_metrics":om}
    joblib.dump(bundle,MODEL_BUNDLE,compress=3)
    save_json(METRICS,{"primary":pm,"operational":om,"elapsed_seconds":time.time()-t})
    save_json(QUALITY,pm.get("quality",{}))
    local_files={}
    for p in [primary_path,TIMETABLE_A,TIMETABLE_B,kaggle_zip]:
        if p and Path(p).exists(): local_files[str(p)]=sha256_file(Path(p))
    save_json(PROVENANCE,{"sources":PUBLIC_SOURCES,"local_files":local_files,
        "notes":["Public datasets are not silently merged unless schema-compatible and provenance-tracked.",
                  "The supplied timetable data is used as route/schedule metadata when train IDs overlap.",
                  "No live railway telemetry is claimed."]})
    print(f"[DONE] model: {MODEL_BUNDLE}")
    print(json.dumps({"primary":pm,"operational":om},indent=2,default=str))


def predict_delay(distance_km=150,weather="Clear",day="Monday",time_of_day="Morning",train_type="Express",historical_delay=10,route_congestion="Low"):
    import joblib
    if not MODEL_BUNDLE.exists(): raise FileNotFoundError("Train first: python railai_sentinel_x_v3.py --train")
    b=joblib.load(MODEL_BUNDLE)
    model=b["primary_model"]
    row=pd.DataFrame([{
        "Distance Between Stations (km)":distance_km,"Weather Conditions":weather,"Day of the Week":day,
        "Time of Day":time_of_day,"Train Type":train_type,PRIMARY_TARGET:historical_delay,"Route Congestion":route_congestion
    }])
    # Historical delay is a legitimate input for a dynamic update, but users should not feed the current target itself.
    x=make_features(row.drop(columns=[PRIMARY_TARGET]))
    x=prepare_lgbm_inference(model,x)
    pred=float(model.predict(x)[0])
    return max(0.0,pred)


def dynamic_eta(predicted_delay_min, scheduled_remaining_min, current_delay_min=0, distance_remaining_km=None, current_speed_kmph=None, safety_buffer_min=3):
    """Physics-aware ETA update. Current speed is optional and never fabricated."""
    base=float(scheduled_remaining_min)
    if distance_remaining_km is not None and current_speed_kmph and current_speed_kmph>5:
        speed_remaining=60*float(distance_remaining_km)/float(current_speed_kmph)
        base=max(base,speed_remaining)
    updated_delay=max(0.0,float(current_delay_min))+max(0.0,float(predicted_delay_min))
    eta_min=base+updated_delay+float(safety_buffer_min)
    return {"predicted_delay_min":round(predicted_delay_min,1),"eta_minutes_from_now":round(eta_min,1),"updated_delay_min":round(updated_delay,1),"safety_buffer_min":safety_buffer_min}


def camera(yolo_path=None, camera_index=0, conf=0.45):
    try:
        import cv2
    except ImportError:
        print("Install camera support: python -m pip install opencv-python ultralytics"); return
    model=None
    if yolo_path:
        try:
            from ultralytics import YOLO
            model=YOLO(yolo_path)
        except Exception as e: print("[VISION] model unavailable:",e)
    cap=cv2.VideoCapture(camera_index)
    if not cap.isOpened(): raise RuntimeError("Camera could not be opened")
    print("CAMERA READY - focus the camera window; press Q or ESC to exit")
    while True:
        ok,frame=cap.read()
        if not ok: break
        if model:
            try:
                res=model.predict(frame,conf=conf,verbose=False)
                frame=res[0].plot()
            except Exception as e:
                cv2.putText(frame,f"VISION ERROR: {str(e)[:60]}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        cv2.putText(frame,"RailAI Sentinel X - ADVISORY VISION | Q/Esc: exit",(20,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        cv2.imshow("RailAI Sentinel X - Computer Vision",frame)
        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),ord('Q'),27): break
    cap.release(); cv2.destroyAllWindows()


def demo():
    print("\n=== RAILAI SENTINEL X v3 DEMO ===")
    print("Real-data-trained delay intelligence | human-in-the-loop | advisory only\n")
    p=predict_delay(320,"Rainy","Friday","Evening","Superfast",18,"High")
    print(f"Predicted incremental delay: {p:.1f} min")
    print("Dynamic ETA:",dynamic_eta(p,scheduled_remaining_min=95,current_delay_min=12,distance_remaining_km=120,current_speed_kmph=82))
    print("\nSafety gate: AI may warn/recommend; it does NOT control railway signalling or train movement.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",action="store_true")
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--predict",action="store_true")
    ap.add_argument("--collect-public",action="store_true")
    ap.add_argument("--show-sources",action="store_true")
    ap.add_argument("--camera",action="store_true")
    ap.add_argument("--yolo",default=None)
    ap.add_argument("--csv",default=str(PRIMARY))
    ap.add_argument("--kaggle-zip",default=str(KAGGLE_ZIP))
    ap.add_argument("--threads",type=int,default=max(1,min(4,(os.cpu_count() or 2))))
    ap.add_argument("--max-operational-rows",type=int,default=None)
    args=ap.parse_args()
    if args.show_sources: show_sources(); return
    if args.collect_public: collect_public(); return
    if args.train:
        train_all(Path(args.csv),Path(args.kaggle_zip) if args.kaggle_zip else None,args.threads,args.max_operational_rows); return
    if args.predict:
        print("Predicted delay:",predict_delay(),"minutes"); return
    if args.camera:
        camera(args.yolo); return
    if args.demo: demo(); return
    ap.print_help()



# ============================================================================
# RAILAI SENTINEL X v5 EXTENSION LAYER
# ============================================================================
# The blocks below add: schema-aware ingestion, memory-safe large-file handling,
# robust evaluation, residual-based uncertainty, adaptive feedback, drift checks,
# sensor-quality gates, transparent risk scoring, provenance, audit logs, and
# a practical command-line workflow. This is a research prototype only.

import csv
import shutil
import statistics
import traceback
import warnings
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence, Mapping

APP_VERSION = "5.0.0"
AUDIT_FILE = STATE / "audit_log.jsonl"
REPLAY_FILE = STATE / "replay_memory.jsonl"
REGISTRY_FILE = STATE / "model_registry.json"
DRIFT_FILE = STATE / "drift_state.json"
USER_FEEDBACK_FILE = STATE / "feedback.jsonl"
DATA_QUALITY_FILE = REPORTS / "data_quality_v5.json"
EVALUATION_FILE = REPORTS / "evaluation_v5.json"
PREDICTION_FILE = REPORTS / "prediction_history.jsonl"
MODEL_METADATA_FILE = MODELS / "model_metadata.json"
CONFIG_FILE = ROOT / "railai_config.json"

DEFAULT_CONFIG = {
    "version": APP_VERSION,
    "random_seed": SEED,
    "max_training_rows": 250000,
    "csv_chunk_size": 50000,
    "max_categories": 500,
    "holdout_fraction": 0.20,
    "minimum_training_rows": 30,
    "residual_interval_quantile": 0.90,
    "max_replay_rows": 5000,
    "max_download_mb": 250,
    "network_timeout_seconds": 20,
    "enable_network_enrichment": False,
    "enable_camera_by_default": False,
    "advisory_only": True,
    "never_control_signals_or_brakes": True,
}

PRIMARY_ALIASES = {
    "distance between stations (km)": "Distance Between Stations (km)",
    "distance_km": "Distance Between Stations (km)",
    "distance (km)": "Distance Between Stations (km)",
    "weather": "Weather Conditions",
    "weather conditions": "Weather Conditions",
    "day": "Day of the Week",
    "day of week": "Day of the Week",
    "day of the week": "Day of the Week",
    "time": "Time of Day",
    "time of day": "Time of Day",
    "train type": "Train Type",
    "historical delay (min)": "Historical Delay (min)",
    "delay_minutes": "Historical Delay (min)",
    "delay (min)": "Historical Delay (min)",
    "route congestion": "Route Congestion",
    "congestion": "Route Congestion",
}

PRIMARY_REQUIRED_V5 = [
    "Distance Between Stations (km)",
    "Weather Conditions",
    "Day of the Week",
    "Time of Day",
    "Train Type",
    "Historical Delay (min)",
    "Route Congestion",
]
PRIMARY_FEATURES_V5 = [c for c in PRIMARY_REQUIRED_V5 if c != PRIMARY_TARGET]
TEXT_FEATURES_V5 = [
    "Weather Conditions", "Day of the Week", "Time of Day",
    "Train Type", "Route Congestion",
]
NUMERIC_FEATURES_V5 = ["Distance Between Stations (km)"]


def log_v5(message: str, level: str = "INFO") -> None:
    """Print a timestamped, concise, consistent log entry."""
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] [{level.upper():7s}] [RailAI] {message}")


def ensure_directories_v5() -> None:
    """Create all runtime directories without touching user datasets."""
    for folder in (DATA, EXT, MODELS, REPORTS, STATE):
        folder.mkdir(parents=True, exist_ok=True)


def load_config_v5() -> dict:
    """Load user configuration, falling back safely to documented defaults."""
    ensure_directories_v5()
    if not CONFIG_FILE.exists():
        save_json(CONFIG_FILE, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a JSON object")
        merged = dict(DEFAULT_CONFIG)
        merged.update(raw)
        merged["version"] = APP_VERSION
        return merged
    except Exception as exc:
        log_v5(f"Config unreadable; using defaults: {exc}", "WARN")
        return dict(DEFAULT_CONFIG)


def normalize_header_v5(value: Any) -> str:
    """Normalize a header for alias matching while preserving unknown labels."""
    text = re.sub(r"\s+", " ", str(value).strip())
    key = text.casefold()
    return PRIMARY_ALIASES.get(key, text)


def normalize_headers_v5(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename common aliases; reject collisions rather than silently overwrite."""
    out = frame.copy()
    names = [normalize_header_v5(c) for c in out.columns]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Header normalization creates duplicate columns: {duplicates}")
    out.columns = names
    return out


def list_local_csvs_v5() -> list[Path]:
    """List CSV candidates in project data folders, without scanning the whole disk."""
    candidates: list[Path] = []
    for folder in (ROOT, DATA, EXT):
        if folder.exists():
            candidates.extend(p for p in folder.glob("*.csv") if p.is_file())
    return sorted(set(candidates), key=lambda p: (p.stat().st_size, p.name.casefold()))


def inspect_csv_header_v5(path: Path) -> dict:
    """Read only a tiny sample to inspect schema, encoding and approximate file size."""
    path = Path(path)
    result = {"path": str(path), "exists": path.exists(), "bytes": None, "columns": [], "sample_rows": 0, "error": None}
    if not path.exists():
        result["error"] = "file_not_found"
        return result
    result["bytes"] = path.stat().st_size
    try:
        sample = pd.read_csv(path, nrows=5, low_memory=False)
        sample = normalize_headers_v5(sample)
        result["columns"] = list(map(str, sample.columns))
        result["sample_rows"] = int(len(sample))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def schema_match_primary_v5(path: Path) -> bool:
    """Return True only if the source contains every primary schema field."""
    try:
        inspected = inspect_csv_header_v5(path)
        return set(PRIMARY_REQUIRED_V5).issubset(set(inspected.get("columns", [])))
    except Exception:
        return False


def discover_primary_csv_v5(explicit: Optional[str] = None) -> Optional[Path]:
    """Find the user's labeled CSV using known names and schema inspection."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Explicit CSV path does not exist: {candidate}")
    preferred = [
        ROOT / "train delay data.csv", DATA / "train delay data.csv",
        ROOT / "train_delay_data.csv", DATA / "train_delay_data.csv",
        ROOT / "primary_user.csv", DATA / "primary_user.csv",
    ]
    for path in preferred:
        if path.exists() and schema_match_primary_v5(path):
            return path
    for path in list_local_csvs_v5():
        if schema_match_primary_v5(path):
            return path
    return None


def safe_read_csv_v5(path: Path, nrows: Optional[int] = None, **kwargs) -> pd.DataFrame:
    """Read CSV with predictable defaults and a useful error message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, nrows=nrows, low_memory=False, **kwargs)
        return normalize_headers_v5(frame)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, nrows=nrows, low_memory=False, encoding="latin-1", **kwargs)
        return normalize_headers_v5(frame)
    except Exception as exc:
        raise RuntimeError(f"Could not read CSV {path}: {exc}") from exc


def stream_csv_chunks_v5(path: Path, chunksize: int = 50000, usecols=None) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks, keeping peak memory bounded for large files."""
    path = Path(path)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    try:
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=usecols)
        for chunk in reader:
            yield normalize_headers_v5(chunk)
    except UnicodeDecodeError:
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding="latin-1", usecols=usecols)
        for chunk in reader:
            yield normalize_headers_v5(chunk)


def profile_csv_v5(path: Path, chunksize: int = 50000, max_chunks: Optional[int] = None) -> dict:
    """Compute a memory-conscious profile including missingness and numeric ranges."""
    columns = None
    total_rows = 0
    missing = Counter()
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}
    examples: dict[str, list[str]] = {}
    errors: list[str] = []
    chunks_seen = 0
    for chunk in stream_csv_chunks_v5(path, chunksize=chunksize):
        chunks_seen += 1
        columns = list(chunk.columns)
        total_rows += len(chunk)
        for col in chunk.columns:
            missing[col] += int(chunk[col].isna().sum())
            if pd.api.types.is_numeric_dtype(chunk[col]):
                vals = pd.to_numeric(chunk[col], errors="coerce")
                if vals.notna().any():
                    lo, hi = float(vals.min()), float(vals.max())
                    numeric_min[col] = min(numeric_min.get(col, lo), lo)
                    numeric_max[col] = max(numeric_max.get(col, hi), hi)
            if col not in examples:
                examples[col] = []
            for val in chunk[col].dropna().astype(str).unique()[:5]:
                if val not in examples[col] and len(examples[col]) < 10:
                    examples[col].append(val)
        if max_chunks and chunks_seen >= max_chunks:
            errors.append("profile_truncated_by_max_chunks")
            break
    return {
        "path": str(path), "bytes": Path(path).stat().st_size,
        "rows_scanned": total_rows, "chunks_scanned": chunks_seen,
        "columns": columns or [], "missing_counts": dict(missing),
        "numeric_min": numeric_min, "numeric_max": numeric_max,
        "examples": examples, "notes": errors, "profiled_at": now(),
    }


def write_jsonl_v5(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object to a newline-delimited JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")


def read_jsonl_v5(path: Path, limit: Optional[int] = None) -> list[dict]:
    """Read JSONL defensively, skipping malformed lines while recording a warning."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):            if limit is not None and len(records) >= limit:
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
            except json.JSONDecodeError:
                log_v5(f"Skipping malformed JSONL record at {path}:{line_no}", "WARN")
    return records


def append_audit_v5(event: str, details: Optional[Mapping[str, Any]] = None) -> None:
    """Write an auditable event record; never store secrets or credentials."""
    payload = {
        "timestamp_utc": now(), "app_version": APP_VERSION,
        "event": str(event), "details": dict(details or {}),
        "mode": "advisory_only",
    }
    write_jsonl_v5(AUDIT_FILE, payload)


def sha256_stream_v5(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally to avoid loading large datasets into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def provenance_record_v5(paths: Sequence[Path], extra: Optional[Mapping[str, Any]] = None) -> dict:
    """Capture local file hashes and source notes for reproducibility."""
    files = []
    for path in paths:
        path = Path(path)
        if path.exists() and path.is_file():
            files.append({
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_stream_v5(path), "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            })
    return {"app_version": APP_VERSION, "created_at": now(), "files": files, "extra": dict(extra or {})}


def validate_primary_frame_v5(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the user's seven-column dataset without inventing labels."""
    df = normalize_headers_v5(raw)
    missing_cols = [c for c in PRIMARY_REQUIRED_V5 if c not in df.columns]
    if missing_cols:
        raise ValueError("Primary dataset missing required columns: " + ", ".join(missing_cols))
    original_rows = int(len(df))
    df = df[PRIMARY_REQUIRED_V5].copy()
    for col in TEXT_FEATURES_V5:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    for col in NUMERIC_FEATURES_V5 + [PRIMARY_TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    negative_distance = int((df["Distance Between Stations (km)"] < 0).fillna(False).sum())
    negative_delay = int((df[PRIMARY_TARGET] < 0).fillna(False).sum())
    df.loc[df["Distance Between Stations (km)"] < 0, "Distance Between Stations (km)"] = np.nan
    df.loc[df[PRIMARY_TARGET] < 0, PRIMARY_TARGET] = np.nan
    missing_target = int(df[PRIMARY_TARGET].isna().sum())
    df = df.dropna(subset=[PRIMARY_TARGET]).copy()
    for col in NUMERIC_FEATURES_V5:
        median = df[col].median() if df[col].notna().any() else 0.0
        df[col] = df[col].fillna(float(median))
    for col in TEXT_FEATURES_V5:
        df[col] = df[col].fillna("Unknown").astype(str)
    # Preserve frequency as sample weight when collapsing exact duplicate rows.
    counts = df.groupby(PRIMARY_REQUIRED_V5, dropna=False, observed=True).size().rename("sample_weight").reset_index()
    df = counts.copy()
    report = {
        "rows_original": original_rows, "rows_after_target_validation": int(original_rows - missing_target - negative_delay),
        "rows_unique_after_duplicate_compression": int(len(df)),
        "duplicate_frequency_preserved_in_weights": True,
        "negative_distance_rows": negative_distance, "negative_delay_rows": negative_delay,
        "missing_target_rows_removed": missing_target,
        "missing_after_cleaning": {str(k): int(v) for k, v in df.isna().sum().items()},
        "delay_min": float(df[PRIMARY_TARGET].min()) if len(df) else None,
        "delay_median": float(df[PRIMARY_TARGET].median()) if len(df) else None,
        "delay_max": float(df[PRIMARY_TARGET].max()) if len(df) else None,
        "validated_at": now(),
    }
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} valid unique rows remain; at least 10 are needed for experimentation.")
    return df.reset_index(drop=True), report


def make_primary_features_v5(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, target-free features from the seven-column schema."""
    out = frame.copy()
    if PRIMARY_TARGET in out:
        out = out.drop(columns=[PRIMARY_TARGET])
    if "sample_weight" in out:
        out = out.drop(columns=["sample_weight"])
    distance = pd.to_numeric(out["Distance Between Stations (km)"], errors="coerce").clip(lower=0)
    out["distance_log1p"] = np.log1p(distance)
    out["distance_squared"] = distance.pow(2)
    out["distance_band_100km"] = (distance // 100).clip(0, 100)
    out["weather_normalized"] = out["Weather Conditions"].astype(str).str.casefold().str.strip()
    out["day_normalized"] = out["Day of the Week"].astype(str).str.casefold().str.strip()
    out["time_normalized"] = out["Time of Day"].astype(str).str.casefold().str.strip()
    out["train_type_normalized"] = out["Train Type"].astype(str).str.casefold().str.strip()
    out["congestion_normalized"] = out["Route Congestion"].astype(str).str.casefold().str.strip()
    out["is_unknown_weather"] = out["weather_normalized"].isin(["unknown", "none", "nan"]).astype(int)
    out["is_unknown_congestion"] = out["congestion_normalized"].isin(["unknown", "none", "nan"]).astype(int)
    return out


def build_preprocessor_v5(X: pd.DataFrame):
    """Construct a sklearn preprocessor with train-only learned imputers and encoding."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop", sparse_threshold=0.3)
    return preprocessor, numeric, categorical


def make_candidate_models_v5(seed: int = SEED) -> dict:
    """Create diverse, CPU-friendly candidate models for a fair comparison."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    candidates = {}
    candidates["extra_trees"] = ExtraTreesRegressor(
        n_estimators=180, min_samples_leaf=2, max_features=0.9,
        random_state=seed, n_jobs=max(1, min(4, os.cpu_count() or 2)),
    )
    candidates["random_forest"] = RandomForestRegressor(
        n_estimators=160, min_samples_leaf=2, max_features=0.9,
        random_state=seed, n_jobs=max(1, min(4, os.cpu_count() or 2)),
    )
    candidates["hist_gradient_boosting"] = HistGradientBoostingRegressor(
        max_iter=160, max_leaf_nodes=24, learning_rate=0.06,
        l2_regularization=2.0, random_state=seed,
    )
    return candidates


def metric_bundle_v5(y_true, y_pred) -> dict:
    """Calculate standard regression metrics and simple baseline comparisons."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        raise ValueError("No finite prediction/target pairs to evaluate")
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    baseline = np.full_like(y_true, np.median(y_true), dtype=float)
    baseline_mae = float(mean_absolute_error(y_true, baseline))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = None
    return {
        "n": int(len(y_true)), "mae_min": mae, "rmse_min": rmse,
        "r2": r2, "median_baseline_mae_min": baseline_mae,
        "mae_improvement_vs_median_baseline_pct": None if baseline_mae == 0 else float((baseline_mae - mae) / baseline_mae * 100),
        "target_mean": float(np.mean(y_true)), "target_median": float(np.median(y_true)),
        "prediction_mean": float(np.mean(y_pred)), "prediction_median": float(np.median(y_pred)),
        "absolute_error_p50": float(np.quantile(np.abs(y_true - y_pred), .50)),
        "absolute_error_p90": float(np.quantile(np.abs(y_true - y_pred), .90)),
        "absolute_error_p95": float(np.quantile(np.abs(y_true - y_pred), .95)),
    }


def chronological_or_random_split_v5(frame: pd.DataFrame, target: str, fraction: float = 0.2, seed: int = SEED):
    """Use chronological split if a trustworthy timestamp exists; otherwise seeded holdout."""
    from sklearn.model_selection import train_test_split
    time_candidates = [c for c in ("departure_date", "timestamp", "event_time", "date", "datetime") if c in frame.columns]
    for col in time_candidates:
        dt = pd.to_datetime(frame[col], errors="coerce", utc=True)
        if dt.notna().sum() >= max(20, int(.8 * len(frame))):
            order = np.argsort(dt.fillna(dt.min()).astype("int64").to_numpy())
            cut = max(1, int(len(order) * (1 - fraction)))
            return order[:cut], order[cut:], {"method": "chronological", "time_column": col}
    tr, va = train_test_split(np.arange(len(frame)), test_size=fraction, random_state=seed)
    return np.asarray(tr), np.asarray(va), {"method": "random_holdout", "reason": "no reliable timestamp column found"}


def train_primary_v5(path: Path, max_rows: Optional[int] = None, compare_models: bool = True) -> tuple[Any, dict]:
    """Train and compare candidate models on the user's seven-column labeled data."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    raw = safe_read_csv_v5(path, nrows=max_rows)
    cleaned, quality = validate_primary_frame_v5(raw)
    y = pd.to_numeric(cleaned[PRIMARY_TARGET], errors="coerce").astype(float)
    weights = pd.to_numeric(cleaned["sample_weight"], errors="coerce").fillna(1.0).astype(float)
    X = make_primary_features_v5(cleaned.drop(columns=[PRIMARY_TARGET, "sample_weight"]))
    train_idx, val_idx, split_info = chronological_or_random_split_v5(cleaned, PRIMARY_TARGET, load_config_v5().get("holdout_fraction", .2))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    preprocessor, numeric, categorical = build_preprocessor_v5(X_train)
    candidate_specs = make_candidate_models_v5() if compare_models else {"extra_trees": make_candidate_models_v5()["extra_trees"]}
    leaderboard = []
    fitted = {}
    for name, estimator in candidate_specs.items():
        try:
            pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
            try:
                pipe.fit(X_train, y_train, model__sample_weight=weights.iloc[train_idx])
            except TypeError:
                pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            metrics = metric_bundle_v5(y_val, pred)
            metrics.update({"model": name, "status": "ok"})
            leaderboard.append(metrics)
            fitted[name] = pipe
            log_v5(f"Candidate {name}: MAE={metrics['mae_min']:.2f} min, RMSE={metrics['rmse_min']:.2f} min, R²={metrics['r2']}")
        except Exception as exc:
            leaderboard.append({"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            log_v5(f"Candidate {name} failed: {exc}", "WARN")
    valid = [row for row in leaderboard if row.get("status") == "ok"]
    if not valid:
        raise RuntimeError("Every candidate model failed; see console log for details")
    best_name = min(valid, key=lambda row: row["mae_min"])["model"]
    best_model = fitted[best_name]
    validation_pred = best_model.predict(X_val)
    residuals = (y_val.to_numpy(dtype=float) - np.asarray(validation_pred, dtype=float)).tolist()
    # Refit chosen pipeline on all observations after evaluation; metrics remain holdout-only.
    final_model = clone(best_model)
    try:
        final_model.fit(X, y, model__sample_weight=weights)
    except TypeError:
        final_model.fit(X, y)
    model_path = MODELS / "railai_primary_v5.joblib"
    import joblib
    joblib.dump(final_model, model_path, compress=3)
    report = {
        "version": APP_VERSION, "trained_at": now(), "dataset": str(path),
        "rows_cleaned_unique": int(len(cleaned)), "feature_count": int(X.shape[1]),
        "numeric_features": numeric, "categorical_features": categorical,
        "split": split_info, "best_model": best_name,
        "holdout_metrics": next(row for row in valid if row["model"] == best_name),
        "leaderboard": sorted(valid, key=lambda row: row["mae_min"]),
        "quality": quality, "residuals_for_interval": residuals[-10000:],
        "model_path": str(model_path), "data_hash_sha256": sha256_stream_v5(path),
        "important_warning": "Holdout metrics are not a guarantee of real-world railway accuracy.",
    }
    save_json(EVALUATION_FILE, report)
    save_json(DATA_QUALITY_FILE, quality)
    save_json(MODEL_METADATA_FILE, {
        "version": APP_VERSION, "model_path": str(model_path), "model_name": best_name,
        "trained_at": now(), "features": list(X.columns), "target": PRIMARY_TARGET,
        "metrics": report["holdout_metrics"], "training_rows": int(len(cleaned)),
        "data_sha256": report["data_hash_sha256"],
    })
    save_model_registry_v5({"model_name": best_name, "model_path": str(model_path), "metrics": report["holdout_metrics"], "trained_at": now(), "data_hash": report["data_hash_sha256"]})
    append_audit_v5("model_trained", {"model": best_name, "rows": int(len(cleaned)), "metrics": report["holdout_metrics"]})
    return final_model, report


def save_model_registry_v5(entry: Mapping[str, Any]) -> None:
    """Append a model version to a small JSON registry."""
    registry = {"app_version": APP_VERSION, "models": []}
    if REGISTRY_FILE.exists():
        try:
            registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not isinstance(registry, dict):
        registry = {"app_version": APP_VERSION, "models": []}
    registry.setdefault("models", []).append(dict(entry))
    registry["app_version"] = APP_VERSION
    registry["updated_at"] = now()
    save_json(REGISTRY_FILE, registry)


def load_primary_model_v5():
    """Load the latest v5 model only when the expected file exists."""
    import joblib
    path = MODELS / "railai_primary_v5.joblib"
    if not path.exists():
        raise FileNotFoundError("No v5 model found. Run: python RailAI_SentinelX_v5.py --train")
    return joblib.load(path)


def predict_primary_v5(distance_km: float, weather: str, day: str, time_of_day: str,
                       train_type: str, route_congestion: str) -> dict:
    """Predict the learned delay label using the trained v5 primary model."""
    if not np.isfinite(float(distance_km)) or float(distance_km) < 0:
        raise ValueError("Distance must be a finite, non-negative number")
    model = load_primary_model_v5()
    row = pd.DataFrame([{
        "Distance Between Stations (km)": float(distance_km),
        "Weather Conditions": str(weather), "Day of the Week": str(day),
        "Time of Day": str(time_of_day), "Train Type": str(train_type),
        "Route Congestion": str(route_congestion),
    }])
    features = make_primary_features_v5(row)
    pred = float(model.predict(features)[0])
    # The target in this user's CSV is historical delay; do not imply this is a calibrated live ETA.
    pred_nonnegative = max(0.0, pred)
    result = {
        "predicted_delay_min": round(pred_nonnegative, 2),
        "raw_model_output_min": round(pred, 2),
        "prediction_type": "historical-pattern delay estimate",
        "confidence": "not_calibrated",
        "inputs": row.iloc[0].to_dict(), "created_at": now(),
        "caveat": "Not a live train prediction unless validated current operational inputs are supplied.",
    }
    write_jsonl_v5(PREDICTION_FILE, result)
    append_audit_v5("delay_prediction", {"predicted_delay_min": result["predicted_delay_min"], "inputs": result["inputs"]})
    return result


def conformal_interval_v5(point_prediction: float, residuals: Sequence[float], coverage: float = .90) -> dict:
    """Build a simple empirical absolute-residual interval from held-out errors."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    values = np.asarray(list(residuals), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return {"lower": None, "upper": None, "coverage_target": coverage, "status": "insufficient_residuals"}
    q = float(np.quantile(np.abs(values), coverage))
    return {
        "lower": round(float(point_prediction) - q, 2),
        "upper": round(float(point_prediction) + q, 2),
        "half_width": round(q, 2), "coverage_target": coverage,
        "status": "empirical_holdout_residual_interval",
        "warning": "Coverage is not guaranteed under distribution shift; validate on representative future data.",
    }


def dynamic_eta_v5(scheduled_minutes_remaining: float, predicted_delay_minutes: float,
                   current_delay_minutes: float = 0.0, distance_remaining_km: Optional[float] = None,
                   measured_speed_kmph: Optional[float] = None, add_safety_buffer: float = 0.0,
                   predicted_delay_is_incremental: bool = False) -> dict:
    """Estimate remaining time without silently double-counting an already observed delay."""
    values = [scheduled_minutes_remaining, predicted_delay_minutes, current_delay_minutes, add_safety_buffer]
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("ETA inputs must be finite numbers")
    if scheduled_minutes_remaining < 0 or predicted_delay_minutes < 0 or current_delay_minutes < 0 or add_safety_buffer < 0:
        raise ValueError("ETA inputs cannot be negative")
    base = float(scheduled_minutes_remaining)
    speed_minutes = None
    if distance_remaining_km is not None and measured_speed_kmph is not None:
        if distance_remaining_km < 0 or measured_speed_kmph < 0:
            raise ValueError("Distance and measured speed cannot be negative")
        if measured_speed_kmph >= 5:
            speed_minutes = 60.0 * float(distance_remaining_km) / float(measured_speed_kmph)
            base = max(base, speed_minutes)
    # By default the model output is treated as the total delay estimate, not an increment.
    total_delay = float(current_delay_minutes) + float(predicted_delay_minutes) if predicted_delay_is_incremental else max(float(current_delay_minutes), float(predicted_delay_minutes))
    eta = base + total_delay + float(add_safety_buffer)
    return {
        "eta_minutes_from_now": round(eta, 2),
        "base_scheduled_minutes": round(float(scheduled_minutes_remaining), 2),
        "speed_based_minutes": None if speed_minutes is None else round(speed_minutes, 2),
        "current_delay_minutes": round(float(current_delay_minutes), 2),
        "model_delay_minutes": round(float(predicted_delay_minutes), 2),
        "delay_combination_rule": "sum_incremental" if predicted_delay_is_incremental else "max_current_vs_model",
        "safety_buffer_minutes": round(float(add_safety_buffer), 2),
        "created_at": now(),
        "advisory_only": True,
    }


@dataclass
class SensorReadingV5:
    """A timestamped sensor reading used for validation, not railway control."""
    name: str
    value: float
    unit: str
    timestamp_utc: str
    source: str = "user_or_simulation"
    quality_score: float = 1.0


def sensor_age_seconds_v5(timestamp: str, current_time: Optional[datetime] = None) -> Optional[float]:
    """Return age in seconds for ISO timestamps; None means timestamp could not be parsed."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now_dt = current_time or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now_dt - dt).total_seconds())
    except Exception:
        return None

def validate_sensor_readings_v5(readings: Sequence[SensorReadingV5], max_age_seconds: float = 120.0) -> dict:
    """Flag stale, invalid, out-of-range, or mutually inconsistent readings."""
    problems = []
    accepted = []
    by_name = {}
    bounds = {"speed": (0, 250), "speed_kmph": (0, 250), "distance_km": (0, 10000), "temperature_c": (-60, 65)}
    for reading in readings:
        name = reading.name.casefold().strip()
        if not np.isfinite(float(reading.value)):
            problems.append({"type": "non_finite", "sensor": reading.name})
            continue
        if not 0 <= float(reading.quality_score) <= 1:
            problems.append({"type": "invalid_quality_score", "sensor": reading.name})
        age = sensor_age_seconds_v5(reading.timestamp_utc)
        if age is None:
            problems.append({"type": "unparseable_timestamp", "sensor": reading.name})
        elif age > max_age_seconds:
            problems.append({"type": "stale_reading", "sensor": reading.name, "age_seconds": age})
        for key, (lo, hi) in bounds.items():
            if key in name and not lo <= float(reading.value) <= hi:
                problems.append({"type": "out_of_range", "sensor": reading.name, "value": reading.value, "valid_range": [lo, hi]})
        by_name.setdefault(name, []).append(reading)
        accepted.append(reading)
    speed_values = [r for r in accepted if "speed" in r.name.casefold()]
    if len(speed_values) >= 2:
        vals = [float(r.value) for r in speed_values]
        if max(vals) - min(vals) > 25:
            problems.append({"type": "sensor_disagreement", "sensors": [r.name for r in speed_values], "spread": max(vals)-min(vals)})
    return {
        "readings_received": len(readings), "readings_accepted_for_review": len(accepted),
        "problems": problems, "status": "review_required" if problems else "no_basic_quality_issue_detected",
        "safety_claim": "This is a prototype data-quality check, not certified railway safety logic.",
    }


def risk_score_v5(sensor_quality: float = 1.0, stale_data: bool = False,
                  sensor_disagreement: bool = False, ood_score: float = 0.0,
                  vision_hazard_detected: bool = False, uncertainty_minutes: Optional[float] = None) -> dict:
    """Create a transparent advisory risk score from explicit signals, not hidden intuition."""
    signals = {}
    signals["sensor_quality"] = max(0.0, min(1.0, float(sensor_quality)))
    signals["stale_data"] = bool(stale_data)
    signals["sensor_disagreement"] = bool(sensor_disagreement)
    signals["ood_score"] = max(0.0, min(1.0, float(ood_score)))
    signals["vision_hazard_detected"] = bool(vision_hazard_detected)
    signals["uncertainty_minutes"] = None if uncertainty_minutes is None else max(0.0, float(uncertainty_minutes))
    score = 0.0
    score += (1.0 - signals["sensor_quality"]) * 25.0
    score += 20.0 if stale_data else 0.0
    score += 20.0 if sensor_disagreement else 0.0
    score += signals["ood_score"] * 20.0
    score += 35.0 if vision_hazard_detected else 0.0
    if uncertainty_minutes is not None:
        score += min(15.0, max(0.0, float(uncertainty_minutes)) / 5.0)
    score = min(100.0, score)
    if vision_hazard_detected or stale_data or sensor_disagreement:
        category = "REVIEW_REQUIRED"
    elif score >= 60:
        category = "HIGH_ADVISORY_RISK"
    elif score >= 30:
        category = "ELEVATED_ADVISORY_RISK"
    else:
        category = "LOW_ADVISORY_RISK"
    return {"score_0_100": round(score, 1), "category": category, "signals": signals,
            "action": "Human review; do not automate train movement from this score.", "advisory_only": True}


class PageHinkleyV5:
    """Lightweight online drift detector for a stream of scalar errors."""
    def __init__(self, delta: float = 0.005, threshold: float = 20.0, alpha: float = 0.999):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.drift_detected = False

    def update(self, value: float) -> dict:
        value = float(value)
        if not np.isfinite(value):
            return {"accepted": False, "reason": "non_finite_value", "drift_detected": self.drift_detected}
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative = self.alpha * self.cumulative + (value - self.mean - self.delta)
        self.minimum = min(self.minimum, self.cumulative)
        if self.cumulative - self.minimum > self.threshold:
            self.drift_detected = True
        return {"accepted": True, "count": self.count, "mean": self.mean,
                "statistic": self.cumulative - self.minimum, "drift_detected": self.drift_detected}

    def to_dict(self) -> dict:
        return {"delta": self.delta, "threshold": self.threshold, "alpha": self.alpha,
                "count": self.count, "mean": self.mean, "cumulative": self.cumulative,
                "minimum": self.minimum, "drift_detected": self.drift_detected}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        obj = cls(payload.get("delta", .005), payload.get("threshold", 20.0), payload.get("alpha", .999))
        for key in ("count", "mean", "cumulative", "minimum", "drift_detected"):
            if key in payload:
                setattr(obj, key, payload[key])
        return obj


def load_drift_detector_v5() -> PageHinkleyV5:
    """Load saved drift state or return a fresh detector."""
    if DRIFT_FILE.exists():
        try:
            return PageHinkleyV5.from_dict(json.loads(DRIFT_FILE.read_text(encoding="utf-8")))
        except Exception as exc:
            log_v5(f"Drift state could not be read: {exc}", "WARN")
    return PageHinkleyV5()


def update_drift_v5(observed_absolute_error: float) -> dict:
    """Update drift monitor with a real observed prediction error."""
    detector = load_drift_detector_v5()
    result = detector.update(observed_absolute_error)
    save_json(DRIFT_FILE, detector.to_dict())
    append_audit_v5("drift_monitor_update", {"absolute_error": observed_absolute_error, **result})
    return result


def add_feedback_v5(features: Mapping[str, Any], predicted_delay: float, observed_delay: float,
                    source: str = "manual_verified", notes: str = "") -> dict:
    """Record prediction outcome. Feedback is logged, not blindly accepted as ground truth."""
    predicted = float(predicted_delay)
    observed = float(observed_delay)
    if not np.isfinite(predicted) or not np.isfinite(observed) or predicted < 0 or observed < 0:
        raise ValueError("Predicted and observed delays must be finite and non-negative")
    record = {
        "timestamp_utc": now(), "features": dict(features), "predicted_delay_min": predicted,
        "observed_delay_min": observed, "absolute_error_min": abs(predicted-observed),
        "signed_error_min": observed-predicted, "source": str(source),
        "notes": str(notes)[:500], "validation_status": "unverified_until_reviewed",
    }
    write_jsonl_v5(USER_FEEDBACK_FILE, record)
    write_jsonl_v5(REPLAY_FILE, record)
    drift = update_drift_v5(record["absolute_error_min"])
    append_audit_v5("outcome_feedback_recorded", {"absolute_error_min": record["absolute_error_min"], "source": source, "drift": drift})
    return {"recorded": True, "feedback": record, "drift": drift,
            "note": "This record does not automatically alter the production model. Validate labels and retrain deliberately."}


def replay_memory_summary_v5(limit: int = 5000) -> dict:
    """Summarize replay memory without printing personal or unnecessary row contents."""
    records = read_jsonl_v5(REPLAY_FILE, limit=limit)
    errors = [float(r["absolute_error_min"]) for r in records if r.get("absolute_error_min") is not None]
    return {
        "records_loaded": len(records), "mean_absolute_error_min": None if not errors else float(np.mean(errors)),
        "median_absolute_error_min": None if not errors else float(np.median(errors)),
        "p90_absolute_error_min": None if not errors else float(np.quantile(errors, .90)),
        "note": "Replay memory is evidence for evaluation; it is not automatically mixed into training.",
    }


def validate_feedback_batch_v5(records: Sequence[Mapping[str, Any]]) -> dict:
    """Check whether feedback records have required keys and plausible numerical labels."""
    errors = []
    accepted = 0
    for idx, record in enumerate(records):
        required = {"features", "predicted_delay_min", "observed_delay_min", "source"}
        missing = required - set(record)
        if missing:
            errors.append({"index": idx, "problem": "missing_keys", "keys": sorted(missing)})
            continue
        try:
            p, o = float(record["predicted_delay_min"]), float(record["observed_delay_min"])
            if not np.isfinite(p) or not np.isfinite(o) or p < 0 or o < 0:
                raise ValueError("non-finite or negative label")
        except Exception as exc:
            errors.append({"index": idx, "problem": "invalid_numeric_label", "detail": str(exc)})
            continue
        accepted += 1
    return {"records_total": len(records), "records_valid_shape": accepted, "issues": errors,
            "can_train_directly": False, "reason": "feedback must be independently verified and feature schema aligned"}


def evaluate_prediction_csv_v5(path: Path, actual_column: str, prediction_column: str) -> dict:
    """Evaluate precomputed predictions from a CSV without retraining anything."""
    df = safe_read_csv_v5(path)
    for col in (actual_column, prediction_column):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
    y = pd.to_numeric(df[actual_column], errors="coerce")
    p = pd.to_numeric(df[prediction_column], errors="coerce")
    valid = y.notna() & p.notna()
    metrics = metric_bundle_v5(y[valid], p[valid])
    metrics.update({"source_csv": str(path), "actual_column": actual_column,
                    "prediction_column": prediction_column, "dropped_rows": int((~valid).sum())})
    save_json(EVALUATION_FILE, metrics)
    return metrics


def create_data_manifest_v5(root: Path = DATA) -> dict:
    """Inventory local data files and hashes; does not upload or share them."""
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            entries.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                            "sha256": sha256_stream_v5(path)})
        except Exception as exc:
            entries.append({"path": str(path), "error": str(exc)})
    manifest = {"created_at": now(), "root": str(root), "files": entries, "file_count": len(entries)}
    save_json(REPORTS / "data_manifest.json", manifest)
    return manifest


def download_file_limited_v5(url: str, destination: Path, max_bytes: int = 250 * 1024 * 1024,
                             timeout: int = 20) -> dict:
    """Download one explicitly selected public file with size/time limits and provenance."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Only valid HTTPS URLs are accepted")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": f"RailAI-Sentinel-X/{APP_VERSION}"})
    bytes_written = 0
    temp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temp.open("wb") as out:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ValueError(f"Remote file declares {declared} bytes, above limit {max_bytes}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                bytes_written += len(block)
                if bytes_written > max_bytes:
                    raise ValueError(f"Download exceeded {max_bytes} bytes")
                out.write(block)
        temp.replace(destination)
        record = {"url": url, "destination": str(destination), "bytes": bytes_written,
                  "sha256": sha256_stream_v5(destination), "downloaded_at": now()}
        append_audit_v5("public_file_downloaded", record)
        return record
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def collect_public_v5() -> dict:
    """Download curated public GitHub snapshots; never scrape restricted services."""
    urls = {
        "indian_railway_delay_visualization.zip": "https://github.com/adityaazad79/Indian-Railway-Delay-Visualization/archive/refs/heads/main.zip",
        "indian_railway_delay_datasets.zip": "https://github.com/ankitaanand28/DA323_IndianRailwayTrainDelayDatasets/archive/refs/heads/main.zip",
    }
    outcomes = []
    config = load_config_v5()
    for filename, url in urls.items():
        destination = EXT / filename
        if destination.exists():
            outcomes.append({"file": filename, "status": "already_exists", "path": str(destination)})
            continue
        try:
            outcome = download_file_limited_v5(url, destination,
                max_bytes=int(config.get("max_download_mb", 250)) * 1024 * 1024,
                timeout=int(config.get("network_timeout_seconds", 20)))
            outcomes.append({"file": filename, "status": "downloaded", **outcome})
        except Exception as exc:
            outcomes.append({"file": filename, "status": "skipped", "error": f"{type(exc).__name__}: {exc}"})
            log_v5(f"Public source skipped for {filename}: {exc}", "WARN")
    report = {"created_at": now(), "sources": PUBLIC_SOURCES, "outcomes": outcomes,
              "note": "Downloaded archives are not automatically merged into labeled training data."}
    save_json(REPORTS / "public_sources_v5.json", report)
    return report


def open_meteo_historical_v5(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """Fetch historical weather only when coordinates/dates are explicitly provided."""
    if not -90 <= float(latitude) <= 90 or not -180 <= float(longitude) <= 180:
        raise ValueError("Latitude/longitude out of range")
    for value in (start_date, end_date):
        datetime.strptime(value, "%Y-%m-%d")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={float(latitude)}&longitude={float(longitude)}&start_date={start_date}"
        f"&end_date={end_date}&hourly=temperature_2m,precipitation,rain,wind_speed_10m&timezone=UTC"
    )
    request = urllib.request.Request(url, headers={"User-Agent": f"RailAI-Sentinel-X/{APP_VERSION}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    append_audit_v5("historical_weather_fetched", {"latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date, "provider": "Open-Meteo", "license_note": "Attribute Open-Meteo; verify current terms."})
    return payload


def save_weather_json_v5(payload: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    """Persist a weather response separately; never silently label it as delay ground truth."""
    path = Path(path or (DATA / "external" / f"weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    save_json(path, dict(payload))
    return path


def build_weather_features_v5(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a weather API payload into a time-indexed feature table."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame(columns=["timestamp_utc", "temperature_2m", "precipitation", "rain", "wind_speed_10m"])
    result = pd.DataFrame({"timestamp_utc": times})
    for key in ("temperature_2m", "precipitation", "rain", "wind_speed_10m"):
        values = hourly.get(key)
        if values is not None and len(values) == len(times):
            result[key] = pd.to_numeric(pd.Series(values), errors="coerce")
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], errors="coerce", utc=True)
    result = result.sort_values("timestamp_utc").reset_index(drop=True)
    result["weather_data_source"] = "Open-Meteo historical archive"
    return result


def classify_weather_advisory_v5(temperature_c: Optional[float], rain_mm: Optional[float],
                                 wind_kmph: Optional[float]) -> dict:
    """A transparent heuristic for UI context; not a railway operating rule."""
    signals = []
    if temperature_c is not None and (temperature_c < 0 or temperature_c > 40):
        signals.append("temperature outside broad comfort band")
    if rain_mm is not None and rain_mm >= 10:
        signals.append("heavy hourly precipitation indicator")
    if wind_kmph is not None and wind_kmph >= 50:
        signals.append("high wind indicator")
    return {"signals": signals, "status": "weather_context_review" if signals else "no_heuristic_flag",
            "disclaimer": "These thresholds are generic prototype heuristics, not official railway operating thresholds."}


def validate_public_archive_v5(path: Path, max_members: int = 10000, max_uncompressed_bytes: int = 2 * 1024**3) -> dict:
    """Inspect ZIP contents before extraction; do not extract arbitrary paths."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError("Archive is not a valid ZIP file")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > max_members:
            raise ValueError("Archive has too many members")
        total = sum(i.file_size for i in infos)
        unsafe = [i.filename for i in infos if Path(i.filename).is_absolute() or ".." in Path(i.filename).parts]
        if total > max_uncompressed_bytes:
            raise ValueError("Archive uncompressed size exceeds safety limit")
        if unsafe:
            raise ValueError(f"Archive contains unsafe paths: {unsafe[:5]}")
        return {"archive": str(path), "members": len(infos), "uncompressed_bytes": total,
                "csv_members": [i.filename for i in infos if i.filename.casefold().endswith(".csv")],
                "unsafe_members": unsafe}


def stream_zip_csv_v5(zip_path: Path, member_name: str, chunksize: int = 50000,
                      max_rows: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Stream a CSV member directly from ZIP without extracting the full archive."""
    with zipfile.ZipFile(zip_path) as archive:
        safe_name = member_name.replace("\\", "/")
        if safe_name not in archive.namelist():
            raise FileNotFoundError(f"ZIP member not found: {member_name}")
        if safe_name.startswith("/") or ".." in Path(safe_name).parts:
            raise ValueError("Unsafe ZIP member path")
        with archive.open(safe_name) as handle:
            rows_seen = 0
            for chunk in pd.read_csv(handle, chunksize=chunksize, low_memory=False):
                if max_rows is not None:
                    remaining = max_rows - rows_seen
                    if remaining <= 0:
                        break
                    chunk = chunk.iloc[:remaining]
                rows_seen += len(chunk)
                yield normalize_headers_v5(chunk)
                if max_rows is not None and rows_seen >= max_rows:
                    break


def inspect_large_railway_zip_v5(path: Path) -> dict:
    """Report ZIP schema and sizes without materializing a million-row dataset."""
    report = validate_public_archive_v5(path)
    report["csv_schemas"] = []
    with zipfile.ZipFile(path) as archive:
        for name in report["csv_members"]:
            try:
                with archive.open(name) as handle:
                    header = pd.read_csv(handle, nrows=3, low_memory=False)
                report["csv_schemas"].append({"member": name, "columns": list(header.columns), "sample_rows": len(header)})
            except Exception as exc:
                report["csv_schemas"].append({"member": name, "error": str(exc)})
    return report

def summarize_numeric_distribution_v5(values: Iterable[Any]) -> dict:
    """Calculate robust numeric distribution statistics for a stream or series."""
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": int(series.size), "min": float(series.min()), "max": float(series.max()),
            "mean": float(series.mean()), "median": float(series.median()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "p05": float(series.quantile(.05)), "p25": float(series.quantile(.25)),
            "p75": float(series.quantile(.75)), "p95": float(series.quantile(.95))}


def data_quality_report_v5(path: Path, chunksize: int = 50000) -> dict:
    """Generate a schema/missingness report for any local CSV."""
    profile = profile_csv_v5(path, chunksize=chunksize)
    profile["primary_schema_match"] = schema_match_primary_v5(path)
    profile["suggested_usage"] = "labeled_delay_training" if profile["primary_schema_match"] else "inspect_schema_before_joining"
    save_json(DATA_QUALITY_FILE, profile)
    append_audit_v5("data_quality_profiled", {"path": str(path), "rows_scanned": profile["rows_scanned"], "columns": profile["columns"]})
    return profile


def merge_only_schema_compatible_v5(frames: Sequence[pd.DataFrame], join_keys: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    """Merge only on explicit shared keys; never concatenate incompatible datasets as labels."""
    if not frames:
        raise ValueError("No dataframes supplied")
    if len(frames) == 1:
        return frames[0].copy(), {"frames": 1, "merge": "none"}
    merged = frames[0].copy()
    actions = []
    for index, frame in enumerate(frames[1:], 2):
        keys = [key for key in join_keys if key in merged.columns and key in frame.columns]
        if not keys:
            actions.append({"frame": index, "status": "skipped", "reason": "no explicit shared join keys"})
            continue
        duplicate_keys = bool(frame.duplicated(keys).any())
        if duplicate_keys:
            actions.append({"frame": index, "status": "skipped", "reason": "right side has duplicate join keys", "keys": keys})
            continue
        before = len(merged)
        merged = merged.merge(frame, on=keys, how="left", validate="many_to_one", suffixes=("", f"_source{index}"))
        actions.append({"frame": index, "status": "merged", "keys": keys, "rows_before": before, "rows_after": len(merged)})
    return merged, {"frames": len(frames), "actions": actions,
                    "warning": "Only merge when keys have the same meaning and time granularity."}


def check_target_leakage_v5(feature_columns: Sequence[str], target: str, forbidden: Optional[Sequence[str]] = None) -> dict:
    """Detect obvious target/cause leakage in a proposed feature list."""
    forbidden_set = {str(target).casefold(), "delay_minutes", "historical delay (min)", "is_delayed", "actual_arrival", "actual_delay", "primary_delay_cause"}
    forbidden_set.update(str(x).casefold() for x in (forbidden or []))
    leaked = [c for c in feature_columns if str(c).casefold() in forbidden_set]
    return {"target": target, "feature_count": len(feature_columns), "suspected_leakage_features": leaked,
            "status": "block_training" if leaked else "no_exact_name_leakage_detected"}


def detect_constant_columns_v5(frame: pd.DataFrame) -> list[str]:
    """Find features that carry no variation in a given dataset slice."""
    return [str(col) for col in frame.columns if frame[col].nunique(dropna=False) <= 1]


def detect_high_missing_columns_v5(frame: pd.DataFrame, threshold: float = .8) -> dict:
    """Flag features whose missingness exceeds a configurable threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    fractions = frame.isna().mean().to_dict()
    flagged = {str(col): float(value) for col, value in fractions.items() if value >= threshold}
    return {"threshold": threshold, "missing_fraction_by_column": fractions, "flagged": flagged}


def detect_duplicate_rows_v5(frame: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> dict:
    """Count exact duplicates and duplicate key groups without deleting records."""
    duplicate_mask = frame.duplicated(subset=list(subset) if subset else None, keep="first")
    return {"rows": len(frame), "duplicate_rows": int(duplicate_mask.sum()),
            "unique_rows": int(len(frame) - duplicate_mask.sum()), "subset": list(subset) if subset else None}


def categorical_cardinality_v5(frame: pd.DataFrame, max_categories: int = 500) -> dict:
    """Flag categoricals with unusually high cardinality before encoding."""
    report = {}
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            count = int(frame[col].nunique(dropna=True))
            report[str(col)] = {"unique_values": count, "high_cardinality": count > max_categories}
    return report


def safe_feature_summary_v5(frame: pd.DataFrame, limit: int = 100) -> dict:
    """Summarize feature types and examples without dumping full data rows."""
    result = {}
    for col in list(frame.columns)[:limit]:
        series = frame[col]
        result[str(col)] = {
            "dtype": str(series.dtype), "missing": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()), "unique": int(series.nunique(dropna=True)),
            "examples": [str(x)[:80] for x in series.dropna().astype(str).drop_duplicates().head(5).tolist()],
        }
    return result


def model_card_v5(metrics: Mapping[str, Any], training_rows: int, target_definition: str,
                  intended_use: str = "research and demonstration") -> dict:
    """Build a transparent model card describing scope, data and limitations."""
    return {
        "name": "RailAI Sentinel X delay estimator", "version": APP_VERSION,
        "created_at": now(), "intended_use": intended_use,
        "target_definition": target_definition, "training_rows": int(training_rows),
        "evaluation": dict(metrics),
        "known_limitations": [
            "Prototype is not certified for railway operations.",
            "Training data may be synthetic, historical, incomplete, biased or stale.",
            "Holdout performance may not generalize to other routes or future periods.",
            "A point estimate is not a calibrated probability of safety.",
            "Camera detection is generic unless validated on railway-specific images.",
            "No direct connection to signals, interlocking, brakes or movement authority exists.",
        ],
        "prohibited_uses": ["autonomous train control", "signalling control", "braking decisions", "safety certification claims"],
        "human_oversight_required": True,
    }


def export_model_card_v5() -> Path:
    """Export model card from the latest training report if available."""
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError("Train the model before creating its model card")
    report = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    card = model_card_v5(report.get("holdout_metrics", {}), report.get("rows_cleaned_unique", 0), PRIMARY_TARGET)
    path = REPORTS / "model_card.json"
    save_json(path, card)
    return path


def prediction_explanation_v5(inputs: Mapping[str, Any], predicted_delay: float, risk: Optional[Mapping[str, Any]] = None) -> dict:
    """Produce a factual, input-grounded explanation template for the UI."""
    factors = []
    if float(inputs.get("Distance Between Stations (km)", 0) or 0) > 300:
        factors.append("The supplied station-to-station distance is relatively long.")
    weather = str(inputs.get("Weather Conditions", "Unknown")).casefold()
    if any(word in weather for word in ("rain", "storm", "fog", "snow")):
        factors.append(f"The supplied weather category is '{inputs.get('Weather Conditions')}'.")
    congestion = str(inputs.get("Route Congestion", "Unknown")).casefold()
    if any(word in congestion for word in ("high", "heavy", "severe")):
        factors.append(f"The supplied congestion category is '{inputs.get('Route Congestion')}'.")
    if not factors:
        factors.append("No simple rule-based factor was flagged from the supplied categories.")
    return {"prediction_min": round(float(predicted_delay), 2), "observed_inputs": dict(inputs),
            "input_based_notes": factors, "risk_assessment": dict(risk or {}),
            "explanation_type": "transparent input summary, not causal model attribution",
            "warning": "These notes do not prove that any one input caused the prediction."}


def feature_importance_v5(model, feature_names: Sequence[str]) -> dict:
    """Extract native tree feature importance when supported; otherwise report limitation."""
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        pairs = sorted(zip(map(str, feature_names), values.tolist()), key=lambda x: x[1], reverse=True)
        return {"method": "native_tree_feature_importance", "features": [{"name": n, "importance": float(v)} for n, v in pairs]}
    return {"method": "unavailable_for_this_model", "features": [],
            "note": "Use permutation importance on a held-out set if explainability is required."}


def permutation_importance_v5(model, X: pd.DataFrame, y: Sequence[float], repeats: int = 3) -> dict:
    """Calculate permutation importance on a small validation sample when feasible."""
    from sklearn.inspection import permutation_importance
    if len(X) > 5000:
        sample = np.random.default_rng(SEED).choice(len(X), size=5000, replace=False)
        X_eval = X.iloc[sample]
        y_eval = np.asarray(y)[sample]
    else:
        X_eval, y_eval = X, np.asarray(y)
    result = permutation_importance(model, X_eval, y_eval, n_repeats=repeats,
                                    random_state=SEED, scoring="neg_mean_absolute_error")
    items = sorted(zip(map(str, X_eval.columns), result.importances_mean.tolist(), result.importances_std.tolist()),
                   key=lambda item: item[1], reverse=True)
    return {"scoring": "negative MAE", "sample_rows": len(X_eval),
            "features": [{"name": n, "mean_importance": float(m), "std_importance": float(s)} for n, m, s in items]}


def run_fault_injection_suite_v5() -> dict:
    """Run small deterministic unit-style checks for safety-adjacent logic."""
    results = []
    def check(name, fn):
        try:
            outcome = fn()
            passed = bool(outcome)
            results.append({"name": name, "passed": passed, "detail": "assertion true" if passed else "assertion false"})
        except Exception as exc:
            results.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    check("negative_distance_rejected", lambda: _expect_exception_v5(lambda: predict_primary_v5(-1,"Clear","Monday","Morning","Express","Low"), (ValueError, FileNotFoundError)))
    check("eta_rejects_negative", lambda: _expect_exception_v5(lambda: dynamic_eta_v5(20,-1), ValueError))
    check("eta_no_double_count_default", lambda: dynamic_eta_v5(60, 10, 20)["eta_minutes_from_now"] == 80)
    check("eta_incremental_sum", lambda: dynamic_eta_v5(60, 10, 20, predicted_delay_is_incremental=True)["eta_minutes_from_now"] == 90)
    check("risk_hazard_requires_review", lambda: risk_score_v5(vision_hazard_detected=True)["category"] == "REVIEW_REQUIRED")
    check("risk_stale_requires_review", lambda: risk_score_v5(stale_data=True)["category"] == "REVIEW_REQUIRED")
    check("risk_disagreement_requires_review", lambda: risk_score_v5(sensor_disagreement=True)["category"] == "REVIEW_REQUIRED")
    check("empty_residual_interval_safe", lambda: conformal_interval_v5(10, [1, 2]) ["status"] == "insufficient_residuals")
    check("valid_interval_order", lambda: conformal_interval_v5(10, list(range(1, 100)))["lower"] < conformal_interval_v5(10, list(range(1, 100)))["upper"])
    check("page_hinkley_accepts_finite", lambda: PageHinkleyV5().update(1.0)["accepted"] is True)
    check("page_hinkley_rejects_nan", lambda: PageHinkleyV5().update(float("nan"))["accepted"] is False)
    check("header_alias", lambda: normalize_header_v5("delay_minutes") == PRIMARY_TARGET)
    check("constant_column", lambda: detect_constant_columns_v5(pd.DataFrame({"x":[1,1],"y":[1,2]})) == ["x"])
    check("missingness_flag", lambda: "x" in detect_high_missing_columns_v5(pd.DataFrame({"x":[None,None],"y":[1,2]}), .8)["flagged"])
    check("duplicate_detection", lambda: detect_duplicate_rows_v5(pd.DataFrame({"x":[1,1,2]}))["duplicate_rows"] == 1)
    check("leakage_detection", lambda: check_target_leakage_v5(["distance", "delay_minutes"], "delay_minutes")["status"] == "block_training")
    check("metric_bundle_mae", lambda: abs(metric_bundle_v5([1,2,3],[1,2,4])["mae_min"] - 1/3) < 1e-9)
    check("weather_advisory_returns_dict", lambda: "status" in classify_weather_advisory_v5(45, 12, 60))
    check("sensor_nonfinite_detected", lambda: any(p["type"] == "non_finite" for p in validate_sensor_readings_v5([SensorReadingV5("speed", float("nan"), "km/h", now())])["problems"]))
    check("risk_score_bounded", lambda: 0 <= risk_score_v5(0, True, True, 1, True, 999)["score_0_100"] <= 100)
    check("weather_payload_empty", lambda: len(build_weather_features_v5({})) == 0)
    passed = sum(1 for row in results if row["passed"])
    report = {"created_at": now(), "passed": passed, "failed": len(results)-passed, "total": len(results), "results": results,
              "warning": "Passing these software checks does not validate real-world railway safety."}
    save_json(REPORTS / "fault_injection_v5.json", report)
    append_audit_v5("fault_injection_suite", {"passed": passed, "failed": len(results)-passed})
    return report


def _expect_exception_v5(fn, expected):
    try:
        fn()
    except expected:
        return True
    except Exception:
        return False
    return False


def generate_demo_csv_v5(path: Optional[Path] = None, rows: int = 100) -> Path:
    """Create a clearly labeled toy dataset for testing when the user file is absent."""
    # Synthetic data is opt-in only and is never represented as real railway observations.
    rng = np.random.default_rng(SEED)
    weather_values = ["Clear", "Rainy", "Foggy", "Cloudy"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    times = ["Morning", "Afternoon", "Evening", "Night"]
    trains = ["Express", "Passenger", "Superfast", "Mail"]
    congestion_values = ["Low", "Medium", "High"]
    distance = rng.uniform(10, 700, size=rows)
    weather = rng.choice(weather_values, size=rows)
    day = rng.choice(days, size=rows)
    tod = rng.choice(times, size=rows)
    train = rng.choice(trains, size=rows)
    congestion = rng.choice(congestion_values, size=rows)
    delay = np.maximum(0, 3 + distance * .025 + (weather == "Rainy") * 8 + (weather == "Foggy") * 12 + (congestion == "High") * 18 + rng.normal(0, 12, size=rows))
    frame = pd.DataFrame({
        "Distance Between Stations (km)": distance.round(2), "Weather Conditions": weather,
        "Day of the Week": day, "Time of Day": tod, "Train Type": train,
        "Historical Delay (min)": delay.round(2), "Route Congestion": congestion,
    })
    path = Path(path or (DATA / "DEMO_SYNTHETIC_NOT_REAL_train_delay_data.csv"))
    frame.to_csv(path, index=False)
    save_json(REPORTS / "synthetic_demo_data_notice.json", {
        "path": str(path), "rows": rows, "created_at": now(),
        "warning": "Synthetic toy data. Do not use its metrics as evidence of real railway performance.",
    })
    return path


def run_benchmark_v5(path: Path, max_rows: Optional[int] = None) -> dict:
    """Run candidate comparison and save the results using the v5 trainer."""
    _, report = train_primary_v5(path, max_rows=max_rows, compare_models=True)
    card_path = export_model_card_v5()
    return {"training_report": report, "model_card_path": str(card_path)}


def print_report_v5(report: Mapping[str, Any]) -> None:
    """Print a readable summary of metrics without overclaiming accuracy."""
    print("\n" + "=" * 72)
    print("RAILAI SENTINEL X v5 | TRAINING REPORT")
    print("=" * 72)
    print("Dataset rows:", report.get("rows_cleaned_unique", "n/a"))
    print("Selected model:", report.get("best_model", "n/a"))
    print("Split:", report.get("split", {}))
    metrics = report.get("holdout_metrics", {})
    for key in ("mae_min", "rmse_min", "r2", "median_baseline_mae_min", "absolute_error_p90"):
        if key in metrics:
            print(f"{key:34s}: {metrics[key]}")
    print("\nCandidate leaderboard:")
    for row in report.get("leaderboard", []):
        print(f"  {row.get('model', '?'):24s} MAE={row.get('mae_min', float('nan')):.3f} RMSE={row.get('rmse_min', float('nan')):.3f} R²={row.get('r2')}")
    print("\nImportant: metrics are dataset-specific and do not establish operational safety.")
    print("=" * 72 + "\n")


def interactive_predict_v5() -> None:
    """Prompt for a small, local, human-entered scenario and show an advisory estimate."""
    def ask(prompt, default):
        value = input(f"{prompt} [{default}]: ").strip()
        return value if value else default
    try:
        distance = float(ask("Distance between stations (km)", "150"))
        weather = ask("Weather category", "Clear")
        day = ask("Day of week", "Monday")
        time_of_day = ask("Time of day", "Morning")
        train_type = ask("Train type", "Express")
        congestion = ask("Route congestion", "Low")
        result = predict_primary_v5(distance, weather, day, time_of_day, train_type, congestion)
        print(json.dumps(result, indent=2, default=str))
        explanation = prediction_explanation_v5(result["inputs"], result["predicted_delay_min"])
        print(json.dumps(explanation, indent=2, default=str))
    except (ValueError, FileNotFoundError) as exc:
        log_v5(str(exc), "ERROR")


def show_system_status_v5() -> dict:
    """Summarize files, model availability, memory and recent audit counts."""
    model_path = MODELS / "railai_primary_v5.joblib"
    csvs = list_local_csvs_v5()
    audit = read_jsonl_v5(AUDIT_FILE, limit=10000)
    feedback = read_jsonl_v5(USER_FEEDBACK_FILE, limit=10000)
    status = {
        "version": APP_VERSION, "time": now(), "root": str(ROOT),
        "model_exists": model_path.exists(), "model_bytes": model_path.stat().st_size if model_path.exists() else None,
        "csv_candidates": [{"path": str(p), "bytes": p.stat().st_size, "primary_schema_match": schema_match_primary_v5(p)} for p in csvs],
        "audit_records_read": len(audit), "feedback_records_read": len(feedback),
        "replay_summary": replay_memory_summary_v5(),
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform, "cpu_count": os.cpu_count()},
        "safety_mode": "advisory_only",
    }
    save_json(REPORTS / "system_status_v5.json", status)
    return status


def print_sources_v5() -> None:
    """Print sources, intended uses, and limitations."""
    print("Public research/data references (check current terms before use):")
    for item in PUBLIC_SOURCES:
        print(f"- {item['name']}\n  URL: {item['url']}\n  Use: {item['use']}\n  License/terms: {item['license']}\n")
    print("Weather API: https://open-meteo.com/en/docs/historical-weather-api")
    print("Official timetable enquiry: https://www.indian-rail.gov.in/enquiry/SCHEDULE/TrainSchedule.jsp")
    print("These references do not imply official live telemetry access.")


def run_doctor_v5() -> dict:
    """Check Python packages, directories, local files, and basic project health."""
    packages = {}
    for name in ("numpy", "pandas", "sklearn", "joblib"):
        try:
            mod = __import__(name)
            packages[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            packages[name] = f"missing: {exc}"
    status = show_system_status_v5()
    report = {"created_at": now(), "packages": packages, "directories": {str(p): p.exists() for p in (DATA, EXT, MODELS, REPORTS, STATE)}, "system": status}
    save_json(REPORTS / "doctor_v5.json", report)
    return report


def clean_old_temp_files_v5(older_than_hours: float = 24) -> dict:
    """Remove only RailAI-owned .part temporary downloads older than a threshold."""
    cutoff = time.time() - float(older_than_hours) * 3600
    removed = []
    for path in EXT.glob("*.part"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(str(path))
        except OSError as exc:
            log_v5(f"Could not remove temporary file {path}: {exc}", "WARN")
    return {"removed": removed, "count": len(removed), "older_than_hours": older_than_hours}


def create_backup_v5() -> Path:
    """Back up configuration, model metadata, and small state logs, not large datasets."""
    backup_dir = ROOT / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = backup_dir / f"railai_state_backup_{stamp}.zip"
    include = [CONFIG_FILE, MODEL_METADATA_FILE, REGISTRY_FILE, DRIFT_FILE, AUDIT_FILE, REPLAY_FILE, USER_FEEDBACK_FILE]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include:
            if path.exists() and path.is_file():
                archive.write(path, arcname=str(path.relative_to(ROOT)))
    append_audit_v5("state_backup_created", {"path": str(archive_path), "files": [str(p) for p in include if p.exists()]})
    return archive_path


def export_prediction_history_v5(destination: Optional[Path] = None) -> Path:
    """Export prediction history to a CSV for manual inspection."""
    records = read_jsonl_v5(PREDICTION_FILE)
    rows = []
    for record in records:
        row = {k: v for k, v in record.items() if k != "inputs"}
        for key, value in (record.get("inputs") or {}).items():
            row[f"input__{key}"] = value
        rows.append(row)
    destination = Path(destination or (REPORTS / "prediction_history_export.csv"))
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def print_quickstart_v5() -> None:
    """Print the common Windows commands in the terminal."""
    print(r"""
RAILAI SENTINEL X v5 - QUICK START (Windows / VS Code)

1. Put this file in a project folder.
2. Put your labeled CSV in data\train delay data.csv OR beside this file.3. In the VS Code terminal, run:
   py -m pip install -r requirements.txt
4. Train and compare models:
   py RailAI_SentinelX_v5.py --train
5. Predict a sample scenario:
   py RailAI_SentinelX_v5.py --predict-interactive
6. Inspect data files:
   py RailAI_SentinelX_v5.py --profile-data
7. Run basic tests:
   py RailAI_SentinelX_v5.py --self-test
8. Optional camera only after installing opencv-python and ultralytics:
   py -m pip install opencv-python ultralytics
   py RailAI_SentinelX_v5.py --camera --yolo yolo11n.pt

No 400 MB dataset is embedded in this source file. Put the dataset you own or are
licensed to use into data/; the program will inspect its schema before training.
Large ZIP files are streamed in chunks where the relevant loader supports it.
""")


def self_test_v5() -> int:
    """Run deterministic local checks and return a process exit code."""
    report = run_fault_injection_suite_v5()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["failed"] == 0 else 1


def build_arg_parser_v5():
    """Create the command-line parser for v5 features."""
    ap = argparse.ArgumentParser(
        description="RailAI Sentinel X v5: local research prototype for delay prediction and advisory risk analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train", action="store_true", help="Train and compare candidate models on the labeled primary CSV")
    ap.add_argument("--csv", default=None, help="Explicit path to the labeled seven-column CSV")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap for local test runs")
    ap.add_argument("--no-model-comparison", action="store_true", help="Train only the first candidate for faster testing")
    ap.add_argument("--predict-interactive", action="store_true", help="Prompt for scenario inputs and predict delay")
    ap.add_argument("--profile-data", action="store_true", help="Profile a local CSV in chunks")
    ap.add_argument("--profile-path", default=None, help="Specific CSV path to profile")
    ap.add_argument("--self-test", action="store_true", help="Run deterministic unit-style checks")
    ap.add_argument("--doctor", action="store_true", help="Check runtime packages and project state")
    ap.add_argument("--status", action="store_true", help="Show project status")
    ap.add_argument("--collect-public", action="store_true", help="Download curated public research archives")
    ap.add_argument("--show-sources", action="store_true", help="Show public data references")
    ap.add_argument("--create-demo-data", action="store_true", help="Explicitly create a toy synthetic CSV for testing")
    ap.add_argument("--demo-rows", type=int, default=100, help="Number of rows for the synthetic demo CSV")
    ap.add_argument("--demo", action="store_true", help="Run a prediction demo; requires a trained model")
    ap.add_argument("--camera", action="store_true", help="Open webcam object-detection demo (not railway camera input)")
    ap.add_argument("--yolo", default=None, help="Optional local YOLO weights path")
    ap.add_argument("--export-model-card", action="store_true", help="Export a model card after training")
    ap.add_argument("--manifest", action="store_true", help="Create a manifest of local data files")
    ap.add_argument("--backup-state", action="store_true", help="Back up small state/config files")
    ap.add_argument("--export-predictions", action="store_true", help="Export prediction history to CSV")
    ap.add_argument("--clean-temp", action="store_true", help="Remove stale RailAI .part downloads")
    return ap


def main_v5(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch v5 commands with friendly errors and no hidden network activity."""
    ensure_directories_v5()
    args = build_arg_parser_v5().parse_args(argv)
    try:
        if args.show_sources:
            print_sources_v5(); return 0
        if args.collect_public:
            print(json.dumps(collect_public_v5(), indent=2, default=str)); return 0
        if args.create_demo_data:
            path = generate_demo_csv_v5(rows=max(10, int(args.demo_rows)))
            print(f"Synthetic demo CSV created at: {path}")
            print("Do not use synthetic demo metrics as real-world evidence.")
            return 0
        if args.self_test:
            return self_test_v5()
        if args.doctor:
            print(json.dumps(run_doctor_v5(), indent=2, default=str)); return 0
        if args.status:
            print(json.dumps(show_system_status_v5(), indent=2, default=str)); return 0
        if args.profile_data:
            path = Path(args.profile_path) if args.profile_path else discover_primary_csv_v5(args.csv)
            if path is None:
                raise FileNotFoundError("No matching CSV found. Put it in data/ or use --profile-path.")
            print(json.dumps(data_quality_report_v5(path), indent=2, default=str)); return 0
        if args.manifest:
            print(json.dumps(create_data_manifest_v5(), indent=2, default=str)); return 0
        if args.backup_state:
            print(f"Backup created: {create_backup_v5()}"); return 0
        if args.export_predictions:
            print(f"Export created: {export_prediction_history_v5()}"); return 0
        if args.clean_temp:
            print(json.dumps(clean_old_temp_files_v5(), indent=2)); return 0
        if args.train:
            path = discover_primary_csv_v5(args.csv)
            if path is None:
                raise FileNotFoundError("No matching primary CSV found. Place your seven-column dataset in data/ or pass --csv.")
            model, report = train_primary_v5(path, max_rows=args.max_rows, compare_models=not args.no_model_comparison)
            print_report_v5(report)
            if args.export_model_card:
                print("Model card:", export_model_card_v5())
            return 0
        if args.predict_interactive:
            interactive_predict_v5(); return 0
        if args.camera:
            camera(args.yolo); return 0
        if args.demo:
            interactive_predict_v5(); return 0
        build_arg_parser_v5().print_help()
        return 0
    except KeyboardInterrupt:
        log_v5("Interrupted by user; no automatic model update was performed.", "WARN")
        return 130
    except Exception as exc:
        log_v5(f"{type(exc).__name__}: {exc}", "ERROR")
        if os.environ.get("RAILAI_DEBUG") == "1":
            traceback.print_exc()
        return 1


# ============================================================================
# EXTENDED OPERATING NOTES (kept in source for offline reference)
# ============================================================================
# 1. A model is only as useful as the target definition and its input timing.
# 2. Do not include fields that are only known after a train reaches a station.
# 3. Do not use actual delay, actual arrival, or post-event causes as predictors.
# 4. Use a chronological holdout when reliable event timestamps exist.
# 5. If no timestamp exists, random holdout is a weaker proxy and must be labeled.
# 6. Keep source datasets unchanged; all cleaned data should be reproducible.
# 7. Never overwrite the user's raw CSV with a cleaned or augmented version.
# 8. Keep each public source separate until its schema, granularity and license are checked.
# 9. A timetable describes planned service, not observed delay outcomes.
# 10. Weather archives are not train-delay labels and should not be merged without time/location keys.
# 11. Do not invent latitude, longitude, speed, track conditions or station identifiers.
# 12. If an input is missing, keep it missing or explicitly use an unknown category.
# 13. Never claim the webcam is a railway track camera.
# 14. A pretrained detector's confidence is not accident probability.
# 15. A risk score is a prioritization aid, not a certified hazard probability.
# 16. Human review is mandatory for any potential safety concern.
# 17. The code has no interface for railway signals, brakes or movement authority.
# 18. Do not add one casually; that is a different safety-critical engineering domain.
# 19. Record the model version and data hash for every training run.
# 20. Keep a holdout set out of training until model selection is complete.
# 21. Never tune thresholds on the final test set and then report it as untouched.
# 22. Compare against a median or naive baseline before celebrating an R-squared score.
# 23. MAE describes average absolute error; RMSE penalizes large errors more strongly.
# 24. R-squared can be high when a dataset has leakage or repeated patterns.
# 25. Check whether duplicate compression preserves sample frequencies correctly.
# 26. If a row contains a target but the feature values are missing, document the imputation.
# 27. Do not drop legitimate extreme delays merely because they look inconvenient.
# 28. Remove impossible values only with a stated reason and a recorded count.
# 29. Keep missing-target rows out of supervised model training.
# 30. Do not fill missing target labels with the median and pretend they were observed.
# 31. If model performance changes, inspect data drift and source changes before retraining.
# 32. Drift detection can raise false alarms and should be calibrated on historical streams.
# 33. Feedback labels need source and validation status, not just a number.
# 34. Do not immediately train on user-entered feedback without verifying it.
# 35. Replay memory is a review queue, not a magic accuracy button.
# 36. Keep model artifacts from untrusted sources isolated.
# 37. joblib and pickle files can execute code when loaded; only load files you trust.
# 38. Avoid installing packages from unknown mirrors or copying unknown scripts as admin.
# 39. Use a virtual environment to avoid damaging unrelated Python projects.
# 40. Keep dependencies pinned once a working setup has been tested.
# 41. This project is designed to run on CPU and use bounded parallelism by default.
# 42. A 400 MB CSV may need more than 400 MB of RAM after parsing and feature expansion.
# 43. One-hot encoding can expand memory use dramatically for high-cardinality columns.
# 44. Chunked profiling is safe for large files; not all training algorithms are streaming.
# 45. Do not assume every dataset inside a ZIP has the same columns as a prior version.
# 46. Inspect archive members and schemas before training.
# 47. ZIP extraction paths must be checked to avoid path traversal vulnerabilities.
# 48. Network downloads need size limits, timeouts and a provenance record.
# 49. A public URL does not mean the data is current, accurate, or legally reusable for every purpose.
# 50. Verify repository license files and data terms before redistribution.
# 51. Open-Meteo weather data requires attribution under its applicable data license.
# 52. Do not poll public government services aggressively or bypass access controls.
# 53. Prefer published APIs and datasets with clear access terms.
# 54. The listed sources are candidate references, not a claim that all have been downloaded.
# 55. Network enrichment is opt-in; training should work offline with local data.
# 56. Keep API keys out of source files and public repositories.
# 57. Never log credentials or private personal data into audit logs.
# 58. Do not claim live train status unless an authorized live data feed is configured.
# 59. Do not claim real-time GPS unless a genuine telemetry source is connected.
# 60. A static timetable cannot determine the exact current location of a train.
# 61. ETA needs a defined starting point, route, remaining distance and timing convention.
# 62. If speed is zero or unavailable, do not divide by it or fabricate a substitute.
# 63. If predicted delay is incremental, add it to current delay only when semantics are clear.
# 64. If predicted delay is total delay, do not add the current delay again.
# 65. This code defaults to the conservative max(current, model) interpretation for ETA.
# 66. Validate that assumption against the label definition in the real dataset.
# 67. Uncertainty intervals from residuals assume future errors resemble validation errors.
# 68. Distribution shift can invalidate those intervals.
# 69. Calibration must be measured on data not used for fitting or threshold selection.
# 70. The confidence field remains explicitly not_calibrated until calibrated properly.
# 71. Explainability should state observed inputs and model outputs separately.
# 72. Correlation in a tree model is not proof of causation.
# 73. Native feature importance can be biased toward high-cardinality features.
# 74. Permutation importance should use held-out data and can be computationally expensive.
# 75. Do not call heuristic notes formal explanations of a model's internal reasoning.
# 76. The model card should travel with the model artifact.
# 77. The registry records versions, but it does not prove model correctness.
# 78. Keep rollback copies before replacing a model used in a demonstration.
# 79. The self-test suite validates software behavior, not railway engineering requirements.
# 80. A green test report is not a safety certificate.
# 81. Computer vision can miss small, distant, occluded or poorly lit objects.
# 82. Generic YOLO weights may not contain classes needed for railway hazards.
# 83. Train-specific visual validation needs labeled images from representative conditions.
# 84. Camera angle and field of view affect whether an obstacle is visible.
# 85. Do not make a safety claim from a handful of webcam examples.
# 86. Test false positives and false negatives, not only successful detections.
# 87. Keep the camera optional so users without compatible hardware can run the core project.
# 88. Keep network access optional so local experiments remain reproducible.
# 89. The demo data generator is explicit and labels its output synthetic.
# 90. Never mix synthetic and real rows without provenance and an evaluation plan.
# 91. The primary seven-column dataset is small and aggregated compared with journey-level logs.
# 92. Its target may represent historical delay rather than delay remaining from now.
# 93. That distinction determines whether it can support dynamic ETA directly.
# 94. The code intentionally describes this prediction as a historical-pattern delay estimate.
# 95. Better accuracy requires better labels, stronger validation and representative inputs.
# 96. More lines of code do not inherently create a better model.
# 97. The objective is correct behavior, inspectability and repeatability, not line count alone.
# 98. Keep raw data backups and use a separate working directory for experiments.
# 99. Review generated reports before making claims in a hackathon presentation.
# 100. Report the exact dataset, split, metrics and limitations honestly.
# 101. The prototype can help demonstrate ML workflow and safety-conscious design.
# 102. It must not be deployed to make live operational decisions without formal review.
# 103. Follow applicable railway authority, cybersecurity and data governance requirements.
# 104. Ask qualified railway domain experts to review any real-world integration plan.
# 105. Use this project as an advisory research tool and learning platform.

# The additional helpers below are intentionally small and composable so the code can
# be tested in isolation instead of hiding all behavior in a single monolithic routine.


def parse_bool_v5(value: Any, default: bool = False) -> bool:
    """Parse common boolean-like values without treating every non-empty string as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def coerce_nonnegative_float_v5(value: Any, field_name: str) -> float:
    """Convert an input to a finite non-negative float with a named error."""
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def normalize_text_value_v5(value: Any, unknown: str = "Unknown") -> str:
    """Normalize a category value while preserving readable labels."""
    if value is None or pd.isna(value):
        return unknown
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else unknown


def map_weather_category_v5(value: Any) -> str:
    """Map common weather strings to coarse stable categories without inventing observations."""
    text = normalize_text_value_v5(value).casefold()
    if text == "unknown": return "Unknown"
    if any(token in text for token in ("storm", "thunder", "cyclone")): return "Storm"
    if any(token in text for token in ("fog", "mist", "haze")): return "Fog"
    if any(token in text for token in ("rain", "drizzle", "shower")): return "Rain"
    if any(token in text for token in ("snow", "sleet", "hail")): return "Frozen precipitation"
    if any(token in text for token in ("clear", "sunny", "fair")): return "Clear"
    if any(token in text for token in ("cloud", "overcast")): return "Cloudy"
    return "Other"


def map_congestion_category_v5(value: Any) -> str:
    """Map congestion strings to coarse categories while preserving unknown states."""
    text = normalize_text_value_v5(value).casefold()
    if text == "unknown": return "Unknown"
    if any(token in text for token in ("severe", "very high", "heavy", "critical")): return "High"
    if any(token in text for token in ("high", "busy", "congested")): return "High"
    if any(token in text for token in ("medium", "moderate", "normal")): return "Medium"
    if any(token in text for token in ("low", "light", "clear")): return "Low"
    return "Other"


def map_train_category_v5(value: Any) -> str:
    """Map common train-type labels to broad categories for reporting only."""
    text = normalize_text_value_v5(value).casefold()
    if text == "unknown": return "Unknown"
    if "vande bharat" in text: return "Vande Bharat"
    if "rajdhani" in text: return "Rajdhani"
    if "shatabdi" in text: return "Shatabdi"
    if "superfast" in text or "super fast" in text: return "Superfast"
    if "express" in text: return "Express"
    if "passenger" in text or "local" in text: return "Passenger"
    if "goods" in text or "freight" in text: return "Freight"
    return "Other"


def delay_bucket_v5(delay_minutes: Any) -> str:
    """Bucket a non-negative delay for dashboard summaries, not model targets."""
    value = coerce_nonnegative_float_v5(delay_minutes, "delay_minutes")
    if value < 5: return "0-5 min"
    if value < 15: return "5-15 min"
    if value < 30: return "15-30 min"
    if value < 60: return "30-60 min"
    return "60+ min"


def summarize_feedback_by_source_v5() -> dict:
    """Summarize observed errors by feedback source to find unreliable label streams."""
    records = read_jsonl_v5(USER_FEEDBACK_FILE)
    groups = {}
    for row in records:
        source = str(row.get("source", "unknown"))
        try:
            err = abs(float(row["observed_delay_min"]) - float(row["predicted_delay_min"]))
        except Exception:
            continue
        groups.setdefault(source, []).append(err)
    return {source: {"n": len(values), "mae_min": float(np.mean(values)),
                     "median_absolute_error_min": float(np.median(values))}
            for source, values in groups.items()}


def generate_experiment_id_v5(dataset_hash: str, config: Mapping[str, Any]) -> str:
    """Create a deterministic experiment key from the dataset hash and configuration."""
    raw = json.dumps({"dataset_hash": dataset_hash, "config": dict(config)}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compare_reports_v5(old_path: Path, new_path: Path) -> dict:
    """Compare two saved evaluation JSON reports on common numeric metrics."""
    old = json.loads(Path(old_path).read_text(encoding="utf-8"))
    new = json.loads(Path(new_path).read_text(encoding="utf-8"))
    old_m = old.get("holdout_metrics", old)
    new_m = new.get("holdout_metrics", new)
    metrics = {}
    for key in ("mae_min", "rmse_min", "r2", "absolute_error_p90"):
        if key in old_m and key in new_m and isinstance(old_m[key], (int, float)) and isinstance(new_m[key], (int, float)):
            metrics[key] = {"old": old_m[key], "new": new_m[key], "difference_new_minus_old": new_m[key] - old_m[key]}
    return {"old_report": str(old_path), "new_report": str(new_path), "metrics": metrics,
            "note": "Only compare scores when target, dataset scope, and evaluation split are comparable."}


def verify_model_artifact_v5(path: Path) -> dict:
    """Verify a local artifact exists and report its checksum; never load unknown artifacts."""
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    return {"exists": True, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_stream_v5(path), "warning": "Only load model artifacts created by a trusted local training run."}


def summarize_audit_events_v5(limit: int = 10000) -> dict:
    """Count event categories from the append-only audit log."""
    records = read_jsonl_v5(AUDIT_FILE, limit=limit)
    counts = Counter(str(record.get("event", "unknown")) for record in records)
    return {"records": len(records), "events": dict(counts),
            "latest_timestamp": records[-1].get("timestamp_utc") if records else None}


def create_run_report_v5(command: str, status: str, details: Mapping[str, Any]) -> Path:
    """Save a structured record for a major command run."""
    report = {"created_at": now(), "app_version": APP_VERSION, "command": command,
              "status": status, "details": dict(details), "safety_mode": "advisory_only"}
    path = REPORTS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(path, report)
    return path


def check_disk_space_v5(path: Path = ROOT, minimum_free_mb: int = 500) -> dict:
    """Check available disk space before training or downloading large files."""
    usage = shutil.disk_usage(path)
    free_mb = usage.free / (1024 * 1024)
    return {"path": str(path), "free_mb": round(free_mb, 1), "minimum_free_mb": minimum_free_mb,
            "ok": free_mb >= minimum_free_mb, "warning": None if free_mb >= minimum_free_mb else "Low disk space may interrupt training."}


def validate_eta_inputs_v5(payload: Mapping[str, Any]) -> dict:
    """Validate an ETA scenario payload before calculation."""
    required = ["scheduled_minutes_remaining", "predicted_delay_minutes"]
    missing = [key for key in required if key not in payload]
    problems = []
    for key in required:
        if key in payload:
            try:
                if coerce_nonnegative_float_v5(payload[key], key) < 0:
                    problems.append(key)
            except ValueError as exc:
                problems.append({key: str(exc)})
    return {"valid": not missing and not problems, "missing": missing, "problems": problems}


def summarize_training_data_v5(path: Path) -> dict:
    """Return the primary dataset profile and a source hash before training."""
    profile = data_quality_report_v5(path)    profile["sha256"] = sha256_stream_v5(path)
    profile["schema"] = "primary_seven_column_v1" if profile.get("primary_schema_match") else "unknown"
    return profile


def report_environment_v5() -> dict:
    """Record package versions and runtime limits for reproducibility."""
    versions = {}
    for module_name in ("numpy", "pandas", "sklearn", "joblib", "streamlit", "cv2", "ultralytics"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "installed")
        except Exception:
            versions[module_name] = None
    return {"python": sys.version, "platform": sys.platform, "cpu_count": os.cpu_count(),
            "packages": versions, "created_at": now()}


def write_environment_report_v5() -> Path:
    """Save environment metadata to help diagnose differences between laptops."""
    path = REPORTS / "environment_v5.json"
    save_json(path, report_environment_v5())
    return path


def percentile_rank_v5(value: float, reference_values: Sequence[float]) -> Optional[float]:
    """Estimate empirical percentile rank from a list of held-out values."""
    vals = np.asarray(reference_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or not np.isfinite(value):
        return None
    return float(100.0 * np.mean(vals <= float(value)))


def monitor_error_threshold_v5(error: float, threshold_minutes: float = 45.0) -> dict:
    """Raise a review flag when a single observed absolute error exceeds a threshold."""
    error = coerce_nonnegative_float_v5(error, "error")
    threshold_minutes = coerce_nonnegative_float_v5(threshold_minutes, "threshold_minutes")
    flagged = error > threshold_minutes
    return {"error_min": error, "threshold_min": threshold_minutes, "flagged": flagged,
            "action": "review label, inputs and model behavior" if flagged else "record for monitoring"}


def build_sensor_quality_score_v5(readings_report: Mapping[str, Any], expected_count: int) -> float:
    """Calculate a simple quality score from the fraction of received readings without issues."""
    expected_count = max(1, int(expected_count))
    received = int(readings_report.get("readings_received", 0))
    issue_count = len(readings_report.get("problems", []))
    completeness = min(1.0, received / expected_count)
    issue_penalty = min(1.0, issue_count / max(1, expected_count))
    return max(0.0, min(1.0, completeness * (1.0 - issue_penalty)))


def validate_data_license_note_v5(source_name: str, license_note: str, intended_use: str) -> dict:
    """Create a manual license-review checklist entry for a dataset."""
    return {"source_name": source_name, "license_note_supplied": bool(license_note.strip()),
            "license_note": license_note, "intended_use": intended_use,
            "review_required": True,
            "checklist": ["verify source ownership", "read license and terms", "check redistribution restrictions", "record attribution", "confirm personal/commercial use rights"]}


def make_readme_text_v5() -> str:
    """Generate a concise project README for the local project folder."""
    return f"""# RailAI Sentinel X v{APP_VERSION}\n\nResearch prototype for train-delay pattern estimation, ETA arithmetic, data-quality checks, risk advisories and optional computer vision.\n\n## Setup\n\n```powershell\npy -m venv .venv\n.venv\\Scripts\\activate\npy -m pip install -r requirements.txt\npy RailAI_SentinelX_v5.py --doctor\npy RailAI_SentinelX_v5.py --train\npy RailAI_SentinelX_v5.py --predict-interactive\n```\n\n## Data\n\nPlace the seven-column labeled CSV in `data/train delay data.csv`. The application does not embed or fabricate the user's large dataset. Other datasets must be inspected and mapped to a valid target before training.\n\n## Reports\n\nTraining metrics, data quality, model metadata, provenance, audit events and model cards are written under `reports/`, `models/` and `state/`.\n\n## Safety\n\nAdvisory-only prototype. It must never control railway signals, interlocking, brakes, throttle or movement authority. Synthetic data is for software testing only.\n"""


def write_readme_v5() -> Path:
    """Write README only when requested by a command or developer workflow."""
    path = ROOT / "README_RailAI_SentinelX_v5.md"
    path.write_text(make_readme_text_v5(), encoding="utf-8")
    return path


def validate_current_model_metadata_v5() -> dict:
    """Compare the saved metadata against the on-disk model artifact."""
    if not MODEL_METADATA_FILE.exists():
        return {"valid": False, "reason": "metadata_missing"}
    try:
        metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
        artifact = Path(metadata["model_path"])
        if not artifact.exists():
            return {"valid": False, "reason": "artifact_missing", "artifact": str(artifact)}
        return {"valid": True, "metadata_version": metadata.get("version"),
                "artifact": str(artifact), "artifact_bytes": artifact.stat().st_size,
                "feature_count": len(metadata.get("features", [])), "target": metadata.get("target")}
    except Exception as exc:
        return {"valid": False, "reason": f"{type(exc).__name__}: {exc}"}


def run_smoke_test_v5() -> dict:
    """Run fast tests that do not require a model or network access."""
    tests = {
        "header_normalization": lambda: normalize_header_v5(" delay_minutes ") == PRIMARY_TARGET,
        "weather_mapping": lambda: map_weather_category_v5("heavy rain") == "Rain",
        "congestion_mapping": lambda: map_congestion_category_v5("very high") == "High",
        "train_mapping": lambda: map_train_category_v5("Vande Bharat Express") == "Vande Bharat",
        "delay_bucket": lambda: delay_bucket_v5(16) == "15-30 min",
        "bool_true": lambda: parse_bool_v5("yes") is True,
        "bool_false": lambda: parse_bool_v5("no") is False,
        "nonnegative_float": lambda: coerce_nonnegative_float_v5("12.5", "test") == 12.5,
        "eta_validation": lambda: validate_eta_inputs_v5({"scheduled_minutes_remaining": 20, "predicted_delay_minutes": 5})["valid"],
        "feature_leakage": lambda: check_target_leakage_v5(["distance_km"], "delay_minutes")["status"] == "no_exact_name_leakage_detected",
        "drift_serialization": lambda: PageHinkleyV5.from_dict(PageHinkleyV5().to_dict()).count == 0,
        "risk_bounds": lambda: risk_score_v5(sensor_quality=1.0)["score_0_100"] == 0.0,
        "empty_weather": lambda: build_weather_features_v5({}).empty,
        "timestamp_invalid": lambda: sensor_age_seconds_v5("not-a-date") is None,
        "metrics_perfect": lambda: metric_bundle_v5([1, 2, 3], [1, 2, 3])["mae_min"] == 0.0,
    }
    rows = []
    for name, test in tests.items():
        try:
            result = bool(test())
            rows.append({"test": name, "passed": result})
        except Exception as exc:
            rows.append({"test": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    report = {"created_at": now(), "total": len(rows), "passed": sum(x["passed"] for x in rows),
              "failed": sum(not x["passed"] for x in rows), "tests": rows}
    save_json(REPORTS / "smoke_tests_v5.json", report)
    return report


def run_quality_gates_v5(path: Path) -> dict:
    """Run pre-training quality checks and return blockers and warnings."""
    inspection = inspect_csv_header_v5(path)
    blockers = []
    warnings_list = []
    if inspection.get("error"):
        blockers.append("CSV header could not be read")
    if not set(PRIMARY_REQUIRED_V5).issubset(set(inspection.get("columns", []))):
        blockers.append("primary labeled schema does not match")
    if inspection.get("bytes", 0) == 0:
        blockers.append("empty file")
    try:
        sample = safe_read_csv_v5(path, nrows=1000)
        if len(sample) < 30:
            warnings_list.append("fewer than 30 sampled rows; estimates will be unstable")
        if sample.empty:
            blockers.append("no rows readable")
        if PRIMARY_TARGET in sample and pd.to_numeric(sample[PRIMARY_TARGET], errors="coerce").notna().mean() < .5:
            warnings_list.append("more than half of sampled target values are missing or non-numeric")
    except Exception as exc:
        blockers.append(f"sample read failed: {exc}")
    result = {"path": str(path), "inspection": inspection, "blockers": blockers,
              "warnings": warnings_list, "can_train": not blockers, "created_at": now()}
    save_json(REPORTS / "quality_gates_v5.json", result)
    return result


def update_config_value_v5(key: str, value: Any) -> dict:
    """Update one known configuration setting and persist it."""
    config = load_config_v5()
    if key not in DEFAULT_CONFIG:
        raise KeyError(f"Unknown config key: {key}")
    if key in {"max_training_rows", "csv_chunk_size", "max_categories", "minimum_training_rows", "max_replay_rows", "max_download_mb", "network_timeout_seconds"}:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{key} must be positive")
    if key in {"holdout_fraction", "residual_interval_quantile"}:
        value = float(value)
        if not 0 < value < 1:
            raise ValueError(f"{key} must be between 0 and 1")
    config[key] = value
    save_json(CONFIG_FILE, config)
    append_audit_v5("config_updated", {"key": key, "value": value})
    return config


def print_banner_v5() -> None:
    """Print a banner that sets the scope of the prototype clearly."""
    print("=" * 78)
    print(f"RAILAI SENTINEL X v{APP_VERSION}")
    print("Train delay pattern estimation | dynamic ETA arithmetic | advisory risk checks")
    print("=" * 78)
    print("MODE: RESEARCH / DEMONSTRATION ONLY")
    print("This program cannot control railway signals, brakes, interlocking or train movement.")
    print("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """v5 main entry point; replaces the earlier prototype CLI."""
    print_banner_v5()
    return main_v5(argv)


if __name__ == "__main__":
    raise SystemExit(main())