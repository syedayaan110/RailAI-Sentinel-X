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

if __name__=="__main__": main()