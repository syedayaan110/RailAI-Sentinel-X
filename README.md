# RailAI Sentinel X

**Experimental railway delay-estimation and advisory decision-support prototype.**

RailAI Sentinel X explores dataset inspection, historical delay-pattern modelling, ETA arithmetic, risk-oriented checks, data quality, provenance/audit records, adaptive simulation, optional weather enrichment, and an optional computer-vision demonstration.

> [!CAUTION]
> **Research/demo only.** This project is not certified, approved, or validated for railway operations. It must never control railway signals, interlocking, brakes, throttle, movement authority, or train movement. Outputs are advisory only and require qualified human review.

## Repository structure

```text
RailAI-Sentinel-X/
├── RailAI_SentinelX_v5.py
├── requirements.txt
├── requirements-camera.txt
├── requirements-adaptive.txt
├── experiments/
│   └── railai_v5_adaptive_ai.py
├── archive/
│   ├── RailAI_SentinelX_v4_FIXED.py
│   └── railai_sentinel_x_v3.py
├── data/README.md
├── docs/
│   ├── PROJECT_STATUS.md
│   └── SAFETY_AND_VALIDATION.md
└── .github/workflows/ci.yml
```

The v5 script is the primary entry point. The adaptive simulation is a separate experiment, not a module imported by the primary application. Earlier versions are retained for reference.

## Requirements

- **Python 3.15** (recommended/default; CI also checks Python 3.12)
- Core: NumPy, pandas, scikit-learn, joblib
- Optional camera demo: OpenCV and Ultralytics
- Optional adaptive demo: requests

## Setup

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify

```bash
python RailAI_SentinelX_v5.py --doctor
python RailAI_SentinelX_v5.py --self-test
```

The built-in software self-test suite completed with **21 passed, 0 failed** in the preparation environment. GitHub Actions tests Python 3.15 (project default) and Python 3.12 (compatibility check). These checks do not establish prediction accuracy, railway safety, certification, or deployment readiness. GitHub Actions runs syntax and smoke checks on pushes and pull requests.

## Run the primary v5 prototype

Inspect project/data status:

```bash
python RailAI_SentinelX_v5.py --status
python RailAI_SentinelX_v5.py --profile-data
python RailAI_SentinelX_v5.py --show-sources
python RailAI_SentinelX_v5.py --manifest
```

### Add a labelled dataset

Put a licensed CSV in `data/`, preferably named `train delay data.csv`. The default labelled schema expects:

- `Distance Between Stations (km)`
- `Weather Conditions`
- `Day of the Week`
- `Time of Day`
- `Train Type`
- `Historical Delay (min)` (target)
- `Route Congestion`

Train and compare models:

```bash
python RailAI_SentinelX_v5.py --train
```

For a small software smoke run:

```bash
python RailAI_SentinelX_v5.py --train --max-rows 500
```

A capped run is not appropriate for judging model quality. Check target definitions, timestamps, duplicates, missingness, dataset provenance, and possible leakage before interpreting results.

### Generate synthetic demo data

```bash
python RailAI_SentinelX_v5.py --create-demo-data --demo-rows 150
```

Synthetic records are for software tests and demonstrations only. Do not present metrics from synthetic data as evidence of real railway performance.

### Interactive estimate

```bash
python RailAI_SentinelX_v5.py --predict-interactive
```

This is a historical-pattern estimate, not a verified live train ETA or guaranteed arrival time.

## Optional computer-vision demo

Install optional dependencies and provide compatible model weights:

```bash
python -m pip install -r requirements-camera.txt
python RailAI_SentinelX_v5.py --camera --yolo yolo11n.pt
```

The generic pretrained YOLO model is not railway-specific. A webcam demo is not a trackside camera, and missed or incorrect detections are possible.

## Adaptive simulation

```bash
python -m pip install -r requirements-adaptive.txt
python experiments/railai_v5_adaptive_ai.py
```

This is a separate simulation using generated scenarios and optional weather enrichment, not live railway telemetry.

## Data, model files, and generated artifacts

Large datasets, downloaded archives, model weights, trained model bundles, logs, caches, and local runtime state are intentionally excluded from Git. Add only data you are authorised to use and redistribute. Check each dataset's licence and provenance. Never commit API keys, credentials, private sensor data, or personal information.

Read [data/README.md](data/README.md), [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md), and [docs/SAFETY_AND_VALIDATION.md](docs/SAFETY_AND_VALIDATION.md).

## Current limitations

No labelled railway dataset, operational sensor stream, or railway-specific labelled vision dataset was supplied with this repository. A full training benchmark has therefore not been claimed. The adaptive simulation and primary trainer are separate prototypes, not one integrated production system. No live railway deployment or certification is claimed.
