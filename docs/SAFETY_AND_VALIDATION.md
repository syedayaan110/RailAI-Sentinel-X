# Safety and validation boundaries

RailAI Sentinel X is an experimental research and demonstration prototype. It is not certified, railway-authorised, or validated for live railway operations.

## Hard safety boundary

Do not use this software to control or issue commands to railway signals, interlocking, brakes, throttle, movement authority, or train movement. All outputs are advisory and require qualified human review. Do not connect this prototype to operational railway control systems.

## Evidence required before stronger claims

- Clearly define the prediction target, prediction horizon, and operating context.
- Record provenance, licensing, schema, and data-quality checks for every dataset.
- Use representative railway data and chronological train/validation/test splits where timestamps permit.
- Check target quality, leakage, duplicates, missingness, and class/target distributions.
- Compare against simple baselines and report appropriate metrics such as MAE/RMSE for delay estimates.
- Evaluate uncertainty and robustness to stale, missing, noisy, contradictory, and out-of-distribution inputs.
- Conduct railway-domain engineering, cybersecurity, human-factors, and regulatory reviews before any operational consideration.

## Computer vision

The optional generic pretrained detector is not trained or validated for railway-specific hazards. A webcam demo is not a trackside camera. Detection confidence is not accident probability, and missed detections are possible.

## Data and credentials

Keep API keys, access tokens, credentials, private sensor data, and restricted datasets out of source control. Check dataset and model licences before redistribution. Treat downloaded model files and serialized Python artifacts as untrusted until their source is verified.
