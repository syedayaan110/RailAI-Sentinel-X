# Project status

## Included

- Primary RailAI Sentinel X v5 dataset-first Python prototype.
- Separate adaptive/simulation prototype in `experiments/`.
- Earlier v3 and v4 Python scripts retained under `archive/`.
- Core and optional dependency manifests.
- Setup and usage instructions.
- Data-folder guidance.
- Safety, validation, and limitation notes.
- GitHub Actions syntax and smoke-test workflow.

## Verification performed before publication

- Python syntax parsing passed for all four Python source files.
- Primary v5 `--doctor` completed and detected core packages.
- Primary v5 `--self-test` completed: **21 passed, 0 failed**.
- A secret-pattern scan found no private-key markers, GitHub personal access tokens, or AWS access-key patterns in committed text files.

## Not included / not claimed

- No labelled railway CSV or journey-level dataset archive was supplied.
- No live operational train feed or railway sensor stream was supplied.
- Optional YOLO model weights are not included; the generic pretrained model is not railway-specific.
- A full training benchmark was not performed because labelled data was unavailable.
- The adaptive simulation and primary trainer are separate prototypes, not one integrated production system.
- No Streamlit web dashboard or deployment configuration is included in this source bundle.
- No live railway deployment, railway certification, or operational safety validation is claimed.

Passing software self-tests does not establish prediction accuracy, railway safety, certification, or readiness for deployment.
