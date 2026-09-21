# Wellness Tourism Package - purchase prediction

End-to-end MLOps pipeline: Hugging Face Hub (data, model, app) + MLflow + GitHub Actions.

| Folder | Purpose |
|---|---|
| `data/` | raw dataset registered on the Hugging Face dataset repo |
| `model_building/` | `data_register.py`, `prep.py`, `train.py` and the latest evaluation report |
| `deployment/` | Dockerfile, Streamlit app and its requirements |
| `hosting/` | script that pushes the deployment files to the Hugging Face Space |

Live app: https://huggingface.co/spaces/siberiainstitute-a11y/tourism-package-prediction-app

_Last published from the project notebook: 2026-09-21 03:55 UTC_
