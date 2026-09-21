"""Step 1 - Data registration.

Creates (if needed) a Hugging Face *dataset* repository and uploads the raw
tourism.csv file to it, so every later stage reads the data from one governed,
versioned location instead of from a local disk.
"""
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

# ---------------------------------------------------------------- configuration
# The token comes from the environment: a Colab secret locally, a GitHub secret in CI.
api = HfApi(token=os.getenv("HF_TOKEN"))
# The namespace is resolved from the token, so no user name is hard-coded in the repo.
HF_USER = os.getenv("HF_USERNAME") or api.whoami()["name"]
DATASET_REPO = f"{HF_USER}/tourism-package-prediction-data"

# Paths are anchored on this file, so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "tourism.csv"

# ------------------------------------------------------------ sanity check first
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw dataset not found at {RAW_FILE}. Upload tourism.csv to the data folder.")

raw = pd.read_csv(RAW_FILE)
if "ProdTaken" not in raw.columns:
    raise ValueError("The target column 'ProdTaken' is missing - refusing to register this file.")
print(f"Raw file OK: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

# --------------------------------------------------------------- register on HF
# exist_ok=True makes the step idempotent: the CI pipeline can re-run it safely.
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

api.upload_file(
    path_or_fileobj=str(RAW_FILE),
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Register raw tourism dataset",
)
print(f"Dataset registered at https://huggingface.co/datasets/{DATASET_REPO}")
