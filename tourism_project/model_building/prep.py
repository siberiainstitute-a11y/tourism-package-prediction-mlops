"""Step 2 - Data preparation.

Loads the registered raw data from the Hugging Face dataset repo, cleans it,
creates a stratified train/test split, saves both files locally and uploads
them back to the same dataset repo for the training stage.
"""
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- configuration
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
HF_USER = os.getenv("HF_USERNAME") or api.whoami()["name"]
DATASET_REPO = f"{HF_USER}/tourism-package-prediction-data"

TARGET = "ProdTaken"
TEST_SIZE = 0.20
SEED = 42

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------- 1. load straight from the Hub
raw_path = hf_hub_download(
    repo_id=DATASET_REPO, filename="tourism.csv", repo_type="dataset", token=HF_TOKEN
)
df = pd.read_csv(raw_path)
print(f"Loaded from the Hub: {df.shape[0]:,} rows x {df.shape[1]} columns")

# -------------------------------------------------------------------- 2. clean
# (a) Identifier-like columns carry no behavioural signal and would only let the
#     model memorise rows: the exported index and the customer ID are removed.
id_like = [c for c in df.columns if c.startswith("Unnamed") or c == "CustomerID"]
df = df.drop(columns=id_like)
print(f"Dropped identifier columns: {id_like}")

# (b) Normalise text categories: trim stray spaces and repair the known
#     'Fe Male' typo so one gender is not split into two levels.
text_cols = df.select_dtypes(include="object").columns
for col in text_cols:
    df[col] = df[col].str.strip()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# (c) Once the ID is gone, exact duplicate rows add no information and could
#     land on both sides of the split (leakage), so they are removed.
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_before - len(df)} duplicate rows -> {len(df):,} rows remain")

# (d) Missing values are reported here; imputation itself lives inside the model
#     pipeline so that it is learned from the training fold only.
n_missing = int(df.isna().sum().sum())
print(f"Missing cells after cleaning: {n_missing}")

# -------------------------------------------------------------------- 3. split
# Only ~19% of customers buy, so the split is stratified on the target.
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
)
train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train: {train_df.shape} | purchase rate {train_df[TARGET].mean():.3f}")
print(f"Test : {test_df.shape} | purchase rate {test_df[TARGET].mean():.3f}")

# ------------------------------------------------- 4. push the splits to the Hub
for path in (train_path, test_path):
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add {path.name} produced by prep.py",
    )
print(f"Train/test splits uploaded to https://huggingface.co/datasets/{DATASET_REPO}")
