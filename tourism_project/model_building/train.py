"""Step 3 - Model training, experiment tracking and model registration.

Loads the train/test splits from the Hugging Face dataset repo, tunes an XGBoost
classifier with a cross-validated grid search, logs every tried configuration
to MLflow, evaluates the winner and registers it on the Hugging Face model hub.
"""
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------- configuration
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
HF_USER = os.getenv("HF_USERNAME") or api.whoami()["name"]
DATASET_REPO = f"{HF_USER}/tourism-package-prediction-data"
MODEL_REPO = f"{HF_USER}/tourism-package-prediction-model"

TARGET = "ProdTaken"
SEED = 42

BUILD_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BUILD_DIR / "artifacts"   # model binaries (not committed to git)
REPORT_DIR = BUILD_DIR / "reports"       # small JSON report (committed by the CI job)
ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# The MLflow server is started by the notebook / the GitHub Actions job.
# Child runs are created for ~100 configurations: keep the console log readable.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

# ------------------------------------------------ 1. load the splits from the Hub
def load_split(filename: str) -> pd.DataFrame:
    """Download one CSV from the dataset repo and return it as a DataFrame."""
    local = hf_hub_download(repo_id=DATASET_REPO, filename=filename,
                            repo_type="dataset", token=HF_TOKEN)
    return pd.read_csv(local)

train_df, test_df = load_split("train.csv"), load_split("test.csv")
X_train, y_train = train_df.drop(columns=TARGET), train_df[TARGET]
X_test, y_test = test_df.drop(columns=TARGET), test_df[TARGET]
print(f"Train {X_train.shape} | Test {X_test.shape}")

# ------------------------------------------------------ 2. model and parameters
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

# Tree ensembles do not need scaling. Imputers are kept as a safety net for
# future data; the one-hot encoder ignores categories it has never seen.
preprocess = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# Buyers are the minority (~1 in 5): weight them up by the negative/positive ratio.
imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric="logloss",
                            random_state=SEED, n_jobs=1)),
])

param_grid = {
    "model__n_estimators": [200, 400],       # number of boosted trees
    "model__max_depth": [4, 6, 8],           # tree complexity
    "model__learning_rate": [0.05, 0.1],     # shrinkage applied to each tree
    "model__subsample": [0.8, 1.0],          # share of rows sampled per tree
    "model__colsample_bytree": [0.7, 1.0],   # share of features sampled per tree
    "model__reg_lambda": [1.0, 5.0],         # L2 regularisation strength
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def score(y_true, proba, threshold) -> dict:
    """Headline classification metrics for a given probability cut-off."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


# ------------------------------------------------- 3. tune, log and evaluate
with mlflow.start_run(run_name="xgboost-grid-search"):
    # F1 on the buyer class is the tuning objective: it balances missed buyers
    # (lost revenue) against wasted calls (marketing cost).
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)

    # One child run per configuration, so the full search is auditable in MLflow.
    trials = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    for _, trial in trials.iterrows():
        with mlflow.start_run(run_name=f"trial-rank-{int(trial['rank_test_score']):03d}", nested=True):
            mlflow.log_params(trial["params"])
            mlflow.log_metrics({"cv_f1_mean": trial["mean_test_score"],
                                "cv_f1_std": trial["std_test_score"],
                                "fit_time_s": trial["mean_fit_time"]})

    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)
    mlflow.log_params({"scale_pos_weight": round(imbalance_ratio, 4), "cv_folds": cv.get_n_splits(),
                       "n_configurations": len(trials)})
    mlflow.log_metric("best_cv_f1", search.best_score_)
    print(f"Tried {len(trials)} configurations | best CV F1 = {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)

    # Decision threshold: chosen on out-of-fold training predictions (never on
    # the test set) as the cut-off that maximises F1.
    oof_proba = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.20, 0.71, 0.05), 2)
    threshold = float(max(grid, key=lambda t: f1_score(y_train, (oof_proba >= t).astype(int))))
    mlflow.log_param("decision_threshold", threshold)
    print(f"Decision threshold selected from out-of-fold predictions: {threshold}")

    train_scores = score(y_train, best_model.predict_proba(X_train)[:, 1], threshold)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_scores = score(y_test, test_proba, threshold)
    mlflow.log_metrics({f"train_{k}": v for k, v in train_scores.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items()})

    print("\nPerformance (train vs test):")
    print(pd.DataFrame({"train": train_scores, "test": test_scores}).round(4).to_string())
    print("\nTest confusion matrix [rows = actual, cols = predicted]:")
    print(confusion_matrix(y_test, (test_proba >= threshold).astype(int)))

    # --------------------------------------------- 4. persist model + metadata
    model_file = ARTIFACT_DIR / "tourism_model.joblib"
    meta_file = ARTIFACT_DIR / "model_metadata.json"
    joblib.dump(best_model, model_file)

    metadata = {
        "algorithm": "XGBClassifier",
        "decision_threshold": threshold,
        "best_params": search.best_params_,
        "best_cv_f1": round(float(search.best_score_), 4),
        "train_metrics": {k: round(float(v), 4) for k, v in train_scores.items()},
        "test_metrics": {k: round(float(v), 4) for k, v in test_scores.items()},
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }
    meta_file.write_text(json.dumps(metadata, indent=2))
    (REPORT_DIR / "latest_metrics.json").write_text(json.dumps(metadata, indent=2))

    mlflow.log_artifact(str(model_file), artifact_path="model")
    mlflow.log_artifact(str(meta_file), artifact_path="model")

# ---------------------------------------- 5. register the best model on the Hub
api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
for file in (model_file, meta_file):
    api.upload_file(
        path_or_fileobj=str(file),
        path_in_repo=file.name,
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message=f"Register {file.name} (test F1 = {test_scores['f1']:.3f})",
    )
print(f"\nBest model registered at https://huggingface.co/{MODEL_REPO}")
