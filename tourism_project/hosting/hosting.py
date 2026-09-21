"""Step 4 - Hosting.

Pushes everything in tourism_project/deployment (Dockerfile, app.py,
requirements.txt, README.md) to a Docker-based Hugging Face Space.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USER = os.getenv("HF_USERNAME") or api.whoami()["name"]
SPACE_REPO = f"{HF_USER}/tourism-package-prediction-app"
MODEL_REPO = f"{HF_USER}/tourism-package-prediction-model"

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deployment"

# A Docker Space builds the image from our Dockerfile; exist_ok keeps re-runs safe.
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker",
                private=False, exist_ok=True)

# Tell the app which model repo to read, without hard-coding it in app.py.
api.add_space_variable(repo_id=SPACE_REPO, key="MODEL_REPO_ID", value=MODEL_REPO)

# Uploading the folder creates a new commit, which triggers a rebuild of the Space.
api.upload_folder(
    folder_path=str(DEPLOY_DIR),
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy Streamlit app",
)
print(f"Deployment files pushed. App: https://huggingface.co/spaces/{SPACE_REPO}")
