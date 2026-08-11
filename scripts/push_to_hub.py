"""
scripts/push_to_hub.py
───────────────────────
Uploads trained model artifacts from models/artifacts/ to Hugging Face Hub repository.

Usage:
    python scripts/push_to_hub.py [--model churn-model]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger("push_to_hub")


def push_to_hub(model_name: str = "churn-model") -> None:
    settings = get_settings()

    if not settings.hf_repo_id:
        log.error("HF_REPO_ID is not configured in .env! Please set HF_REPO_ID.")
        sys.exit(1)

    token = settings.hf_hub_token or None
    artifacts_dir = Path(settings.model_registry_path) / model_name

    if not artifacts_dir.exists():
        log.error("Artifact directory does not exist at: %s", artifacts_dir)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        log.info("Creating / checking HF Hub repository: %s", settings.hf_repo_id)
        api.create_repo(repo_id=settings.hf_repo_id, repo_type="model", exist_ok=True)

        log.info(
            "Uploading %s artifacts from %s to HF Hub...", model_name, artifacts_dir
        )
        api.upload_folder(
            folder_path=str(artifacts_dir),
            path_in_repo=model_name,
            repo_id=settings.hf_repo_id,
            repo_type="model",
        )
        log.info(
            "Successfully pushed %s artifacts to Hugging Face Hub: https://huggingface.co/%s",
            model_name,
            settings.hf_repo_id,
        )

    except Exception as e:
        log.error("Failed to push artifacts to Hugging Face Hub: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push model artifacts to Hugging Face Hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="churn-model",
        help="Model name to push (default: churn-model)",
    )
    args = parser.parse_args()

    push_to_hub(args.model)
