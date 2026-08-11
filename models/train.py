"""
models/train.py
───────────────
Training script for the churn-model (Phase 1 baseline: Logistic Regression).

Usage:
    python models/train.py [--data-path data/raw/<filename>.csv] [--version v1]

Design principles enforced here:
  1. Train/test split BEFORE any preprocessing fitting — no leakage.
  2. sklearn Pipeline: preprocessing + model are ONE serialized object.
     This guarantees that the exact same transformations applied during
     training are always applied at inference.
  3. Baseline (majority-class) accuracy is logged so the real model's
     gain over a trivial predictor is always explicit.
  4. WHY NOT ACCURACY ALONE: The Telco churn dataset is imbalanced
     (~73% non-churners). A model that always predicts "No Churn" gets
     ~73% accuracy while being completely useless. We therefore report
     precision, recall, F1, and ROC-AUC — metrics that are meaningful
     across both classes.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Make project root importable when running as a script ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.feature_config import (  # noqa: E402
    ALL_FEATURES,
    ARTIFACT_FILENAME,
    CATEGORICAL_FEATURES,
    CURRENT_VERSION_FILENAME,
    METADATA_FILENAME,
    MODEL_NAME,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from models.pipeline_shared import (  # noqa: E402
    build_pipeline,
    compute_training_snapshot,
    evaluate_pipeline as evaluate,
    preprocess_raw,
)
from simulator.split import make_splits, save_sim_split  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
ARTIFACTS_ROOT = PROJECT_ROOT / "models" / "artifacts"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_path: Path) -> pd.DataFrame:
    """Load raw CSV; fail fast with a descriptive error if the file is missing."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n\nDataset not found at: {data_path}\n"
            f"Please download the Telco Customer Churn CSV and place it at:\n"
            f"  {data_path}\n"
            f"Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        )
    df = pd.read_csv(data_path)
    log.info(
        "Loaded %d rows × %d columns from %s", len(df), len(df.columns), data_path.name
    )
    return df


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Git commit hash (best-effort; not fatal if git is absent)
# ---------------------------------------------------------------------------
def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------
def save_artifacts(
    pipeline: Pipeline,
    version: str,
    metrics: dict,
    snapshot: dict,
    n_train: int,
    n_test: int,
) -> Path:
    artifact_dir = ARTIFACTS_ROOT / MODEL_NAME / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Serialized pipeline
    pipeline_path = artifact_dir / ARTIFACT_FILENAME
    joblib.dump(pipeline, pipeline_path)
    log.info("Pipeline saved → %s", pipeline_path)

    # 2. Metadata JSON
    metadata = {
        "model_name": MODEL_NAME,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_model": "LogisticRegression",
        "n_train": n_train,
        "n_test": n_test,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "eval_metrics": metrics,
        "training_data_snapshot": snapshot,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }
    metadata_path = artifact_dir / METADATA_FILENAME
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved → %s", metadata_path)

    # 3. Update the current-version pointer
    # Using a flat JSON pointer file (not a DB row) for simplicity:
    # inspectable with `cat`, no DB connection required at training time,
    # swappable with a single file write. A DB-backed registry comes in Phase 4.
    pointer_path = ARTIFACTS_ROOT / MODEL_NAME / CURRENT_VERSION_FILENAME
    with open(pointer_path, "w") as f:
        json.dump({"version": version, "artifact_dir": str(artifact_dir)}, f, indent=2)
    log.info("Version pointer updated → %s (version=%s)", pointer_path, version)

    return artifact_dir


# ---------------------------------------------------------------------------
# Deployment record helper (records deployment history in DB)
# ---------------------------------------------------------------------------
def record_deployment(
    model_name: str,
    version: str,
    git_commit: str,
    eval_metrics: dict,
) -> None:
    """
    Record a new model deployment in the database.
    Updates previous deployments' is_current status to False.
    Fails gracefully if the database is unreachable (e.g., offline training).
    """
    model_id = f"{model_name}:{version}"
    try:
        from app.database import SessionLocal
        from agent.db_models import Deployment

        db = SessionLocal()
        try:
            # Mark prior deployments for this model as non-current
            prior_deployments = (
                db.query(Deployment)
                .filter(Deployment.model_id.like(f"{model_name}:%"))
                .all()
            )
            for d in prior_deployments:
                d.is_current = False

            new_deploy = Deployment(
                model_id=model_id,
                version=version,
                git_commit=git_commit,
                eval_metrics=eval_metrics,
                is_current=True,
                deployed_at=datetime.now(timezone.utc),
            )
            db.add(new_deploy)
            db.commit()
            log.info("Recorded deployment in DB: %s (is_current=True)", model_id)
        except Exception:
            db.rollback()
            log.warning("Failed to record deployment in DB (non-fatal)")
        finally:
            db.close()
    except Exception:
        log.warning("Database unavailable for recording deployment (non-fatal)")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(data_path: Path, version: str, save_sim_split_flag: bool = False) -> None:
    log.info("═" * 60)
    log.info("Starting training — model=%s version=%s", MODEL_NAME, version)
    log.info("═" * 60)

    # 1. Load raw data and separate simulation split (last 15% sorted by customerID)
    raw_df = load_data(data_path)
    train_eligible_raw, sim_raw = make_splits(raw_df)

    if save_sim_split_flag:
        save_sim_split(raw_df)

    # 2. Preprocess train-eligible data
    df = preprocess_raw(train_eligible_raw)

    # 3. Separate features and target
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    log.info("Class distribution: %s", y.value_counts().to_dict())
    churn_rate = y.mean()
    log.info(
        "Churn rate: %.1f%% — WHY ACCURACY ALONE MISLEADS: a trivial model "
        "that always predicts 'No Churn' would score %.1f%% accuracy while "
        "having zero recall for churners. We therefore report precision, recall, "
        "F1, and ROC-AUC.",
        churn_rate * 100,
        (1 - churn_rate) * 100,
    )

    # 4. Train/test split BEFORE any preprocessing fitting — prevents leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info("Split: %d train / %d test", len(X_train), len(X_test))

    # 5. Baseline: majority-class dummy classifier
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_acc = dummy.score(X_test, y_test)
    log.info(
        "Baseline (majority-class) accuracy: %.4f — this is the floor to beat",
        dummy_acc,
    )

    # 6. Build and train the real pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 7. Evaluate
    metrics = evaluate(pipeline, X_test, y_test)

    # 8. Compute training data snapshot for drift detection
    snapshot = compute_training_snapshot(X_train, pipeline)

    # 9. Save artifacts
    artifact_dir = save_artifacts(
        pipeline=pipeline,
        version=version,
        metrics=metrics,
        snapshot=snapshot,
        n_train=len(X_train),
        n_test=len(X_test),
    )

    # 10. Record deployment in DB
    record_deployment(
        model_name=MODEL_NAME,
        version=version,
        git_commit=get_git_commit(),
        eval_metrics=metrics,
    )

    log.info("═" * 60)
    log.info("Training complete. Artifacts at: %s", artifact_dir)
    log.info("═" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Telco Customer Churn CSV file.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag for this training run (e.g. v1, v2, 20240801).",
    )
    parser.add_argument(
        "--save-sim-split",
        action="store_true",
        help="Save the held-out simulation split (15%%) to data/simulation/sim_data.csv.",
    )
    args = parser.parse_args()
    train(
        data_path=args.data_path,
        version=args.version,
        save_sim_split_flag=args.save_sim_split,
    )
