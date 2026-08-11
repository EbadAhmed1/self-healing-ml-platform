"""
models/retrain.py
─────────────────
Automated production retraining script & Metric Gate evaluator.

REQUIREMENTS:
  1. Trigger: Triggered when confirmed drift/accuracy alerts exist.
  2. Data Extraction: Pulls recent labeled production data (predictions + outcomes).
     Min sample guard: MIN_RETRAIN_SAMPLES = 100.
  3. Metric Gate: Candidate MUST beat baseline F1 on the held-out validation set
     by GATE_MIN_IMPROVEMENT = 0.01 (1.0% F1 margin).
     - If candidate fails: status="rejected", logged to `deployments` table, NOT deployed.
     - If candidate passes: status="canary", assigned to 10% canary deployment.
  4. Concurrency: Lock file `retraining.lock` prevents concurrent runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import select

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_models import Deployment  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.db_models import Prediction  # noqa: E402
from models.feature_config import (  # noqa: E402
    ALL_FEATURES,
    ARTIFACT_FILENAME,
    CANARY_FILENAME,
    CURRENT_VERSION_FILENAME,
    METADATA_FILENAME,
    MODEL_NAME,
    TARGET_COL,
)
from models.pipeline_shared import (  # noqa: E402
    build_pipeline,
    compute_training_snapshot,
    evaluate_pipeline,
    preprocess_raw,
)
from simulator.db_models import Outcome  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("retrain")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_RETRAIN_SAMPLES = 100
GATE_MIN_IMPROVEMENT = 0.01  # 1.0% F1 score improvement margin requirement
LOCK_FILE_PATH = PROJECT_ROOT / "models" / "retraining.lock"
ARTIFACTS_ROOT = PROJECT_ROOT / "models" / "artifacts"


# ---------------------------------------------------------------------------
# Data Extractor
# ---------------------------------------------------------------------------
def fetch_labeled_data_from_db(db) -> pd.DataFrame:
    """
    Query predictions joined with outcomes from the database.

    Returns:
        DataFrame containing input features and true_label (mapped to Churn 1/0).
    """
    stmt = (
        select(Prediction, Outcome)
        .join(Outcome, Prediction.id == Outcome.prediction_id)
        .order_by(Prediction.created_at.asc())
    )
    results = db.execute(stmt).all()

    if not results:
        return pd.DataFrame()

    rows = []
    for pred, outcome in results:
        row = dict(pred.input_features)
        row[TARGET_COL] = 1 if outcome.true_label else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Baseline Model Loader
# ---------------------------------------------------------------------------
def load_current_baseline_model() -> tuple[Any | None, str]:
    """Load the current production model pipeline and version tag."""
    pointer_path = ARTIFACTS_ROOT / MODEL_NAME / CURRENT_VERSION_FILENAME
    if not pointer_path.exists():
        return None, "none"

    try:
        with open(pointer_path) as f:
            pointer = json.load(f)
        version = pointer["version"]
        artifact_path = ARTIFACTS_ROOT / MODEL_NAME / version / ARTIFACT_FILENAME
        if artifact_path.exists():
            pipeline = joblib.load(artifact_path)
            return pipeline, version
    except Exception as exc:
        log.warning("Could not load current baseline model: %s", exc)

    return None, "none"


# ---------------------------------------------------------------------------
# Metric Gate & Retraining Logic
# ---------------------------------------------------------------------------
def run_retraining_pipeline(
    db_session_factory=SessionLocal,
    candidate_version: str | None = None,
    override_min_samples: int | None = None,
) -> dict:
    """
    Execute full retraining pipeline with Metric Gate evaluation.

    Returns:
        Summary dict containing status, metrics, and gate decision.
    """
    min_samples = override_min_samples or MIN_RETRAIN_SAMPLES

    # Lock check for race prevention
    if LOCK_FILE_PATH.exists():
        log.warning(
            "Retraining lock file exists at %s — skipping concurrent run.",
            LOCK_FILE_PATH,
        )
        return {"status": "locked", "reason": "Concurrent retraining run active"}

    # Acquire lock
    try:
        LOCK_FILE_PATH.touch()
    except Exception:
        pass

    try:
        db = db_session_factory()
        try:
            # 1. Fetch labeled DB data
            df_raw = fetch_labeled_data_from_db(db)
            n_samples = len(df_raw)

            if n_samples < min_samples:
                msg = f"Insufficient labeled samples (N={n_samples} < {min_samples}) — skipping retraining."
                log.info(msg)
                return {"status": "skipped", "reason": msg, "n_samples": n_samples}

            log.info("Fetched %d labeled records from DB for retraining.", n_samples)

            # 2. Preprocess data
            df = preprocess_raw(df_raw)
            feature_cols = [c for c in ALL_FEATURES if c in df.columns]
            X = df[feature_cols]
            y = df[TARGET_COL]

            # Split 80% train / 20% held-out val
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )

            # 3. Load baseline model & evaluate on X_val
            baseline_pipeline, baseline_version = load_current_baseline_model()
            baseline_f1 = 0.0
            if baseline_pipeline is not None:
                baseline_metrics = evaluate_pipeline(baseline_pipeline, X_val, y_val)
                baseline_f1 = baseline_metrics["f1"]
                log.info(
                    "Current baseline (%s) F1 on held-out set: %.4f",
                    baseline_version,
                    baseline_f1,
                )
            else:
                log.info("No current baseline found — baseline F1 set to 0.0")

            # 4. Train candidate model
            if candidate_version is None:
                # Auto-increment version e.g. v2
                try:
                    num = int(baseline_version.lstrip("v")) + 1
                    candidate_version = f"v{num}"
                except Exception:
                    candidate_version = f"retrain-{int(datetime.now().timestamp())}"

            log.info("Training candidate model version '%s'...", candidate_version)
            candidate_pipeline = build_pipeline(random_state=42)
            candidate_pipeline.fit(X_train, y_train)

            candidate_metrics = evaluate_pipeline(candidate_pipeline, X_val, y_val)
            candidate_f1 = candidate_metrics["f1"]
            log.info(
                "Candidate model (%s) F1 on held-out set: %.4f",
                candidate_version,
                candidate_f1,
            )

            # 5. METRIC GATE EVALUATION
            required_f1 = baseline_f1 + GATE_MIN_IMPROVEMENT
            gate_passed = candidate_f1 >= required_f1

            model_id = f"{MODEL_NAME}:{candidate_version}"
            data_summary = {
                "n_samples": n_samples,
                "data_source": "db_joined_predictions_outcomes",
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            }

            if not gate_passed:
                # REJECTION PATH
                rejection_msg = (
                    f"Candidate F1 ({candidate_f1:.4f}) failed Metric Gate against "
                    f"baseline F1 ({baseline_f1:.4f}) + required margin ({GATE_MIN_IMPROVEMENT:.4f}). "
                    f"Required: >={required_f1:.4f}."
                )
                log.warning("METRIC GATE DECISION: REJECTED! %s", rejection_msg)

                # Record rejection in deployments table
                rejected_deploy = Deployment(
                    model_id=model_id,
                    version=candidate_version,
                    git_commit="unknown",
                    eval_metrics=candidate_metrics,
                    is_current=False,
                    status="rejected",
                    training_data_summary=data_summary,
                    traffic_percentage=0.0,
                    rejection_reason=rejection_msg,
                    deployed_at=datetime.now(timezone.utc),
                )
                db.add(rejected_deploy)
                db.commit()

                return {
                    "status": "rejected",
                    "gate_passed": False,
                    "candidate_version": candidate_version,
                    "baseline_version": baseline_version,
                    "candidate_f1": candidate_f1,
                    "baseline_f1": baseline_f1,
                    "rejection_reason": rejection_msg,
                }

            # APPROVAL PATH -> CANARY PROMOTION
            log.info(
                "METRIC GATE DECISION: APPROVED! Candidate F1 (%.4f) >= Required (%.4f)",
                candidate_f1,
                required_f1,
            )

            # Save artifact & metadata
            artifact_dir = ARTIFACTS_ROOT / MODEL_NAME / candidate_version
            artifact_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(candidate_pipeline, artifact_dir / ARTIFACT_FILENAME)

            snapshot = compute_training_snapshot(X_train, candidate_pipeline)
            metadata = {
                "model_name": MODEL_NAME,
                "version": candidate_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "eval_metrics": candidate_metrics,
                "baseline_eval_metrics": {
                    "baseline_version": baseline_version,
                    "f1": baseline_f1,
                },
                "data_summary": data_summary,
                "training_snapshot": snapshot,
            }
            with open(artifact_dir / METADATA_FILENAME, "w") as f:
                json.dump(metadata, f, indent=2)

            # Create canary pointer file
            canary_pointer = {
                "canary_version": candidate_version,
                "baseline_version": baseline_version,
                "traffic_percentage": 10.0,
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            canary_pointer_path = ARTIFACTS_ROOT / MODEL_NAME / CANARY_FILENAME
            with open(canary_pointer_path, "w") as f:
                json.dump(canary_pointer, f, indent=2)

            # Write deployment row with status="canary"
            canary_deploy = Deployment(
                model_id=model_id,
                version=candidate_version,
                git_commit="unknown",
                eval_metrics=candidate_metrics,
                is_current=False,
                status="canary",
                training_data_summary=data_summary,
                traffic_percentage=10.0,
                deployed_at=datetime.now(timezone.utc),
            )
            db.add(canary_deploy)
            db.commit()

            log.info(
                "Canary deployment initiated for %s with 10%% traffic allocation.",
                model_id,
            )

            return {
                "status": "canary",
                "gate_passed": True,
                "candidate_version": candidate_version,
                "baseline_version": baseline_version,
                "candidate_f1": candidate_f1,
                "baseline_f1": baseline_f1,
                "traffic_percentage": 10.0,
            }

        finally:
            db.close()

    finally:
        # Release lock
        if LOCK_FILE_PATH.exists():
            try:
                LOCK_FILE_PATH.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated ML Retraining Pipeline")
    parser.add_argument(
        "--version", type=str, default=None, help="Candidate version tag (e.g. v2)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Min DB samples required to retrain",
    )
    args = parser.parse_args()

    res = run_retraining_pipeline(
        candidate_version=args.version, override_min_samples=args.min_samples
    )
    print(json.dumps(res, indent=2))
