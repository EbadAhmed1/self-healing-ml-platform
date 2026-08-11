"""
monitoring/check_accuracy.py
─────────────────────────────
Rolling Accuracy and Performance Metric Tracking Engine.

USAGE:
  python monitoring/check_accuracy.py [--hours 24] [--min-samples 30] [--model-id churn-model:v1]

DESIGN & ARCHITECTURE:
  1. Inner-joins recent `predictions` with delayed `outcomes` in the database.
  2. Small Sample Size Protection:
     If the number of joined prediction-outcome pairs is less than `min_samples`
     (default: 30), logs a clear "insufficient data" note and SKIPS writing a report.
     This prevents reporting misleading 100% or 0% metrics on 3 data points.
  3. Computes rolling evaluation metrics using scikit-learn:
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC (Area Under ROC Curve)
  4. Writes results to the `accuracy_reports` table.
  5. Alert Triggering:
     If F1-score drops below the baseline threshold (or by >15% from training baseline),
     inserts a row into the `alerts` table (severity: critical).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sqlalchemy import select

# ── Make project root importable when running as a script ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal  # noqa: E402
from app.db_models import Prediction  # noqa: E402
from models.feature_config import (  # noqa: E402
    CURRENT_VERSION_FILENAME,
    METADATA_FILENAME,
    MODEL_NAME,
)
from monitoring.db_models import AccuracyReport, Alert  # noqa: E402
from simulator.db_models import Outcome  # noqa: E402

log = logging.getLogger("accuracy_check")

# Minimum samples required before reporting accuracy metrics
DEFAULT_MIN_SAMPLES = 30
F1_DROP_THRESHOLD = 0.15  # 15% drop from baseline triggers critical alert


# ---------------------------------------------------------------------------
# Metadata Loader
# ---------------------------------------------------------------------------
def load_baseline_metrics(model_name: str = MODEL_NAME) -> tuple[str, dict]:
    """Load baseline evaluation metrics from metadata.json."""
    artifacts_dir = PROJECT_ROOT / "models" / "artifacts" / model_name
    pointer_path = artifacts_dir / CURRENT_VERSION_FILENAME

    if not pointer_path.exists():
        return f"{model_name}:v1", {}

    with open(pointer_path) as f:
        pointer = json.load(f)

    version = pointer["version"]
    model_id = f"{model_name}:{version}"
    metadata_path = Path(pointer["artifact_dir"]) / METADATA_FILENAME

    if not metadata_path.exists():
        metadata_path = artifacts_dir / version / METADATA_FILENAME

    if not metadata_path.exists():
        return model_id, {}

    with open(metadata_path) as f:
        metadata = json.load(f)

    metrics = metadata.get("eval_metrics", {})
    return model_id, metrics


# ---------------------------------------------------------------------------
# Accuracy Evaluation Logic
# ---------------------------------------------------------------------------
def run_accuracy_check(
    hours: int = 24,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    target_model_id: str | None = None,
    db_session_factory=SessionLocal,
) -> AccuracyReport | None:
    """
    Compute rolling accuracy metrics by joining predictions and outcomes.

    Returns:
        AccuracyReport ORM instance if evaluation ran, or None if skipped
        due to insufficient sample size.
    """
    model_id, baseline_metrics = load_baseline_metrics()
    if target_model_id:
        model_id = target_model_id

    db = db_session_factory()
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)

    try:
        # Join predictions and outcomes on prediction_id
        stmt = (
            select(Prediction.prediction, Prediction.confidence, Outcome.true_label)
            .join(Outcome, Prediction.id == Outcome.prediction_id)
            .where(Prediction.model_id == model_id)
            .where(Prediction.created_at >= window_start)
        )
        results = db.execute(stmt).all()
        n_samples = len(results)

        log.info(
            "Evaluating accuracy for %s (window: last %d hours, found %d joined pairs)",
            model_id,
            hours,
            n_samples,
        )

        # ── SMALL SAMPLE GUARD ──────────────────────────────────────────────
        if n_samples < min_samples:
            log.info(
                "INSUFFICIENT DATA: Found %d joined prediction-outcome pairs for %s "
                "(minimum required: %d). Skipping accuracy report creation.",
                n_samples,
                model_id,
                min_samples,
            )
            return None

        # Extract vectors for metric calculations
        preds = [r[0] for r in results]
        confidences = [r[1] for r in results]
        true_labels = [r[2] for r in results]

        # Calculate metrics using scikit-learn
        prec = float(precision_score(true_labels, preds, zero_division=0))
        rec = float(recall_score(true_labels, preds, zero_division=0))
        f1 = float(f1_score(true_labels, preds, zero_division=0))

        # ROC-AUC requires both classes present in ground truth
        try:
            auc = float(roc_auc_score(true_labels, confidences))
        except ValueError:
            auc = (
                0.5  # default neutral score if only one class is present in true_labels
            )

        report = AccuracyReport(
            model_id=model_id,
            window_start=window_start,
            window_end=now,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            roc_auc=round(auc, 4),
            n_samples=n_samples,
            created_at=now,
        )
        db.add(report)

        # Check for performance degradation against baseline
        baseline_f1 = baseline_metrics.get("f1", 0.0)
        if baseline_f1 > 0 and (baseline_f1 - f1) >= F1_DROP_THRESHOLD:
            alert = Alert(
                model_id=model_id,
                alert_type="accuracy_drop",
                severity="critical",
                triggered_at=now,
                details={
                    "current_f1": round(f1, 4),
                    "baseline_f1": round(baseline_f1, 4),
                    "drop": round(baseline_f1 - f1, 4),
                    "n_samples": n_samples,
                    "threshold": F1_DROP_THRESHOLD,
                },
            )
            db.add(alert)
            log.warning(
                "ACCURACY DROP ALERT: F1 dropped from baseline %.4f → %.4f (drop=%.4f >= %.2f)",
                baseline_f1,
                f1,
                baseline_f1 - f1,
                F1_DROP_THRESHOLD,
            )

        db.commit()

        log.info(
            "Accuracy report written for %s (n=%d): P=%.4f R=%.4f F1=%.4f AUC=%.4f",
            model_id,
            n_samples,
            prec,
            rec,
            f1,
            auc,
        )
        return report

    except Exception:
        log.exception("Accuracy check failed for %s", model_id)
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute rolling accuracy metrics by joining predictions and outcomes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Time window in hours of recent predictions to evaluate.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum required prediction-outcome pairs before generating a report.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID to evaluate. Defaults to current.json.",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    run_accuracy_check(
        hours=args.hours, min_samples=args.min_samples, target_model_id=args.model_id
    )
