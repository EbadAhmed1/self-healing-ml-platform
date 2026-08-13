"""
monitoring/check_drift.py
─────────────────────────
Feature and prediction output drift detection runner.

Calculates Population Stability Index (PSI) per feature and prediction
confidence by comparing recent production data (from the `predictions` table)
against baseline distributions saved in metadata.json at model training time.

SUPPORTED TENANTS:
  - Dynamically detects features and baseline distributions from metadata.json
    for any model_name/model_id (churn-model, fraud-model, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

# ── Make project root importable when running as a script ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal  # noqa: E402
from app.db_models import Prediction  # noqa: E402
from models.feature_config import (  # noqa: E402
    CATEGORICAL_FEATURES,  # noqa: F401
    CURRENT_VERSION_FILENAME,
    METADATA_FILENAME,
    MODEL_NAME,
    NUMERIC_FEATURES,  # noqa: F401
)
from monitoring.db_models import Alert, DriftReport  # noqa: E402
from monitoring.psi import (  # noqa: E402
    MIN_SAMPLE_SIZE,
    calculate_categorical_psi,
    calculate_numeric_psi,
    classify_psi,
)

log = logging.getLogger("drift_check")


# ---------------------------------------------------------------------------
# Metadata & Baseline Loader
# ---------------------------------------------------------------------------
def load_baseline_snapshot(
    model_name: str = MODEL_NAME, version: str | None = None
) -> tuple[str, dict]:
    """
    Load the training snapshot dictionary from the current model's metadata.json.
    """
    try:
        from app.model_loader import download_from_hf_hub

        download_from_hf_hub(model_name, PROJECT_ROOT / "models" / "artifacts")
    except Exception:
        pass

    artifacts_dir = PROJECT_ROOT / "models" / "artifacts" / model_name

    if version is None:
        pointer_path = artifacts_dir / CURRENT_VERSION_FILENAME
        if not pointer_path.exists():
            log.warning(
                "Model version pointer not found for '%s' at: %s — skipping drift check.",
                model_name,
                pointer_path,
            )
            return f"{model_name}:v1", {}
        with open(pointer_path) as f:
            pointer = json.load(f)
        version = pointer["version"]

    model_id = f"{model_name}:{version}"
    metadata_path = artifacts_dir / version / METADATA_FILENAME

    if not metadata_path.exists():
        log.warning(
            "Metadata file not found for '%s' at: %s — skipping drift check.",
            model_name,
            metadata_path,
        )
        return model_id, {}

    with open(metadata_path) as f:
        meta = json.load(f)

    snapshot = meta.get("training_data_snapshot", {})
    return model_id, snapshot


# ---------------------------------------------------------------------------
# Main Drift Check Function
# ---------------------------------------------------------------------------
def run_drift_check(
    hours: int = 24,
    target_model_id: str | None = None,
    model_name: str = MODEL_NAME,
    version: str | None = None,
    db_session_factory=SessionLocal,
) -> list[DriftReport]:
    """
    Run PSI drift analysis for input features and prediction outputs.

    Returns:
        list[DriftReport]: List of created DriftReport ORM instances.
    """
    model_id, snapshot = load_baseline_snapshot(model_name=model_name, version=version)
    if target_model_id:
        model_id = target_model_id

    db = db_session_factory()
    reports: list[DriftReport] = []
    alerts: list[Alert] = []

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(Prediction)
            .where(Prediction.model_id == model_id)
            .where(Prediction.created_at >= cutoff)
        )
        recent_preds = db.scalars(stmt).all()
        sample_size = len(recent_preds)

        log.info(
            "Running drift check for %s (last %d hours, %d predictions found)",
            model_id,
            hours,
            sample_size,
        )

        if sample_size < MIN_SAMPLE_SIZE:
            log.warning(
                "Insufficient sample size for %s: %d predictions found (minimum required: %d). "
                "Writing 'insufficient_data' report and skipping drift alert generation.",
                model_id,
                sample_size,
                MIN_SAMPLE_SIZE,
            )
            report = DriftReport(
                model_id=model_id,
                feature_name="_all_features",
                psi_score=0.0,
                method="PSI",
                sample_size=sample_size,
                status="insufficient_data",
                checked_at=datetime.now(timezone.utc),
            )
            db.add(report)
            db.commit()
            return [report]

        feature_dicts = [p.input_features for p in recent_preds]

        # ── 1. NUMERIC FEATURE DRIFT ─────────────────────────────────────────
        num_snapshot = snapshot.get("numeric", {})
        for feature, base_info in num_snapshot.items():
            current_vals = [
                d[feature]
                for d in feature_dicts
                if isinstance(d, dict) and d.get(feature) is not None
            ]

            psi = calculate_numeric_psi(base_info, current_vals)
            status = classify_psi(psi, sample_size)

            report = DriftReport(
                model_id=model_id,
                feature_name=feature,
                psi_score=psi,
                method="PSI",
                sample_size=sample_size,
                status=status,
                checked_at=datetime.now(timezone.utc),
            )
            reports.append(report)
            db.add(report)

            if status in ("moderate", "significant"):
                severity = "critical" if status == "significant" else "warning"
                alert = Alert(
                    model_id=model_id,
                    alert_type="feature_drift",
                    severity=severity,
                    triggered_at=datetime.now(timezone.utc),
                    details={
                        "feature": feature,
                        "psi_score": psi,
                        "status": status,
                        "sample_size": sample_size,
                        "baseline_mean": base_info.get("mean"),
                        "current_mean": (
                            float(sum(current_vals) / len(current_vals))
                            if current_vals
                            else None
                        ),
                    },
                )
                alerts.append(alert)
                db.add(alert)

        # ── 2. CATEGORICAL FEATURE DRIFT ──────────────────────────────────────
        cat_snapshot = snapshot.get("categorical", {})
        for feature, base_info in cat_snapshot.items():
            current_vals = [
                str(d[feature])
                for d in feature_dicts
                if isinstance(d, dict) and d.get(feature) is not None
            ]

            psi = calculate_categorical_psi(base_info, current_vals)
            status = classify_psi(psi, sample_size)

            report = DriftReport(
                model_id=model_id,
                feature_name=feature,
                psi_score=psi,
                method="PSI",
                sample_size=sample_size,
                status=status,
                checked_at=datetime.now(timezone.utc),
            )
            reports.append(report)
            db.add(report)

            if status in ("moderate", "significant"):
                severity = "critical" if status == "significant" else "warning"
                alert = Alert(
                    model_id=model_id,
                    alert_type="feature_drift",
                    severity=severity,
                    triggered_at=datetime.now(timezone.utc),
                    details={
                        "feature": feature,
                        "psi_score": psi,
                        "status": status,
                        "sample_size": sample_size,
                    },
                )
                alerts.append(alert)
                db.add(alert)

        # ── 3. PREDICTION OUTPUT DRIFT ────────────────────────────────────────
        pred_snapshot = snapshot.get("prediction", {})
        confidences = [p.confidence for p in recent_preds if p.confidence is not None]
        if pred_snapshot and confidences:
            psi = calculate_numeric_psi(pred_snapshot, confidences)
            status = classify_psi(psi, sample_size)

            report = DriftReport(
                model_id=model_id,
                feature_name="_prediction",
                psi_score=psi,
                method="PSI",
                sample_size=sample_size,
                status=status,
                checked_at=datetime.now(timezone.utc),
            )
            reports.append(report)
            db.add(report)

            if status in ("moderate", "significant"):
                severity = "critical" if status == "significant" else "warning"
                alert = Alert(
                    model_id=model_id,
                    alert_type="prediction_drift",
                    severity=severity,
                    triggered_at=datetime.now(timezone.utc),
                    details={
                        "feature": "_prediction",
                        "psi_score": psi,
                        "status": status,
                        "sample_size": sample_size,
                    },
                )
                alerts.append(alert)
                db.add(alert)

        db.commit()

        dr_count = len(reports)
        alert_count = len(alerts)

        log.info(
            "Drift check complete for %s. Created %d report(s), fired %d alert(s).",
            model_id,
            dr_count,
            alert_count,
        )

        return reports

    except Exception:
        db.rollback()
        log.exception("Error executing drift check for model_id=%s", target_model_id)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature drift check.")
    parser.add_argument("--hours", type=int, default=24, help="Window size in hours")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument(
        "--target-model-id", type=str, default=None, help="Explicit model_id"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_drift_check(
        hours=args.hours,
        target_model_id=args.target_model_id,
        model_name=args.model_name,
    )
