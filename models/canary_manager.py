"""
models/canary_manager.py
────────────────────────
Canary observation monitor for auto-promotion and failure rollback.

RULES:
  1. Auto-Promotion:
     If the canary model version serves >= 50 predictions with 0 alerts fired against it,
     it is automatically promoted to 100% main production model (`current.json` updated to candidate).

  2. Auto-Rollback:
     If a drift or accuracy alert fires against the canary model version specifically,
     the canary is immediately rolled back to 0% (`canary.json` removed, deployment status="canary_failed").
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_models import Deployment  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.db_models import Prediction  # noqa: E402
from models.feature_config import (  # noqa: E402
    CANARY_FILENAME,
    CURRENT_VERSION_FILENAME,
    MODEL_NAME,
)
from monitoring.db_models import Alert  # noqa: E402

log = logging.getLogger("canary_manager")

ARTIFACTS_ROOT = PROJECT_ROOT / "models" / "artifacts"
MIN_CANARY_PREDICTIONS = 50


def evaluate_canary_deployment(db_session_factory=SessionLocal) -> dict:
    """
    Inspect active canary deployment and execute auto-promotion or rollback.
    """
    canary_pointer_path = ARTIFACTS_ROOT / MODEL_NAME / CANARY_FILENAME
    if not canary_pointer_path.exists():
        return {"status": "no_active_canary"}

    try:
        with open(canary_pointer_path) as f:
            canary_info = json.load(f)
    except Exception as exc:
        log.warning("Could not read canary pointer file: %s", exc)
        return {"status": "error", "error": str(exc)}

    canary_version = canary_info.get("canary_version")
    baseline_version = canary_info.get("baseline_version")
    canary_model_id = f"{MODEL_NAME}:{canary_version}"

    db = db_session_factory()
    try:
        # 1. Check for alerts against canary model
        stmt_alert = select(Alert).where(Alert.model_id == canary_model_id)
        alerts_count = len(db.scalars(stmt_alert).all())

        if alerts_count > 0:
            # ALERT ROLLBACK PATH
            log.warning(
                "CANARY FAILED: %d alert(s) fired against canary model %s. Rolling back canary!",
                alerts_count,
                canary_model_id,
            )

            # Remove canary pointer file
            if canary_pointer_path.exists():
                canary_pointer_path.unlink()

            # Update DB deployment record for canary
            d_canary = db.scalars(
                select(Deployment).where(Deployment.model_id == canary_model_id)
            ).first()
            if d_canary:
                d_canary.status = "canary_failed"
                d_canary.traffic_percentage = 0.0
                d_canary.is_current = False
                db.commit()

            return {
                "status": "canary_failed",
                "canary_version": canary_version,
                "alerts_count": alerts_count,
            }

        # 2. Count predictions served by canary version
        stmt_preds = select(Prediction).where(Prediction.model_id == canary_model_id)
        preds_count = len(db.scalars(stmt_preds).all())

        log.info(
            "Canary %s status: %d/%d predictions served with 0 alerts.",
            canary_model_id,
            preds_count,
            MIN_CANARY_PREDICTIONS,
        )

        if preds_count >= MIN_CANARY_PREDICTIONS:
            # AUTO-PROMOTION PATH
            log.info(
                "CANARY PROMOTED: Candidate version '%s' served %d predictions with 0 alerts. "
                "Promoting to 100%% main production model!",
                canary_version,
                preds_count,
            )

            # Update current.json pointer
            current_pointer_path = (
                ARTIFACTS_ROOT / MODEL_NAME / CURRENT_VERSION_FILENAME
            )
            with open(current_pointer_path, "w") as f:
                json.dump({"version": canary_version}, f, indent=2)

            # Remove canary pointer
            if canary_pointer_path.exists():
                canary_pointer_path.unlink()

            # Update DB deployment flags
            d_baseline = db.scalars(
                select(Deployment).where(
                    Deployment.model_id == f"{MODEL_NAME}:{baseline_version}"
                )
            ).first()
            if d_baseline:
                d_baseline.is_current = False
                d_baseline.traffic_percentage = 0.0

            d_canary = db.scalars(
                select(Deployment).where(Deployment.model_id == canary_model_id)
            ).first()
            if d_canary:
                d_canary.is_current = True
                d_canary.status = "deployed"
                d_canary.traffic_percentage = 100.0
                d_canary.promoted_at = datetime.now(timezone.utc)

            db.commit()

            return {
                "status": "promoted",
                "canary_version": canary_version,
                "baseline_version": baseline_version,
                "predictions_count": preds_count,
            }

        return {
            "status": "observing",
            "canary_version": canary_version,
            "predictions_count": preds_count,
            "min_required": MIN_CANARY_PREDICTIONS,
        }

    finally:
        db.close()


if __name__ == "__main__":
    res = evaluate_canary_deployment()
    print(json.dumps(res, indent=2))
