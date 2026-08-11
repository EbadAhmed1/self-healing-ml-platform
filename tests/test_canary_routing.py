"""
tests/test_canary_routing.py
──────────────────────────────
Pytest test suite for Canary Deployment Traffic Routing, Auto-Promotion, and Failure Rollback.

Tests cover:
  1. Pseudo-random traffic routing: 10% canary traffic served by canary pipeline and logged to `predictions.model_id`.
  2. Auto-promotion: Canary serving >= 50 predictions with 0 alerts auto-promoted to 100% main model.
  3. Alert rollback: Alert fired against canary model immediately rolls canary to 0% (`status="canary_failed"`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import sessionmaker

from agent.db_models import Deployment
from app import model_loader
from app.db_models import Prediction
from models.canary_manager import evaluate_canary_deployment
from monitoring.db_models import Alert


# ===========================================================================
# Fixtures & Setup
# ===========================================================================
@pytest.fixture(autouse=True)
def clean_db(test_engine):
    """Ensure clean database before each test."""
    from app.database import Base

    with test_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


# ===========================================================================
# Canary Routing & Manager Tests
# ===========================================================================
class TestCanaryRoutingAndManager:
    def test_canary_traffic_routing(self):
        """
        Verify pseudo-random roll selects canary pipeline when roll < traffic_percentage.
        """
        mock_main = MagicMock()
        mock_canary = MagicMock()

        model_loader._pipeline = mock_main
        model_loader._model_id = "churn-model:v1"
        model_loader._canary_pipeline = mock_canary
        model_loader._canary_model_id = "churn-model:v2"
        model_loader._canary_traffic_pct = 10.0

        with patch("random.uniform", return_value=5.0):
            pipe, m_id = model_loader.get_pipeline_for_request()
            assert pipe == mock_canary
            assert m_id == "churn-model:v2"

        with patch("random.uniform", return_value=50.0):
            pipe, m_id = model_loader.get_pipeline_for_request()
            assert pipe == mock_main
            assert m_id == "churn-model:v1"

    def test_canary_auto_promotion(self, test_engine, tmp_path):
        """
        Canary serving >= 50 predictions with 0 alerts auto-promoted to 100% main model.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        # Create baseline and canary deployment rows
        d1 = Deployment(
            model_id="churn-model:v1",
            version="v1",
            is_current=True,
            status="deployed",
            traffic_percentage=100.0,
            deployed_at=now,
        )
        d2 = Deployment(
            model_id="churn-model:v2",
            version="v2",
            is_current=False,
            status="canary",
            traffic_percentage=10.0,
            deployed_at=now,
        )
        db.add(d1)
        db.add(d2)

        # Insert 55 predictions served by canary model
        for i in range(55):
            pred = Prediction(
                id=f"canary-p-{i}",
                model_id="churn-model:v2",
                input_features={"tenure": 10},
                prediction=False,
                confidence=0.9,
                created_at=now,
            )
            db.add(pred)

        db.commit()
        db.close()

        # Mock pointer files
        churn_dir = tmp_path / "churn-model"
        churn_dir.mkdir(parents=True, exist_ok=True)
        canary_file = churn_dir / "canary.json"
        current_file = churn_dir / "current.json"

        with open(canary_file, "w") as f:
            json.dump(
                {
                    "canary_version": "v2",
                    "baseline_version": "v1",
                    "status": "active",
                },
                f,
            )
        with open(current_file, "w") as f:
            json.dump({"version": "v1"}, f)

        with patch("models.canary_manager.ARTIFACTS_ROOT", tmp_path):
            res = evaluate_canary_deployment(db_session_factory=SessionLocal)

        assert res["status"] == "promoted"
        assert res["canary_version"] == "v2"
        assert not canary_file.exists()

        # Verify current.json updated to v2
        with open(current_file) as f:
            curr = json.load(f)
        assert curr["version"] == "v2"

        # Verify DB updated
        db = SessionLocal()
        d2_updated = db.query(Deployment).filter(Deployment.version == "v2").first()
        assert d2_updated.status == "deployed"
        assert d2_updated.is_current is True
        assert d2_updated.traffic_percentage == 100.0
        db.close()

    def test_canary_alert_rollback(self, test_engine, tmp_path):
        """
        Alert fired against canary model triggers instant rollback to 0%.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        d2 = Deployment(
            model_id="churn-model:v2",
            version="v2",
            is_current=False,
            status="canary",
            traffic_percentage=10.0,
            deployed_at=now,
        )
        alert = Alert(
            id="alert-canary-fail",
            model_id="churn-model:v2",
            alert_type="feature_drift",
            severity="critical",
            triggered_at=now,
            details={"feature": "tenure"},
            processed=False,
        )
        db.add(d2)
        db.add(alert)
        db.commit()
        db.close()

        churn_dir = tmp_path / "churn-model"
        churn_dir.mkdir(parents=True, exist_ok=True)
        canary_file = churn_dir / "canary.json"
        with open(canary_file, "w") as f:
            json.dump(
                {
                    "canary_version": "v2",
                    "baseline_version": "v1",
                    "status": "active",
                },
                f,
            )

        with patch("models.canary_manager.ARTIFACTS_ROOT", tmp_path):
            res = evaluate_canary_deployment(db_session_factory=SessionLocal)

        assert res["status"] == "canary_failed"
        assert not canary_file.exists()

        # Verify DB deployment updated to canary_failed
        db = SessionLocal()
        d2_updated = db.query(Deployment).filter(Deployment.version == "v2").first()
        assert d2_updated.status == "canary_failed"
        assert d2_updated.traffic_percentage == 0.0
        assert d2_updated.is_current is False
        db.close()
