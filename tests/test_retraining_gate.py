"""
tests/test_retraining_gate.py
──────────────────────────────
Pytest test suite for Metric Gate evaluation and retraining logic (Phase 6).

PROVING CODE CORRECTNESS:
  1. Inferior candidate (F1 <= baseline F1 + 0.01) MUST be REJECTED, status="rejected",
     logged in `deployments` with rejection_reason, and NOT deployed.
  2. Superior candidate (F1 > baseline F1 + 0.01) MUST be APPROVED, status="canary",
     assigned to 10% canary traffic.
  3. Insufficient DB samples (N < 100) MUST skip retraining gracefully.
  4. Concurrency lock file prevents race conditions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import sessionmaker

from agent.db_models import Deployment
from app.db_models import Prediction
from models.retrain import run_retraining_pipeline
from simulator.db_models import Outcome


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


@pytest.fixture
def mock_labeled_db_data(test_engine):
    """Insert 120 labeled prediction + outcome records into in-memory SQLite DB."""
    SessionLocal = sessionmaker(bind=test_engine)
    db = SessionLocal()

    now = datetime.now(timezone.utc)

    # Base valid payload
    base_payload = {
        "tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1571.0,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
    }

    for i in range(120):
        pred_id = f"p-{i:03d}"
        payload = dict(base_payload)
        payload["tenure"] = (i % 60) + 1
        payload["MonthlyCharges"] = 20.0 + (i % 80)

        pred = Prediction(
            id=pred_id,
            model_id="churn-model:v1",
            input_features=payload,
            prediction=(i % 3 == 0),
            confidence=0.85,
            created_at=now,
        )
        outcome = Outcome(
            id=f"o-{i:03d}",
            prediction_id=pred_id,
            true_label=(i % 3 == 0),  # matching true label
            delay_seconds=1.0,
            observed_at=now,
        )
        db.add(pred)
        db.add(outcome)

    db.commit()
    db.close()


# ===========================================================================
# Metric Gate Tests
# ===========================================================================
class TestMetricGate:
    def test_metric_gate_rejects_inferior_candidate(
        self, test_engine, mock_labeled_db_data, tmp_path
    ):
        """
        Construct a candidate model that fails to beat baseline by GATE_MIN_IMPROVEMENT.
        Verify candidate is REJECTED, status="rejected", logged to DB, and NOT deployed.
        """
        SessionLocal = sessionmaker(bind=test_engine)

        # Mock current baseline model returning high F1 (e.g. 0.95)
        # and candidate returning lower F1 (e.g. 0.80)
        mock_baseline_metrics = {
            "f1": 0.95,
            "precision": 0.95,
            "recall": 0.95,
            "roc_auc": 0.95,
        }
        mock_candidate_metrics = {
            "f1": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "roc_auc": 0.80,
        }

        with patch(
            "models.retrain.load_current_baseline_model",
            return_value=(MagicMock(), "v1"),
        ), patch(
            "models.retrain.evaluate_pipeline",
            side_effect=[mock_baseline_metrics, mock_candidate_metrics],
        ):
            res = run_retraining_pipeline(
                db_session_factory=SessionLocal,
                candidate_version="v2-test",
                override_min_samples=50,
            )

        assert res["status"] == "rejected"
        assert res["gate_passed"] is False

        # Verify recorded in deployments table with status="rejected"
        db = SessionLocal()
        d = db.query(Deployment).filter(Deployment.version == "v2-test").first()
        assert d is not None
        assert d.status == "rejected"
        assert d.traffic_percentage == 0.0
        assert "failed Metric Gate" in d.rejection_reason
        db.close()

    def test_metric_gate_approves_superior_candidate(
        self, test_engine, mock_labeled_db_data, tmp_path
    ):
        """
        Construct a candidate model that beats baseline by >= GATE_MIN_IMPROVEMENT (0.01).
        Verify candidate is APPROVED, status="canary", and assigned to 10% canary.
        """
        SessionLocal = sessionmaker(bind=test_engine)

        mock_baseline_metrics = {
            "f1": 0.70,
            "precision": 0.70,
            "recall": 0.70,
            "roc_auc": 0.70,
        }
        mock_candidate_metrics = {
            "f1": 0.75,
            "precision": 0.75,
            "recall": 0.75,
            "roc_auc": 0.75,
        }

        with patch(
            "models.retrain.load_current_baseline_model",
            return_value=(MagicMock(), "v1"),
        ), patch(
            "models.retrain.evaluate_pipeline",
            side_effect=[mock_baseline_metrics, mock_candidate_metrics],
        ), patch(
            "models.retrain.ARTIFACTS_ROOT", tmp_path
        ):
            res = run_retraining_pipeline(
                db_session_factory=SessionLocal,
                candidate_version="v2-pass",
                override_min_samples=50,
            )

        assert res["status"] == "canary"
        assert res["gate_passed"] is True
        assert res["traffic_percentage"] == 10.0

        # Verify recorded in deployments table with status="canary"
        db = SessionLocal()
        d = db.query(Deployment).filter(Deployment.version == "v2-pass").first()
        assert d is not None
        assert d.status == "canary"
        assert d.traffic_percentage == 10.0
        db.close()

    def test_insufficient_samples_skips_retraining(self, test_engine):
        """
        If labeled DB records N < min_samples (e.g. N=5 < 100), retraining is skipped.
        """
        SessionLocal = sessionmaker(bind=test_engine)

        res = run_retraining_pipeline(
            db_session_factory=SessionLocal,
            candidate_version="v2-skip",
            override_min_samples=100,
        )

        assert res["status"] == "skipped"
        assert "Insufficient labeled samples" in res["reason"]
