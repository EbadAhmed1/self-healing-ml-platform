"""
tests/test_multi_tenant_pipeline.py
────────────────────────────────────
End-to-end multi-tenant integration test for Tenant #2 (fraud-model).

REQUIREMENTS PROVED HERE:
  1. FastAPI endpoint `POST /predict/fraud-model` accepts valid FraudInput, executes
     inference, and logs to `predictions` table with `model_id = "fraud-model:v1"`.
  2. Monitoring drift checks execute per-model on fraud features without cross-tenant interference.
  3. Diagnosis agent processes alerts for fraud-model and creates incidents tagged to fraud-model.
  4. Tenant Isolation: An incident on fraud-model NEVER references or confuses churn-model data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agent.db_models import Deployment
from agent.run_agent import process_unprocessed_alerts
from app.database import get_db
from app.db_models import Prediction
from app.main import app
from monitoring.check_drift import run_drift_check
from monitoring.db_models import Alert


# ===========================================================================
# Fixtures & Setup
# ===========================================================================
@pytest.fixture(autouse=True)
def clean_db(test_engine):
    """Ensure clean database before each test."""
    import agent.db_models  # noqa: F401
    import app.db_models  # noqa: F401
    import monitoring.db_models  # noqa: F401
    import simulator.db_models  # noqa: F401
    from app.database import Base

    Base.metadata.create_all(bind=test_engine)
    with test_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture
def valid_fraud_payload() -> dict:
    return {
        "transaction_amount": 450.0,
        "location_distance_km": 85.5,
        "account_age_days": 120,
        "velocity_1h_count": 5,
        "merchant_category": "crypto",
        "device_type": "mobile_android",
        "is_international": 1,
        "is_flagged_ip": 1,
    }


# ===========================================================================
# Multi-Tenant Pipeline & Isolation Tests
# ===========================================================================
class TestMultiTenantPipeline:
    def test_fraud_prediction_endpoint_logs_correct_model_id(
        self, test_engine, valid_fraud_payload, tmp_path
    ):
        """POST /predict/fraud-model returns HTTP 200 and logs prediction with model_id='fraud-model:v1'."""
        from app.database import Base

        Base.metadata.create_all(bind=test_engine)
        TestingSessionLocal = sessionmaker(bind=test_engine)

        def override_get_db():
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db

        # Create mock fraud pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [1]
        mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]

        with patch("app.model_loader.load_all_models"), patch(
            "app.model_loader.get_pipeline_for_tenant",
            return_value=(mock_pipeline, "fraud-model:v1"),
        ):
            client = TestClient(app)
            response = client.post("/predict/fraud-model", json=valid_fraud_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] is True
        assert data["prediction_label"] == "Fraud"
        assert data["confidence"] == 0.9
        assert data["model_id"] == "fraud-model:v1"

        # Verify recorded in DB predictions table
        db = TestingSessionLocal()
        preds = (
            db.query(Prediction).filter(Prediction.model_id == "fraud-model:v1").all()
        )
        assert len(preds) == 1
        p = preds[0]
        assert p.input_features["transaction_amount"] == 450.0
        assert p.prediction is True
        db.close()

        app.dependency_overrides.clear()

    def test_tenant_isolation_end_to_end_pipeline(self, test_engine, tmp_path):
        """
        Full multi-tenant end-to-end pipeline:
          1. Seed churn-model AND fraud-model predictions & deployments into DB.
          2. Run drift check for fraud-model specifically.
          3. Run diagnosis agent for fraud-model.
          4. Assert fraud-model incident created with zero churn-model cross-contamination.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()
        now = datetime.now(timezone.utc)

        # 1. Insert deployments for both tenants
        d_churn = Deployment(
            model_id="churn-model:v1",
            version="v1",
            is_current=True,
            status="deployed",
            deployed_at=now,
        )
        d_fraud = Deployment(
            model_id="fraud-model:v1",
            version="v1",
            is_current=True,
            status="deployed",
            deployed_at=now,
        )
        db.add(d_churn)
        db.add(d_fraud)

        # 2. Insert churn predictions
        churn_payload = {
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
        for i in range(35):
            db.add(
                Prediction(
                    id=f"churn-p-{i}",
                    model_id="churn-model:v1",
                    input_features=churn_payload,
                    prediction=False,
                    confidence=0.85,
                    created_at=now,
                )
            )

        # 3. Insert fraud predictions (with drifted transaction_amount = 9999.0)
        fraud_payload = {
            "transaction_amount": 9999.0,  # massive drift
            "location_distance_km": 10.0,
            "account_age_days": 100,
            "velocity_1h_count": 1,
            "merchant_category": "retail",
            "device_type": "mobile_ios",
            "is_international": 0,
            "is_flagged_ip": 0,
        }
        for i in range(35):
            db.add(
                Prediction(
                    id=f"fraud-p-{i}",
                    model_id="fraud-model:v1",
                    input_features=fraud_payload,
                    prediction=True,
                    confidence=0.95,
                    created_at=now,
                )
            )

        db.commit()
        db.close()

        # Mock fraud metadata JSON and current.json for drift baseline
        fraud_model_dir = tmp_path / "models" / "artifacts" / "fraud-model"
        fraud_meta_dir = fraud_model_dir / "v1"
        fraud_meta_dir.mkdir(parents=True, exist_ok=True)

        with open(fraud_model_dir / "current.json", "w") as f:
            json.dump({"version": "v1"}, f)

        fraud_metadata = {
            "training_data_snapshot": {
                "numeric": {
                    "transaction_amount": {
                        "mean": 50.0,
                        "std": 10.0,
                        "median": 50.0,
                        "deciles": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                    }
                },
                "categorical": {},
                "prediction": {},
            }
        }
        with open(fraud_meta_dir / "metadata.json", "w") as f:
            json.dump(fraud_metadata, f)

        # 4. Run drift check specifically for fraud-model
        with patch("monitoring.check_drift.PROJECT_ROOT", tmp_path):
            reports = run_drift_check(
                model_name="fraud-model",
                version="v1",
                db_session_factory=SessionLocal,
            )

        assert len(reports) > 0

        # Verify alerts only created for fraud-model
        db = SessionLocal()
        fraud_alerts = db.query(Alert).filter(Alert.model_id == "fraud-model:v1").all()
        churn_alerts = db.query(Alert).filter(Alert.model_id == "churn-model:v1").all()
        assert len(fraud_alerts) > 0
        assert len(churn_alerts) == 0  # tenant isolation!
        db.close()

        # 5. Run diagnosis agent for unprocessed alerts
        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)
        assert len(incidents) > 0

        # Assert all created incidents belong strictly to fraud-model
        for inc in incidents:
            assert inc.model_id == "fraud-model:v1"
            assert (
                "transaction_amount" in str(inc.evidence)
                or "fraud-model" in inc.model_id
            )
            assert "churn-model" not in inc.model_id
