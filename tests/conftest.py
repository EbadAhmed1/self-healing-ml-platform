"""
tests/conftest.py
──────────────────
Shared pytest fixtures.

DESIGN:
  - Tests NEVER connect to the real Postgres instance. All DB interactions
    use an in-memory SQLite database via a monkeypatched engine.
  - The model pipeline is monkeypatched at the module level so tests never
    need a real trained artifact on disk.
  - The tiny_dataframe fixture provides a 10-row sample for training tests
    that runs in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# In-memory SQLite engine (replaces Postgres for tests)
# ---------------------------------------------------------------------------
SQLITE_URL = "sqlite://"  # pure in-memory, disappears after test session


@pytest.fixture(scope="session")
def test_engine():
    from sqlalchemy.pool import StaticPool

    import agent.db_models  # noqa: F401
    import app.db_models  # noqa: F401
    import monitoring.db_models  # noqa: F401
    import simulator.db_models  # noqa: F401
    from app.database import Base

    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Fresh DB session per test, rolled back after each test."""
    TestingSessionLocal = sessionmaker(
        bind=test_engine, autocommit=False, autoflush=False
    )
    session = TestingSessionLocal()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# Mock pipeline (avoids needing a real trained artifact)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_pipeline():
    """A MagicMock that mimics the sklearn pipeline's predict interface."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([0])  # predict No Churn
    pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])  # 80% No Churn
    return pipeline


# ---------------------------------------------------------------------------
# FastAPI TestClient with mocked model and DB
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client(mock_pipeline, test_engine):
    """
    Return a TestClient with:
      - load_model() monkeypatched so the lifespan startup never hits disk
      - The model_loader module-level state set to the mock pipeline
      - The DB session pointing to in-memory SQLite
    """
    from unittest.mock import patch

    import app.model_loader as model_loader_module
    from app.database import get_db
    from app.main import app

    # Pre-set the module state so get_pipeline() and get_model_id() work
    model_loader_module._pipeline = mock_pipeline
    model_loader_module._model_id = "churn-model:v1-test"

    # Inject SQLite session
    TestingSessionLocal = sessionmaker(
        bind=test_engine, autocommit=False, autoflush=False
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Patch load_model so the lifespan startup doesn't overwrite our mock
    def fake_load_model(registry_path, model_name="churn-model"):
        return mock_pipeline, f"{model_name}:v1-test"

    with patch("app.model_loader.load_model", side_effect=fake_load_model):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Minimal valid payload (shared across API tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_payload() -> dict:
    return {
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


# ---------------------------------------------------------------------------
# Tiny training sample for unit-testing the training script
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_csv(tmp_path) -> Path:
    """
    Write a minimal 50-row CSV that matches the Telco dataset schema.
    50 rows is enough to run the full train → evaluate pipeline in < 1 second.
    """
    n = 50
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n).tolist(),
            "SeniorCitizen": rng.integers(0, 2, n).tolist(),
            "Partner": rng.choice(["Yes", "No"], n).tolist(),
            "Dependents": rng.choice(["Yes", "No"], n).tolist(),
            "tenure": rng.integers(0, 72, n).tolist(),
            "PhoneService": rng.choice(["Yes", "No"], n).tolist(),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n).tolist(),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n).tolist(),
            "OnlineSecurity": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "OnlineBackup": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "DeviceProtection": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n).tolist(),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n).tolist(),
            "StreamingMovies": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n
            ).tolist(),
            "PaperlessBilling": rng.choice(["Yes", "No"], n).tolist(),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ).tolist(),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2).tolist(),
        }
    )

    # TotalCharges: spaces for brand-new customers (tenure == 0), numeric otherwise.
    # This exactly mirrors the Telco dataset quirk handled by preprocess_raw().
    tenures = df["tenure"].tolist()
    monthly = df["MonthlyCharges"].tolist()
    total_charges = []
    for t, m in zip(tenures, monthly):
        if t == 0:
            total_charges.append(" ")  # simulates the raw dataset spaces
        else:
            total_charges.append(str(round(t * m, 2)))
    df["TotalCharges"] = total_charges

    df["Churn"] = rng.choice(["Yes", "No"], n, p=[0.27, 0.73]).tolist()

    csv_path = tmp_path / "tiny_churn.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
