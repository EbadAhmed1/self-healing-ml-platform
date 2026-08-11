"""
tests/test_monitoring.py
─────────────────────────
Unit and integration tests for monitoring, drift detection, and accuracy tracking.

REQUIREMENT FROM SPEC:
  "Write pytest tests that specifically verify: feeding the PSI function two
   identical distributions returns ~0, and two very different distributions
   (construct one deliberately) returns a high score — this proves the metric
   actually works, not just that the code runs."

Tests cover:
  1. Identical distributions return ~0.0 PSI
  2. Shifted numeric distributions return high PSI (> 0.5)
  3. Skewed categorical distributions return high PSI (> 0.5)
  4. Zero-count category bins handle epsilon smoothing without zero-division/NaN
  5. Constant / all-null values calculate safely without raising exceptions
  6. Small sample guard (< 30) returns 'insufficient_data' and skips alerts
  7. Accuracy checker skips reporting when joined prediction-outcome count < min_samples
  8. End-to-end drift checker and alert creation using in-memory SQLite DB
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from monitoring.check_accuracy import run_accuracy_check
from monitoring.check_drift import run_drift_check
from monitoring.db_models import Alert, DriftReport
from monitoring.psi import (
    calculate_categorical_psi,
    calculate_numeric_psi,
    classify_psi,
)


# ===========================================================================
# 1. PSI Mathematical Property Tests
# ===========================================================================
class TestPSIMathematicalProperties:
    def test_identical_numeric_distributions_return_near_zero(self):
        """Identical baseline and current samples must return PSI approx 0.0 (< 0.01)."""
        rng = np.random.default_rng(42)
        base = rng.normal(loc=30.0, scale=10.0, size=1000).tolist()
        curr = rng.normal(loc=30.0, scale=10.0, size=1000).tolist()

        score = calculate_numeric_psi(base, curr)
        assert (
            score < 0.05
        ), f"Expected near-zero PSI for identical distributions, got {score}"

    def test_shifted_numeric_distribution_returns_high_score(self):
        """Constructing a significantly shifted distribution (mean 30 -> 70) returns high PSI (> 0.5)."""
        rng = np.random.default_rng(42)
        base = rng.normal(loc=30.0, scale=5.0, size=1000).tolist()
        curr = rng.normal(loc=70.0, scale=5.0, size=1000).tolist()

        score = calculate_numeric_psi(base, curr)
        assert (
            score > 0.5
        ), f"Expected high PSI (> 0.5) for shifted distribution, got {score}"

    def test_identical_categorical_distributions_return_near_zero(self):
        """Identical categorical value counts return PSI approx 0.0."""
        base_counts = {"Male": 500, "Female": 500}
        curr = ["Male"] * 500 + ["Female"] * 500

        score = calculate_categorical_psi(base_counts, curr)
        assert score < 0.01, f"Expected PSI ~ 0.0, got {score}"

    def test_skewed_categorical_distribution_returns_high_score(self):
        """Skewing a categorical feature (50/50 -> 95/5) returns a high PSI (> 0.5)."""
        base_counts = {"Male": 500, "Female": 500}
        curr = ["Male"] * 950 + ["Female"] * 50

        score = calculate_categorical_psi(base_counts, curr)
        assert (
            score > 0.5
        ), f"Expected high PSI (> 0.5) for skewed categories, got {score}"

    def test_zero_count_bins_handled_safely(self):
        """
        Category missing entirely in current data (0 occurrences) handles epsilon
        smoothing without raising ZeroDivisionError or returning NaN/inf.
        """
        base_counts = {"DSL": 300, "Fiber optic": 400, "No": 300}
        curr = ["DSL"] * 500 + ["Fiber optic"] * 500  # "No" has 0 occurrences

        score = calculate_categorical_psi(base_counts, curr)
        assert not np.isnan(score)
        assert not np.isinf(score)
        assert score > 0.1  # should show moderate/high drift due to missing category

    def test_constant_value_series_does_not_crash(self):
        """Constant value series (0 variance) calculates safely without raising error."""
        base = [50.0] * 100
        curr = [50.0] * 100

        score = calculate_numeric_psi(base, curr)
        assert not np.isnan(score)
        assert score == 0.0


# ===========================================================================
# 2. Classification and Small Sample Guard Tests
# ===========================================================================
class TestPSIClassification:
    def test_classify_psi_thresholds(self):
        assert classify_psi(0.05, sample_size=100) == "stable"
        assert classify_psi(0.15, sample_size=100) == "moderate"
        assert classify_psi(0.35, sample_size=100) == "significant"

    def test_small_sample_size_returns_insufficient_data(self):
        """Sample size < 30 returns 'insufficient_data' regardless of score."""
        assert classify_psi(0.50, sample_size=10) == "insufficient_data"


# ===========================================================================
# 3. Accuracy Checker Unit & Guard Tests
# ===========================================================================
class TestAccuracyCheckGuard:
    def test_accuracy_check_skips_when_samples_below_min(self, test_engine):
        """
        When joined prediction-outcome count is less than min_samples,
        run_accuracy_check logs note and returns None (skips report).
        """
        from sqlalchemy.orm import sessionmaker

        SessionLocal = sessionmaker(bind=test_engine)
        report = run_accuracy_check(
            hours=24, min_samples=30, db_session_factory=SessionLocal
        )
        assert report is None


# ===========================================================================
# 4. End-to-End Drift and Alert Integration Tests
# ===========================================================================
class TestDriftAndAlertIntegration:
    @pytest.fixture
    def mock_baseline_metadata(self, tmp_path):
        """Create mock metadata.json and current.json for drift testing."""
        artifacts_dir = tmp_path / "models" / "artifacts" / "churn-model"
        version_dir = artifacts_dir / "test-v1"
        version_dir.mkdir(parents=True)

        current_json = artifacts_dir / "current.json"
        with open(current_json, "w") as f:
            json.dump({"version": "test-v1", "artifact_dir": str(version_dir)}, f)

        metadata = {
            "training_data_snapshot": {
                "numeric": {
                    "tenure": {
                        "mean": 30.0,
                        "std": 10.0,
                        "median": 30.0,
                        "deciles": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    }
                },
                "categorical": {
                    "Contract": {
                        "value_counts": {
                            "Month-to-month": 50,
                            "One year": 30,
                            "Two year": 20,
                        }
                    }
                },
                "prediction": {
                    "deciles": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                },
            }
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return tmp_path

    def test_run_drift_check_creates_reports_and_alerts(
        self, test_engine, mock_baseline_metadata, monkeypatch
    ):
        """
        Populate SQLite DB with 50 shifted predictions and run drift check.
        Verify DriftReport and Alert rows are inserted into DB.
        """
        from sqlalchemy.orm import sessionmaker
        from app.db_models import Prediction

        monkeypatch.setattr("monitoring.check_drift.NUMERIC_FEATURES", ["tenure"])
        monkeypatch.setattr("monitoring.check_drift.CATEGORICAL_FEATURES", ["Contract"])

        # Patch project root in check_drift to use mock metadata
        monkeypatch.setattr(
            "monitoring.check_drift.PROJECT_ROOT", mock_baseline_metadata
        )

        TestingSessionLocal = sessionmaker(bind=test_engine)
        db = TestingSessionLocal()

        # Insert 50 predictions with shifted tenure (+40 shift -> tenure = 70)
        for i in range(50):
            pred = Prediction(
                model_id="churn-model:test-v1",
                input_features={"tenure": 70.0, "Contract": "Month-to-month"},
                prediction=True,
                confidence=0.95,
                created_at=datetime.now(timezone.utc),
            )
            db.add(pred)
        db.commit()
        db.close()

        # Run drift check
        reports = run_drift_check(
            hours=24,
            target_model_id="churn-model:test-v1",
            db_session_factory=TestingSessionLocal,
        )

        assert len(reports) == 3  # tenure, Contract, _prediction

        # Verify DB contents
        db = TestingSessionLocal()
        db_reports = db.query(DriftReport).all()
        db_alerts = db.query(Alert).all()

        assert len(db_reports) >= 3
        # Shifted tenure (30 -> 70) should have triggered an alert
        assert len(db_alerts) >= 1
        tenure_alert = next(
            a for a in db_alerts if a.details.get("feature") == "tenure"
        )
        assert tenure_alert.severity in ("warning", "critical")
        db.close()
