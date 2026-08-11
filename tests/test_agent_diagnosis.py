"""
tests/test_agent_diagnosis.py
──────────────────────────────
Deterministic scenario tests for the Diagnosis Agent.

REQUIREMENT FROM SPEC:
  "Write pytest tests using CONSTRUCTED scenarios (fake alerts + fake deployments
   + fake simulation events inserted directly into a test DB) to verify each rule
   fires correctly in isolation. This is important — don't just test against real
   simulator output, test each rule deterministically."

Scenarios tested:
  1. Bad Deploy Rule -> Rollback triggered, confidence 0.90, status 'auto_resolved', pointer updated
  2. Impossible Rollback Safety Guard -> Single v1 deploy forces escalation with explanation
  3. Upstream Data Issue Rule -> Injection event triggers 'fix_upstream_data', confidence 0.85
  4. Isolated Single-Feature Drift Rule -> Single feature drift recommends 'monitor', confidence 0.65
  5. Broad Concept Drift Rule -> Multi-feature drift + accuracy drop recommends 'retrain', confidence 0.70
  6. Unclear Root Cause Fallback -> Low confidence 0.20, status 'escalated'
  7. Alert Deduplication -> Second alert within 15m window deduplicated into primary incident
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from agent.db_models import Deployment
from agent.run_agent import process_unprocessed_alerts
from monitoring.db_models import AccuracyReport, Alert, DriftReport
from simulator.db_models import SimulationEvent


# ===========================================================================
# Fixtures & Setup
# ===========================================================================
@pytest.fixture(autouse=True)
def clean_db(test_engine):
    """Ensure a clean database state before each test in this module."""
    from app.database import Base

    with test_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture
def mock_artifacts(tmp_path):
    """Mock models/artifacts directory structure."""
    artifacts_dir = tmp_path / "models" / "artifacts" / "churn-model"
    v1_dir = artifacts_dir / "v1"
    v2_dir = artifacts_dir / "v2"
    v1_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)

    current_json = artifacts_dir / "current.json"
    with open(current_json, "w") as f:
        json.dump({"version": "v2", "artifact_dir": str(v2_dir)}, f)

    return tmp_path


# ===========================================================================
# 1. Rule Priority 1: Bad Deploy & Rollback Execution
# ===========================================================================
class TestBadDeployAndRollback:
    def test_bad_deploy_rule_fires_rollback_with_high_confidence(
        self, test_engine, mock_artifacts, monkeypatch
    ):
        """
        Scenario: Model v1 deployed 2 hours ago. Model v2 deployed 10 min ago.
        Accuracy drop alert triggers.
        Expected: Bad Deploy rule matches (conf=0.90), status='auto_resolved',
                  current.json pointer updated to v1.
        """
        monkeypatch.setattr("agent.remediation.PROJECT_ROOT", mock_artifacts)

        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        # 1. Insert v1 deployment (older)
        d1 = Deployment(
            id="deploy-v1",
            model_id="churn-model:v1",
            version="v1",
            deployed_at=now - timedelta(hours=2),
            git_commit="git-v1",
            eval_metrics={"f1": 0.85},
            is_current=False,
        )
        # 2. Insert v2 deployment (recent)
        d2 = Deployment(
            id="deploy-v2",
            model_id="churn-model:v2",
            version="v2",
            deployed_at=now - timedelta(minutes=10),
            git_commit="git-v2",
            eval_metrics={"f1": 0.86},
            is_current=True,
        )
        # 3. Insert accuracy drop alert
        alert = Alert(
            id="alert-bad-deploy",
            model_id="churn-model:v2",
            alert_type="accuracy_drop",
            severity="critical",
            triggered_at=now,
            details={"current_f1": 0.50, "baseline_f1": 0.85},
            processed=False,
        )
        db.add_all([d1, d2, alert])
        db.commit()
        db.close()

        # Run agent pass
        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)

        assert len(incidents) == 1
        inc = incidents[0]
        assert "Bad deployment" in inc.hypothesis
        assert inc.confidence == 0.90
        assert inc.recommended_action == "rollback"
        assert inc.status == "auto_resolved"
        assert inc.resolved_at is not None

        # Verify pointer updated back to v1
        pointer_file = (
            mock_artifacts / "models" / "artifacts" / "churn-model" / "current.json"
        )
        with open(pointer_file) as f:
            pointer = json.load(f)
        assert pointer["version"] == "v1"

    def test_rollback_impossible_escalates(
        self, test_engine, mock_artifacts, monkeypatch
    ):
        """
        Scenario: Model v1 deployed 10 min ago (only version in history).
        Accuracy drop alert triggers.
        Expected: Safety guard detects no previous version -> forces escalation,
                  confidence=0.20, status='escalated'.
        """
        monkeypatch.setattr("agent.remediation.PROJECT_ROOT", mock_artifacts)

        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        # Insert only v1 deployment
        d1 = Deployment(
            id="deploy-only-v1",
            model_id="churn-model:v1",
            version="v1",
            deployed_at=now - timedelta(minutes=10),
            git_commit="git-v1",
            eval_metrics={"f1": 0.85},
            is_current=True,
        )
        alert = Alert(
            id="alert-single-v1",
            model_id="churn-model:v1",
            alert_type="accuracy_drop",
            severity="critical",
            triggered_at=now,
            details={"current_f1": 0.50},
            processed=False,
        )
        db.add_all([d1, alert])
        db.commit()
        db.close()

        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)

        assert len(incidents) == 1
        inc = incidents[0]
        assert "Rollback impossible" in inc.hypothesis
        assert inc.status == "escalated"
        assert inc.recommended_action == "escalate"
        assert inc.confidence == 0.20


# ===========================================================================
# 2. Rule Priority 2: Upstream Data Issue
# ===========================================================================
class TestUpstreamDataIssue:
    def test_upstream_data_issue_rule_triggers(self, test_engine):
        """
        Scenario: Active simulation event on 'tenure' overlapping alert time.
        Drift alert triggers.
        Expected: Upstream Data Issue rule matches (conf=0.85), action='fix_upstream_data'.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        # Active simulation injection
        inj = SimulationEvent(
            id="inj-1",
            event_type="drift_feature",
            started_at=now - timedelta(minutes=15),
            ended_at=None,
            parameters={"feature": "tenure", "magnitude": 20.0},
        )
        alert = Alert(
            id="alert-upstream",
            model_id="churn-model:v1",
            alert_type="feature_drift",
            severity="warning",
            triggered_at=now,
            details={"feature": "tenure", "psi": 0.35},
            processed=False,
        )
        db.add_all([inj, alert])
        db.commit()
        db.close()

        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)

        assert len(incidents) == 1
        inc = incidents[0]
        assert "Upstream data" in inc.hypothesis
        assert inc.confidence == 0.85
        assert inc.recommended_action == "fix_upstream_data"
        assert inc.status == "escalated"  # not rollback, so escalated for human fix


# ===========================================================================
# 3. Rule Priority 3 & 4: Feature Drift & Concept Drift
# ===========================================================================
class TestFeatureAndConceptDrift:
    def test_single_feature_drift_recommends_monitor(self, test_engine):
        """
        Scenario: 1 feature drifting ('tenure'), no accuracy drop.
        Expected: Single-Feature Drift rule matches (conf=0.65), action='monitor'.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        drift = DriftReport(
            model_id="churn-model:v1",
            feature_name="tenure",
            psi_score=0.30,
            method="PSI",
            sample_size=100,
            status="significant",
            checked_at=now - timedelta(minutes=5),
        )
        alert = Alert(
            id="alert-single-drift",
            model_id="churn-model:v1",
            alert_type="feature_drift",
            severity="warning",
            triggered_at=now,
            details={"feature": "tenure", "psi": 0.30},
            processed=False,
        )
        db.add_all([drift, alert])
        db.commit()
        db.close()

        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)

        assert len(incidents) == 1
        inc = incidents[0]
        assert "Isolated single-feature input drift" in inc.hypothesis
        assert inc.confidence == 0.65
        assert inc.recommended_action == "monitor"

    def test_broad_concept_drift_recommends_retrain(self, test_engine):
        """
        Scenario: 2 features drifting ('tenure', 'Contract') + accuracy drop.
        Expected: Broad Concept Drift rule matches (conf=0.70), action='retrain'.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        d1 = DriftReport(
            model_id="churn-model:v1",
            feature_name="tenure",
            psi_score=0.35,
            status="significant",
            sample_size=100,
            checked_at=now - timedelta(minutes=5),
        )
        d2 = DriftReport(
            model_id="churn-model:v1",
            feature_name="Contract",
            psi_score=0.40,
            status="significant",
            sample_size=100,
            checked_at=now - timedelta(minutes=5),
        )
        acc = AccuracyReport(
            model_id="churn-model:v1",
            window_start=now - timedelta(hours=24),
            window_end=now,
            f1=0.60,
            n_samples=50,
            created_at=now - timedelta(minutes=5),
        )
        alert = Alert(
            id="alert-broad-drift",
            model_id="churn-model:v1",
            alert_type="accuracy_drop",
            severity="critical",
            triggered_at=now,
            details={"f1": 0.60},
            processed=False,
        )
        db.add_all([d1, d2, acc, alert])
        db.commit()
        db.close()

        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)

        assert len(incidents) == 1
        inc = incidents[0]
        assert "Broad concept drift" in inc.hypothesis
        assert inc.confidence == 0.70
        assert inc.recommended_action == "retrain"


# ===========================================================================
# 4. Fallback & Deduplication
# ===========================================================================
class TestFallbackAndDeduplication:
    def test_unclear_cause_escalates(self, test_engine):
        """
        Scenario: Generic alert with no deploy, injection, or drift context.
        Expected: Unclear Cause fallback rule matches (conf=0.20), action='escalate'.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        alert = Alert(
            id="alert-unclear",
            model_id="churn-model:v1",
            alert_type="unknown_anomaly",
            severity="warning",
            triggered_at=now,
            details={},
            processed=False,
        )
        db.add(alert)
        db.commit()
        db.close()

        incidents = process_unprocessed_alerts(db_session_factory=SessionLocal)

        assert len(incidents) == 1
        inc = incidents[0]
        assert "Unclear root cause" in inc.hypothesis
        assert inc.confidence == 0.20
        assert inc.recommended_action == "escalate"

    def test_alert_deduplication_groups_within_window(self, test_engine):
        """
        Scenario: Alert 1 arrives -> creates Incident 1.
                  Alert 2 arrives 3 min later -> deduplicated into Incident 1.
        """
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()

        now = datetime.now(timezone.utc)

        alert1 = Alert(
            id="alert-dedup-1",
            model_id="churn-model:v1",
            alert_type="feature_drift",
            severity="warning",
            triggered_at=now - timedelta(minutes=5),
            details={"feature": "tenure"},
            processed=False,
        )
        db.add(alert1)
        db.commit()
        db.close()

        # Process alert 1 -> creates incident
        incidents1 = process_unprocessed_alerts(db_session_factory=SessionLocal)
        assert len(incidents1) == 1
        first_inc_id = incidents1[0].id

        # Insert alert 2 (3 minutes later)
        db = SessionLocal()
        alert2 = Alert(
            id="alert-dedup-2",
            model_id="churn-model:v1",
            alert_type="feature_drift",
            severity="warning",
            triggered_at=now - timedelta(minutes=2),
            details={"feature": "MonthlyCharges"},
            processed=False,
        )
        db.add(alert2)
        db.commit()
        db.close()

        # Process alert 2 -> deduplicated into incident 1
        incidents2 = process_unprocessed_alerts(db_session_factory=SessionLocal)
        assert len(incidents2) == 1
        assert incidents2[0].id == first_inc_id

        # Verify alert 2 was marked processed
        db = SessionLocal()
        a2 = db.get(Alert, "alert-dedup-2")
        assert a2.processed is True
        db.close()
