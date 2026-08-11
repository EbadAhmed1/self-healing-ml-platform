"""
agent/evidence.py
─────────────────
Evidence Gathering Engine for Diagnosis Agent.

Given a triggered Alert, EvidenceGatherer queries four database sources to
construct a comprehensive evidence payload for root cause diagnosis:
  1. Deployments table   — identifies if a model deployment occurred shortly before the alert
  2. Simulation Events   — checks if an upstream data pipeline error (injection) overlaps the alert
  3. Drift Reports       — analyzes whether drift is isolated (single feature) or widespread
  4. Accuracy Reports    — determines if performance metrics (F1/AUC) have actually degraded

Full reasoning steps are logged to `reasoning_log` so human operators can audit the
entire evidence trail, not just the final conclusion.
"""

from __future__ import annotations

import logging
from datetime import timedelta, timezone

from sqlalchemy import select

from agent.db_models import Deployment
from monitoring.db_models import AccuracyReport, Alert, DriftReport
from simulator.db_models import SimulationEvent

log = logging.getLogger("agent.evidence")

# Configurable time windows for evidence correlation
DEPLOY_CORRELATION_WINDOW_MINUTES = 30
INJECTION_CORRELATION_WINDOW_MINUTES = 30
LOOKBACK_WINDOW_HOURS = 24


class EvidenceGatherer:
    """
    Gathers structured evidence for a given alert across all platform database tables.
    """

    def __init__(self, db_session) -> None:
        self.db = db_session

    def gather(self, alert: Alert) -> dict:
        """
        Gather evidence payload and construct step-by-step reasoning log for `alert`.

        Returns:
            dict containing:
              - alert_details: metadata from the triggering alert
              - recent_deployments: list of deployment dicts within window
              - active_injections: list of simulation event dicts overlapping alert
              - drift_context: summary of recent feature and prediction drift reports
              - accuracy_context: latest accuracy report and metrics
              - reasoning_log: step-by-step audit log of findings
        """
        reasoning: list[str] = []
        triggered_at = alert.triggered_at

        # Ensure timezone-aware timestamp
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=timezone.utc)

        reasoning.append(
            f"Step 1: Initiating evidence gathering for Alert id={alert.id} "
            f"type={alert.alert_type!r} severity={alert.severity!r} "
            f"model={alert.model_id!r} triggered_at={triggered_at.isoformat()}"
        )

        # ── 1. Check Deployments ─────────────────────────────────────────────
        deploy_cutoff = triggered_at - timedelta(
            minutes=DEPLOY_CORRELATION_WINDOW_MINUTES
        )
        stmt_deploy = (
            select(Deployment)
            .where(Deployment.model_id == alert.model_id)
            .where(Deployment.deployed_at >= deploy_cutoff)
            .where(Deployment.deployed_at <= triggered_at + timedelta(minutes=5))
            .order_by(Deployment.deployed_at.desc())
        )
        recent_deploys = self.db.scalars(stmt_deploy).all()

        deploy_data = [
            {
                "id": d.id,
                "version": d.version,
                "deployed_at": d.deployed_at.isoformat(),
                "git_commit": d.git_commit,
                "is_current": d.is_current,
            }
            for d in recent_deploys
        ]

        if deploy_data:
            reasoning.append(
                f"Step 2: Found {len(deploy_data)} deployment(s) within "
                f"{DEPLOY_CORRELATION_WINDOW_MINUTES}m window: {deploy_data[0]['version']} "
                f"at {deploy_data[0]['deployed_at']}"
            )
        else:
            reasoning.append(
                f"Step 2: No recent deployments found for {alert.model_id} in "
                f"the {DEPLOY_CORRELATION_WINDOW_MINUTES}m window prior to alert."
            )

        # ── 2. Check Simulation / Upstream Data Events ───────────────────────
        inj_cutoff = triggered_at - timedelta(
            minutes=INJECTION_CORRELATION_WINDOW_MINUTES
        )
        stmt_inj = (
            select(SimulationEvent)
            .where(SimulationEvent.started_at <= triggered_at + timedelta(minutes=5))
            .where(
                (SimulationEvent.ended_at.is_(None))
                | (SimulationEvent.ended_at >= inj_cutoff)
            )
            .order_by(SimulationEvent.started_at.desc())
        )
        active_injections = self.db.scalars(stmt_inj).all()

        inj_data = [
            {
                "id": e.id,
                "event_type": e.event_type,
                "started_at": e.started_at.isoformat(),
                "ended_at": e.ended_at.isoformat() if e.ended_at else None,
                "parameters": e.parameters,
            }
            for e in active_injections
        ]

        if inj_data:
            reasoning.append(
                f"Step 3: Found {len(inj_data)} active/overlapping upstream injection event(s): "
                f"{[i['event_type'] for i in inj_data]}"
            )
        else:
            reasoning.append("Step 3: No overlapping upstream injection events found.")

        # ── 3. Check Drift Context ───────────────────────────────────────────
        drift_cutoff = triggered_at - timedelta(hours=LOOKBACK_WINDOW_HOURS)
        stmt_drift = (
            select(DriftReport)
            .where(DriftReport.model_id == alert.model_id)
            .where(DriftReport.checked_at >= drift_cutoff)
            .order_by(DriftReport.checked_at.desc())
        )
        recent_drift_reports = self.db.scalars(stmt_drift).all()

        drifting_features = [
            r.feature_name
            for r in recent_drift_reports
            if r.status in ("moderate", "significant")
            and r.feature_name != "_all_features"
        ]
        unique_drifting_features = sorted(list(set(drifting_features)))

        drift_context = {
            "total_reports_evaluated": len(recent_drift_reports),
            "drifting_features": unique_drifting_features,
            "drifting_count": len(unique_drifting_features),
            "prediction_drift": any(
                r.feature_name == "_prediction"
                and r.status in ("moderate", "significant")
                for r in recent_drift_reports
            ),
        }

        reasoning.append(
            f"Step 4: Drift context evaluated — {len(unique_drifting_features)} feature(s) "
            f"drifting: {unique_drifting_features}. Prediction drift present: {drift_context['prediction_drift']}"
        )

        # ── 4. Check Accuracy Context ────────────────────────────────────────
        stmt_acc = (
            select(AccuracyReport)
            .where(AccuracyReport.model_id == alert.model_id)
            .where(AccuracyReport.created_at >= drift_cutoff)
            .order_by(AccuracyReport.created_at.desc())
        )
        latest_acc = self.db.scalars(stmt_acc).first()

        accuracy_context: dict = {}
        if latest_acc:
            accuracy_context = {
                "f1": latest_acc.f1,
                "precision": latest_acc.precision,
                "recall": latest_acc.recall,
                "roc_auc": latest_acc.roc_auc,
                "n_samples": latest_acc.n_samples,
                "window_end": latest_acc.window_end.isoformat(),
            }
            reasoning.append(
                f"Step 5: Latest accuracy report found — F1={latest_acc.f1} "
                f"Precision={latest_acc.precision} Recall={latest_acc.recall} "
                f"(n={latest_acc.n_samples})"
            )
        else:
            reasoning.append(
                "Step 5: No recent accuracy report available (insufficient joined outcome data)."
            )

        evidence_payload = {
            "alert": {
                "id": alert.id,
                "model_id": alert.model_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "triggered_at": triggered_at.isoformat(),
                "details": alert.details,
            },
            "recent_deployments": deploy_data,
            "active_injections": inj_data,
            "drift_context": drift_context,
            "accuracy_context": accuracy_context,
            "reasoning_log": reasoning,
        }

        return evidence_payload
