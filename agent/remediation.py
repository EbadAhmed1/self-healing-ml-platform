"""
agent/remediation.py
────────────────────
Remediation Engine & Rollback Execution Handler.

CONFIDENCE-GATED DECISION RULES:
  1. Automated execution occurs ONLY IF:
       confidence >= 0.80 AND recommended_action == 'rollback'
  2. For all other actions ('monitor', 'retrain', 'fix_upstream_data', 'escalate')
     OR confidence < 0.80:
       The incident is marked 'escalated' for human operator review (no automated action).

IMPOSSIBLE ROLLBACK SAFETY GUARD:
  If a rollback is recommended but NO previous version exists in the deployment history
  (e.g., this is the first deployed version v1):
    - Automated rollback CANNOT be performed.
    - The engine overrides the status to 'escalated', action to 'escalate', confidence to 0.20.
    - Appends a clear safety log: "Rollback impossible: No prior deployment version available".

POINTER ROLLBACK EXECUTION:
  - Updates `models/artifacts/{model_name}/current.json` back to the previous version's artifact path.
  - Updates `deployments` table: sets previous deployment is_current = True, current is_current = False.
  - Sets incident status = 'auto_resolved' and records resolved_at timestamp.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from agent.db_models import Deployment, Incident
from agent.diagnosis import DiagnosisResult
from models.feature_config import CURRENT_VERSION_FILENAME

log = logging.getLogger("agent.remediation")

CONFIDENCE_THRESHOLD: float = 0.80
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RemediationEngine:
    """
    Evaluates confidence gates and executes automated rollbacks when safe.
    """

    def __init__(self, db_session) -> None:
        self.db = db_session

    def process_incident(
        self,
        alert_id: str,
        model_id: str,
        diagnosis: DiagnosisResult,
        evidence: dict,
    ) -> Incident:
        """
        Create an Incident record, apply confidence-gated decision rules, and
        execute pointer rollback if auto-resolution conditions are met.

        Returns:
            Saved Incident ORM object.
        """
        reasoning: list[str] = evidence.get("reasoning_log", [])
        action = diagnosis.recommended_action
        confidence = diagnosis.confidence
        hypothesis = diagnosis.hypothesis

        reasoning.append(
            f"Step 7: Evaluating remediation gates — Action={action!r}, Confidence={confidence:.2f}, "
            f"Threshold={CONFIDENCE_THRESHOLD:.2f}"
        )

        status = "open"
        resolved_at = None

        # ── CHECK ROLLBACK AUTOMATION GATES ──────────────────────────────────
        if action == "rollback" and confidence >= CONFIDENCE_THRESHOLD:
            # Check for previous deployment
            model_name = model_id.split(":")[0] if ":" in model_id else model_id
            stmt = (
                select(Deployment)
                .where(Deployment.model_id.like(f"{model_name}:%"))
                .order_by(Deployment.deployed_at.desc())
            )
            all_deploys = self.db.scalars(stmt).all()

            current_deploy = next((d for d in all_deploys if d.is_current), None)
            if current_deploy is None and all_deploys:
                current_deploy = all_deploys[0]

            previous_deploys = [
                d
                for d in all_deploys
                if current_deploy is None or d.id != current_deploy.id
            ]

            if not previous_deploys:
                # ── IMPOSSIBLE ROLLBACK SAFETY GUARD ────────────────────────
                reasoning.append(
                    "Step 8 [SAFETY GUARD TRIGGERED]: Rollback requested but NO previous version "
                    "exists in deployment history (first deployed version). Forcing escalation."
                )
                action = "escalate"
                status = "escalated"
                confidence = 0.20
                hypothesis += (
                    " (Rollback impossible: No prior deployment version available)"
                )
            else:
                target_deploy = previous_deploys[0]
                reasoning.append(
                    f"Step 8: Found previous deployment candidate '{target_deploy.version}' "
                    f"deployed at {target_deploy.deployed_at.isoformat()}"
                )

                # Execute physical pointer rollback
                success = self._execute_pointer_rollback(
                    model_name=model_name,
                    target_version=target_deploy.version,
                    current_deploy=current_deploy,
                    target_deploy=target_deploy,
                    reasoning=reasoning,
                )

                if success:
                    status = "auto_resolved"
                    resolved_at = datetime.now(timezone.utc)
                    reasoning.append(
                        f"Step 9 [AUTO-RESOLVED]: Automated rollback to '{target_deploy.version}' "
                        f"executed successfully."
                    )
                else:
                    status = "escalated"
                    action = "escalate"
                    reasoning.append(
                        "Step 9 [ROLLBACK FAILED]: Physical pointer update failed. Escalating incident."
                    )
        else:
            status = "escalated"
            reasoning.append(
                f"Step 8: Action '{action}' does not meet auto-rollback criteria "
                f"(requires action='rollback' AND confidence >= {CONFIDENCE_THRESHOLD:.2f}). "
                f"Marking incident as 'escalated'."
            )

        # Save Incident to DB
        incident = Incident(
            alert_id=alert_id,
            model_id=model_id,
            hypothesis=hypothesis,
            confidence=confidence,
            evidence=evidence,
            recommended_action=action,
            status=status,
            created_at=datetime.now(timezone.utc),
            resolved_at=resolved_at,
        )
        self.db.add(incident)
        self.db.commit()

        log.info(
            "Incident %s created for alert %s: status=%s action=%s conf=%.2f",
            incident.id,
            alert_id,
            status,
            action,
            confidence,
        )
        return incident

    def _execute_pointer_rollback(
        self,
        model_name: str,
        target_version: str,
        current_deploy: Deployment | None,
        target_deploy: Deployment,
        reasoning: list[str],
    ) -> bool:
        """
        Physical rollback: update current.json pointer file and deployments table flags.
        """
        try:
            artifacts_dir = PROJECT_ROOT / "models" / "artifacts" / model_name
            pointer_path = artifacts_dir / CURRENT_VERSION_FILENAME
            target_artifact_dir = artifacts_dir / target_version

            pointer_data = {
                "version": target_version,
                "artifact_dir": str(target_artifact_dir),
            }

            pointer_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pointer_path, "w") as f:
                json.dump(pointer_data, f, indent=2)

            log.info("Updated version pointer %s → %s", pointer_path, target_version)

            # Update DB flags
            if current_deploy:
                current_deploy.is_current = False
            target_deploy.is_current = True
            self.db.commit()

            reasoning.append(
                f"Pointer file {pointer_path.name} updated → version '{target_version}'. "
                f"Database deployment flags updated."
            )
            return True
        except Exception as exc:
            log.exception(
                "Failed to execute pointer rollback to %s: %s", target_version, exc
            )
            self.db.rollback()
            return False
