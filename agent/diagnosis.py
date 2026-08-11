"""
agent/diagnosis.py
──────────────────
Priority Rule-Based Hypothesis Engine for Diagnosis Agent.

PHILOSOPHY:
  Rule-based diagnosis operates 100% deterministically without LLM non-determinism.
  Rules are evaluated in strict priority order (Priority 1 -> Priority 5).
  Each diagnosis returns an explicit numeric confidence score (0.0 to 1.0) and
  a structured recommended action.

RULE PRIORITY LIST:
  1. BAD DEPLOY (Priority 1, Conf=0.90, Action='rollback'):
     Recent deployment <= 30m before alert AND (alert is accuracy_drop OR prediction drift present).

  2. UPSTREAM DATA ISSUE (Priority 2, Conf=0.85, Action='fix_upstream_data'):
     Overlapping upstream simulation/injection event present.

  3. ISOLATED SINGLE-FEATURE DRIFT (Priority 3, Conf=0.65, Action='monitor'):
     Exactly 1 input feature drifting AND no accuracy drop.

  4. BROAD CONCEPT DRIFT (Priority 4, Conf=0.70, Action='retrain'):
     >= 2 input features drifting AND accuracy drop detected.

  5. UNCLEAR CAUSE (Priority 5, Conf=0.20, Action='escalate'):
     Fallback rule when no specific pattern matches. Always escalates to human operator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("agent.diagnosis")


@dataclass
class DiagnosisResult:
    """Structured hypothesis output from the Rule Engine."""

    hypothesis: str
    confidence: float  # explicit numeric score 0.0 - 1.0
    recommended_action: str  # rollback, fix_upstream_data, monitor, retrain, escalate
    rule_name: str


class DiagnosisEngine:
    """
    Evaluates evidence against priority-ranked rules to determine root cause hypothesis.
    """

    def diagnose(self, evidence: dict) -> DiagnosisResult:
        """
        Evaluate evidence payload and return highest-priority matching diagnosis.
        Appends reasoning step to evidence["reasoning_log"].
        """
        reasoning: list[str] = evidence.get("reasoning_log", [])
        alert_info = evidence.get("alert", {})
        alert_type = alert_info.get("alert_type", "")
        deploys = evidence.get("recent_deployments", [])
        injections = evidence.get("active_injections", [])
        drift_ctx = evidence.get("drift_context", {})
        acc_ctx = evidence.get("accuracy_context", {})

        drifting_count = drift_ctx.get("drifting_count", 0)
        drifting_features = drift_ctx.get("drifting_features", [])
        prediction_drift = drift_ctx.get("prediction_drift", False)
        accuracy_drop = alert_type == "accuracy_drop"

        # ── PRIORITY 1: BAD DEPLOYMENT ───────────────────────────────────────
        if deploys and (accuracy_drop or prediction_drift):
            latest_deploy_ver = deploys[0].get("version", "unknown")
            hypothesis = (
                f"Bad deployment: Model version '{latest_deploy_ver}' caused performance "
                f"degradation/prediction drift shortly after deployment."
            )
            reasoning.append(
                f"Step 6 [RULE MATCH: Priority 1 - Bad Deploy]: Recent deployment "
                f"'{latest_deploy_ver}' matches alert window with performance drop. "
                f"Confidence=0.90, Action='rollback'."
            )
            return DiagnosisResult(
                hypothesis=hypothesis,
                confidence=0.90,
                recommended_action="rollback",
                rule_name="bad_deploy",
            )

        # ── PRIORITY 2: UPSTREAM DATA ISSUE ─────────────────────────────────
        if injections:
            primary_inj = injections[0]
            event_type = primary_inj.get("event_type", "unknown")
            params = primary_inj.get("parameters", {})
            feature = params.get("feature", "unknown")
            hypothesis = (
                f"Upstream data pipeline failure / data corruption: Active injection "
                f"'{event_type}' detected on feature '{feature}'."
            )
            reasoning.append(
                f"Step 6 [RULE MATCH: Priority 2 - Upstream Data Issue]: Active injection "
                f"'{event_type}' overlaps alert window. Confidence=0.85, Action='fix_upstream_data'."
            )
            return DiagnosisResult(
                hypothesis=hypothesis,
                confidence=0.85,
                recommended_action="fix_upstream_data",
                rule_name="upstream_data_issue",
            )

        # ── PRIORITY 3: ISOLATED SINGLE-FEATURE DRIFT ───────────────────────
        if drifting_count == 1 and not accuracy_drop:
            single_feat = drifting_features[0]
            hypothesis = (
                f"Isolated single-feature input drift on '{single_feat}': "
                f"Distribution shifted but no model accuracy impact confirmed yet."
            )
            reasoning.append(
                f"Step 6 [RULE MATCH: Priority 3 - Single Feature Drift]: Only feature "
                f"'{single_feat}' drifting with stable accuracy. Confidence=0.65, Action='monitor'."
            )
            return DiagnosisResult(
                hypothesis=hypothesis,
                confidence=0.65,
                recommended_action="monitor",
                rule_name="single_feature_drift",
            )

        # ── PRIORITY 4: BROAD CONCEPT DRIFT ──────────────────────────────────
        if drifting_count >= 2 and (accuracy_drop or acc_ctx):
            hypothesis = (
                f"Broad concept drift: Widespread feature drift across {drifting_count} "
                f"features ({', '.join(drifting_features)}) with confirmed accuracy impact."
            )
            reasoning.append(
                f"Step 6 [RULE MATCH: Priority 4 - Broad Concept Drift]: {drifting_count} "
                f"features drifting with accuracy impact. Confidence=0.70, Action='retrain'."
            )
            return DiagnosisResult(
                hypothesis=hypothesis,
                confidence=0.70,
                recommended_action="retrain",
                rule_name="broad_concept_drift",
            )

        # ── PRIORITY 5: UNCLEAR ROOT CAUSE (FALLBACK) ────────────────────────
        hypothesis = (
            "Unclear root cause: Alert triggered without an obvious recent deployment, "
            "upstream injection, or isolated feature drift pattern."
        )
        reasoning.append(
            "Step 6 [RULE MATCH: Priority 5 - Unclear Cause]: No priority rules matched. "
            "Confidence=0.20, Action='escalate'."
        )
        return DiagnosisResult(
            hypothesis=hypothesis,
            confidence=0.20,
            recommended_action="escalate",
            rule_name="unclear_cause",
        )
