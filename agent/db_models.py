"""
agent/db_models.py
──────────────────
SQLAlchemy ORM models owned by the diagnosis agent.

Tables:
  deployments — tracks model deployment history, versions, and current deployment status
  incidents   — stores agent diagnosis results, evidence trails, confidence scores, and remediation status
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


# ---------------------------------------------------------------------------
# Deployment table
# ---------------------------------------------------------------------------
class Deployment(Base):
    """
    Audit history of deployed model versions.

    Used by the diagnosis agent to determine if a recent deployment occurred
    shortly before an alert, and to identify previous known-good versions
    for automated rollback.
    """

    __tablename__ = "deployments"
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Format: {model_name}:{version}, e.g. churn-model:v1",
    )
    version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Version tag, e.g. v1, v2",
    )
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    git_commit: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="unknown",
    )
    eval_metrics: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Evaluation metrics at time of deployment (precision, recall, f1, roc_auc)",
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if this version is currently active/serving",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="deployed",
        index=True,
        comment="One of: deployed, canary, rejected, canary_failed, rolled_back",
    )
    training_data_summary: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Metadata about training dataset (n_samples, source, time_range)",
    )
    traffic_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=100.0,
        comment="Traffic allocation percentage (0.0 to 100.0)",
    )
    rejection_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Reason if model was rejected by metric gate",
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when canary was promoted to 100%",
    )

    def __repr__(self) -> str:
        return (
            f"<Deployment model={self.model_id!r} version={self.version!r} "
            f"status={self.status!r} traffic={self.traffic_percentage}%>"
        )


# ---------------------------------------------------------------------------
# Incident table
# ---------------------------------------------------------------------------
class Incident(Base):
    """
    Diagnosis report created by the diagnosis agent in response to an alert.

    Statuses:
      - 'open'          — incident created, pending processing or manual review
      - 'auto_resolved' — automated action (e.g. rollback) successfully executed
      - 'escalated'     — low confidence, unclear cause, or impossible rollback
    """

    __tablename__ = "incidents"
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    alert_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="ID of the triggering alert row (not enforced FK)",
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Format: {model_name}:{version}, e.g. churn-model:v1",
    )
    hypothesis: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Root cause hypothesis string",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Numeric confidence score between 0.0 and 1.0",
    )
    evidence: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Structured evidence gathered (deployments, injections, drift context, reasoning log)",
    )
    recommended_action: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Action: rollback, fix_upstream_data, monitor, retrain, escalate",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="open",
        index=True,
        comment="One of: open, auto_resolved, escalated",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when action was executed or resolved",
    )

    # ── LLM Layer Fields (Phase 5) ──────────────────────────────────────────
    llm_explanation: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="LLM generated 2-3 sentence plain English explanation for human reviewers",
    )
    llm_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="LLM confidence score between 0.0 and 1.0",
    )
    llm_suggested_action: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="LLM suggested next step for human operator",
    )

    def __repr__(self) -> str:
        return (
            f"<Incident id={self.id!r} model={self.model_id!r} "
            f"action={self.recommended_action!r} conf={self.confidence:.2f} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# LLMUsage table (Cost & Token Audit Log)
# ---------------------------------------------------------------------------
class LLMUsage(Base):
    """
    Audit log of token consumption and API costs per LLM reasoning call.
    Demonstrates cost tracking and resource monitoring.
    """

    __tablename__ = "llm_usage"
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    incident_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="Associated incident ticket ID",
    )
    model_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="LLM model name (e.g., llama-3.3-70b-versatile)",
    )
    tokens_in: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Input prompt token count",
    )
    tokens_out: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Generated completion token count",
    )
    cost_estimate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Estimated cost in USD for this API call",
    )
    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<LLMUsage model={self.model_name!r} tokens_in={self.tokens_in} "
            f"tokens_out={self.tokens_out} cost=${self.cost_estimate:.6f}>"
        )


# ---------------------------------------------------------------------------
# CanaryDeployment table (Canary State Tracking)
# ---------------------------------------------------------------------------
class CanaryDeployment(Base):
    """
    State tracking table for active canary deployments.
    """

    __tablename__ = "canary_deployments"
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Model name (e.g. churn-model)",
    )
    candidate_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Canary candidate version tag (e.g. v2)",
    )
    baseline_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Current main production version tag (e.g. v1)",
    )
    traffic_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=10.0,
        comment="Current traffic allocation percentage to canary",
    )
    predictions_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total predictions served by canary version during observation",
    )
    min_observation_predictions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=50,
        comment="Required prediction count before auto-promotion evaluation",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        index=True,
        comment="Status: active, promoted, failed",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<CanaryDeployment model={self.model_name!r} candidate={self.candidate_version!r} "
            f"status={self.status!r} count={self.predictions_count}/{self.min_observation_predictions}>"
        )
