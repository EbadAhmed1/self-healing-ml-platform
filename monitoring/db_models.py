"""
monitoring/db_models.py
───────────────────────
SQLAlchemy ORM models owned by the monitoring subsystem.

Tables:
  drift_reports    — stores per-feature and prediction PSI drift scores
  accuracy_reports — stores windowed evaluation metrics (precision, recall, f1, roc_auc)
  alerts           — stores audit trail of triggered drift / accuracy degradation alerts
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


# ---------------------------------------------------------------------------
# DriftReport table
# ---------------------------------------------------------------------------
class DriftReport(Base):
    """
    Stores Population Stability Index (PSI) results for a model's features
    and output predictions.
    """

    __tablename__ = "drift_reports"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Format: {model_name}:{version}, e.g. churn-model:v1",
    )
    feature_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Feature column name or '_prediction' for model outputs",
    )
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    psi_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Calculated Population Stability Index score",
    )
    method: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="PSI",
        comment="Method used to measure distribution shift (PSI)",
    )
    sample_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of production predictions evaluated",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="stable",
        comment="Drift classification: stable (PSI<0.1), moderate (0.1<=PSI<0.25), significant (PSI>=0.25), insufficient_data",
    )

    def __repr__(self) -> str:
        return (
            f"<DriftReport model={self.model_id!r} feature={self.feature_name!r} "
            f"psi={self.psi_score:.4f} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# AccuracyReport table
# ---------------------------------------------------------------------------
class AccuracyReport(Base):
    """
    Stores rolling evaluation metrics calculated by joining logged predictions
    with delayed ground-truth outcomes.
    """

    __tablename__ = "accuracy_reports"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Format: {model_name}:{version}, e.g. churn-model:v1",
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    precision: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Precision score (positive class = churn)"
    )
    recall: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Recall score (positive class = churn)"
    )
    f1: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="F1-score (harmonic mean of precision and recall)"
    )
    roc_auc: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Area under ROC curve"
    )
    n_samples: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of joined prediction-outcome pairs evaluated",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def __repr__(self) -> str:
        f1_str = f"{self.f1:.4f}" if self.f1 is not None else "None"
        return (
            f"<AccuracyReport model={self.model_id!r} f1={f1_str} "
            f"n_samples={self.n_samples}>"
        )


# ---------------------------------------------------------------------------
# Alert table
# ---------------------------------------------------------------------------
class Alert(Base):
    """
    Audit record of triggered alerts when drift or performance metrics cross
    configured thresholds.
    """

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Format: {model_name}:{version}, e.g. churn-model:v1",
    )
    alert_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="One of: feature_drift, prediction_drift, accuracy_drop, high_null_rate",
    )
    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="Severity level: info, warning, critical",
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    processed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="True once evaluated by diagnosis agent",
    )
    details: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Structured payload (e.g. {feature: tenure, psi: 0.35, threshold: 0.25})",
    )

    def __repr__(self) -> str:
        return (
            f"<Alert type={self.alert_type!r} severity={self.severity!r} "
            f"model={self.model_id!r}>"
        )
