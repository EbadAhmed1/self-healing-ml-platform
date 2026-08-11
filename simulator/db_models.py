"""
simulator/db_models.py
──────────────────────
SQLAlchemy ORM models owned by the simulator.

These tables are separate from app/db_models.py because they are
simulator-infrastructure, not serving-layer concerns. They share the same
`Base` (and therefore the same database) so foreign-key-like joins work,
but we use VARCHAR for prediction_id rather than a true FK to allow
graceful skips when a prediction was never logged (e.g., API was down).

Tables:
  outcomes            — ground-truth labels written after a configurable delay
  simulation_events   — audit log of every injection start/stop event
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


# ---------------------------------------------------------------------------
# Outcome table
# ---------------------------------------------------------------------------
class Outcome(Base):
    """
    True label for a past prediction, written after a configurable delay.

    `prediction_id` is VARCHAR (not a real FK) so that:
      - Rows can be inserted even if the prediction row doesn't exist
        (e.g., API was down and prediction wasn't logged)
      - No cascade complexity when predictions are cleaned up

    `delay_seconds` records the *intended* delay at time of queuing,
    allowing Phase 3 to compute "how stale was the label when it arrived?"
    """

    __tablename__ = "outcomes"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    prediction_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="ID of the corresponding predictions row (not enforced FK)",
    )
    true_label: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        comment="True = customer churned, False = did not churn",
    )
    delay_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Intended delay between prediction and outcome arrival",
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="When the true label was written (simulated observation time)",
    )

    def __repr__(self) -> str:
        return (
            f"<Outcome prediction_id={self.prediction_id!r} "
            f"true_label={self.true_label} delay={self.delay_seconds:.0f}s>"
        )


# ---------------------------------------------------------------------------
# SimulationEvent table
# ---------------------------------------------------------------------------
class SimulationEvent(Base):
    """
    Audit log of every injection start and stop.

    CRITICAL FOR DEMO: lets you overlay "drift injected at T" on top of
    "drift alert fired at T+Δ" in the Phase 3 dashboard.

    event_type values (enforced in simulator/injection.py):
      drift_feature     — numeric feature shift
      drift_category    — categorical distribution skew
      label_delay_spike — outcome delay multiplied
      corrupt_feature   — null / out-of-range values sent to API
      stop              — one of the above ended
    """

    __tablename__ = "simulation_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="One of: drift_feature, drift_category, label_delay_spike, corrupt_feature, stop",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Null while injection is active; set when stopped",
    )
    parameters: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="e.g. {feature: tenure, magnitude: 20.0} or {feature: Contract, value: Month-to-month, skew_prob: 0.8}",
    )

    def __repr__(self) -> str:
        active = "active" if self.ended_at is None else f"ended@{self.ended_at}"
        return f"<SimulationEvent type={self.event_type!r} {active} params={self.parameters}>"
