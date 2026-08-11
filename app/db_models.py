"""
app/db_models.py
────────────────
SQLAlchemy ORM models.

Table: predictions
  Stores every inference request for audit, monitoring, and drift detection.

Design note on model_id:
  `model_id` is a VARCHAR (not a FK) intentionally. We will have multiple
  models (churn, LTV, etc.) and eventually a proper model registry (Phase 4).
  Keeping it as a plain string means:
    - No FK constraint to maintain now
    - No schema migration needed when a second model is added
    - Easy to query by model name + version: WHERE model_id = 'churn-model:v1'

  Format: "{model_name}:{version}", e.g. "churn-model:v1"
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Format: {model_name}:{version}, e.g. churn-model:v1",
    )
    input_features: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Raw input dict as sent by the caller (post-validation)",
    )
    prediction: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        comment="True = churn predicted, False = no churn",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Probability of the predicted class (from predict_proba)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id!r} model={self.model_id!r} "
            f"prediction={self.prediction} confidence={self.confidence:.4f}>"
        )
