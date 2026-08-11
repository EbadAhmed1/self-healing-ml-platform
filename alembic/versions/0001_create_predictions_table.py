"""Create predictions table

Revision ID: 0001
Revises:
Create Date: 2026-08-09 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.String(128),
            nullable=False,
            comment="Format: {model_name}:{version}, e.g. churn-model:v1",
        ),
        sa.Column(
            "input_features",
            sa.JSON(),
            nullable=False,
            comment="Raw input dict as sent by the caller (post-validation)",
        ),
        sa.Column(
            "prediction",
            sa.Boolean(),
            nullable=False,
            comment="True = churn predicted, False = no churn",
        ),
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=False,
            comment="Probability of the predicted class (from predict_proba)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_predictions_model_id", "predictions", ["model_id"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_model_id", table_name="predictions")
    op.drop_table("predictions")
