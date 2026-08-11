"""Create outcomes table

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-09 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "outcomes",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "prediction_id",
            sa.String(36),
            nullable=False,
            comment="ID of the corresponding predictions row (not enforced FK)",
        ),
        sa.Column(
            "true_label",
            sa.Boolean(),
            nullable=False,
            comment="True = customer churned, False = did not churn",
        ),
        sa.Column(
            "delay_seconds",
            sa.Float(),
            nullable=False,
            comment="Intended delay between prediction and outcome arrival",
        ),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="When the true label was written (simulated observation time)",
        ),
    )
    op.create_index("ix_outcomes_prediction_id", "outcomes", ["prediction_id"])
    op.create_index("ix_outcomes_observed_at", "outcomes", ["observed_at"])


def downgrade() -> None:
    op.drop_index("ix_outcomes_observed_at", table_name="outcomes")
    op.drop_index("ix_outcomes_prediction_id", table_name="outcomes")
    op.drop_table("outcomes")
