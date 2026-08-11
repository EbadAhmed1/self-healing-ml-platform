"""Create retraining and canary tables and columns

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-10 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add lineage & canary columns to deployments
    op.add_column(
        "deployments",
        sa.Column("status", sa.String(32), server_default="deployed", nullable=False),
    )
    op.add_column(
        "deployments",
        sa.Column("training_data_summary", sa.JSON(), nullable=True),
    )
    op.add_column(
        "deployments",
        sa.Column(
            "traffic_percentage", sa.Float(), server_default="100.0", nullable=False
        ),
    )
    op.add_column(
        "deployments",
        sa.Column("rejection_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "deployments",
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_deployments_status", "deployments", ["status"])

    # 2. Create canary_deployments table
    op.create_table(
        "canary_deployments",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "model_name",
            sa.String(64),
            nullable=False,
            index=True,
            comment="Model name (e.g. churn-model)",
        ),
        sa.Column("candidate_version", sa.String(64), nullable=False),
        sa.Column("baseline_version", sa.String(64), nullable=False),
        sa.Column(
            "traffic_percentage",
            sa.Float(),
            server_default="10.0",
            nullable=False,
        ),
        sa.Column(
            "predictions_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "min_observation_predictions",
            sa.Integer(),
            server_default="50",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(32),
            server_default="active",
            nullable=False,
            index=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_canary_deployments_model_name", "canary_deployments", ["model_name"]
    )
    op.create_index("ix_canary_deployments_status", "canary_deployments", ["status"])


def downgrade() -> None:
    op.drop_table("canary_deployments")
    op.drop_index("ix_deployments_status", table_name="deployments")
    op.drop_column("deployments", "promoted_at")
    op.drop_column("deployments", "rejection_reason")
    op.drop_column("deployments", "traffic_percentage")
    op.drop_column("deployments", "training_data_summary")
    op.drop_column("deployments", "status")
