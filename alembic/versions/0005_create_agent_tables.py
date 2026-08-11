"""Create agent tables (deployments, incidents) and add processed column to alerts

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-10 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add processed boolean column to alerts table
    op.add_column(
        "alerts",
        sa.Column(
            "processed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
            comment="True once the diagnosis agent has evaluated this alert",
        ),
    )
    op.create_index("ix_alerts_processed", "alerts", ["processed"])

    # 2. deployments table
    op.create_table(
        "deployments",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.String(128),
            nullable=False,
            comment="Format: {model_name}:{version}, e.g. churn-model:v1",
        ),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column(
            "deployed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "git_commit", sa.String(64), nullable=False, server_default="unknown"
        ),
        sa.Column("eval_metrics", sa.JSON(), nullable=False),
        sa.Column(
            "is_current",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.create_index("ix_deployments_model_id", "deployments", ["model_id"])
    op.create_index("ix_deployments_deployed_at", "deployments", ["deployed_at"])

    # 3. incidents table
    op.create_table(
        "incidents",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "alert_id",
            sa.String(36),
            nullable=False,
            comment="ID of triggering alert",
        ),
        sa.Column(
            "model_id",
            sa.String(128),
            nullable=False,
            comment="Format: {model_name}:{version}",
        ),
        sa.Column("hypothesis", sa.String(256), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("recommended_action", sa.String(128), nullable=False),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="open",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_incidents_alert_id", "incidents", ["alert_id"])
    op.create_index("ix_incidents_model_id", "incidents", ["model_id"])
    op.create_index("ix_incidents_status", "incidents", ["status"])
    op.create_index("ix_incidents_created_at", "incidents", ["created_at"])


def downgrade() -> None:
    op.drop_table("incidents")
    op.drop_table("deployments")
    op.drop_index("ix_alerts_processed", table_name="alerts")
    op.drop_column("alerts", "processed")
