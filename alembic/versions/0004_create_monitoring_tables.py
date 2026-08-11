"""Create monitoring tables (drift_reports, accuracy_reports, alerts)

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-10 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa  # noqa: E402
from alembic import op  # noqa: E402

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. drift_reports table
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.String(128),
            nullable=False,
            index=True,
            comment="Format: {model_name}:{version}, e.g. churn-model:v1",
        ),
        sa.Column(
            "feature_name",
            sa.String(128),
            nullable=False,
            index=True,
            comment="Feature column name or '_prediction' for model outputs",
        ),
        sa.Column(
            "checked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("psi_score", sa.Float(), nullable=False),
        sa.Column("method", sa.String(64), nullable=False, server_default="PSI"),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="stable"),
    )
    op.create_index("ix_drift_reports_model_id", "drift_reports", ["model_id"])
    op.create_index("ix_drift_reports_feature_name", "drift_reports", ["feature_name"])
    op.create_index("ix_drift_reports_checked_at", "drift_reports", ["checked_at"])

    # 2. accuracy_reports table
    op.create_table(
        "accuracy_reports",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.String(128),
            nullable=False,
            index=True,
            comment="Format: {model_name}:{version}, e.g. churn-model:v1",
        ),
        sa.Column(
            "window_start", sa.DateTime(timezone=True), nullable=False, index=True
        ),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_accuracy_reports_model_id", "accuracy_reports", ["model_id"])

    # 3. alerts table
    op.create_table(
        "alerts",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.String(128),
            nullable=False,
            index=True,
            comment="Format: {model_name}:{version}, e.g. churn-model:v1",
        ),
        sa.Column(
            "alert_type",
            sa.String(64),
            nullable=False,
            index=True,
            comment="One of: feature_drift, prediction_drift, accuracy_drop, high_null_rate",
        ),
        sa.Column(
            "severity",
            sa.String(32),
            nullable=False,
            index=True,
            comment="Severity level: info, warning, critical",
        ),
        sa.Column(
            "triggered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("details", sa.JSON(), nullable=False),
    )
    op.create_index("ix_alerts_model_id", "alerts", ["model_id"])
    op.create_index("ix_alerts_alert_type", "alerts", ["alert_type"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])
    op.create_index("ix_alerts_triggered_at", "alerts", ["triggered_at"])


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("accuracy_reports")
    op.drop_table("drift_reports")
