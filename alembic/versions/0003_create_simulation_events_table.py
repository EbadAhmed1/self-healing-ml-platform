"""Create simulation_events table

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-09 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "simulation_events",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "event_type",
            sa.String(64),
            nullable=False,
            index=True,
            comment="One of: drift_feature, drift_category, label_delay_spike, corrupt_feature, stop",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "ended_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Null while injection is active; set when stopped",
        ),
        sa.Column(
            "parameters",
            sa.JSON(),
            nullable=False,
            comment="e.g. {feature: tenure, magnitude: 20.0}",
        ),
    )
    op.create_index(
        "ix_simulation_events_event_type", "simulation_events", ["event_type"]
    )
    op.create_index(
        "ix_simulation_events_started_at", "simulation_events", ["started_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_simulation_events_started_at", table_name="simulation_events")
    op.drop_index("ix_simulation_events_event_type", table_name="simulation_events")
    op.drop_table("simulation_events")
