"""Create llm_usage table and add llm columns to incidents table

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-10 00:00:00.000000 UTC
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add LLM columns to incidents table
    op.add_column("incidents", sa.Column("llm_explanation", sa.Text(), nullable=True))
    op.add_column("incidents", sa.Column("llm_confidence", sa.Float(), nullable=True))
    op.add_column(
        "incidents", sa.Column("llm_suggested_action", sa.String(128), nullable=True)
    )

    # 2. Create llm_usage table
    op.create_table(
        "llm_usage",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "incident_id",
            sa.String(36),
            nullable=False,
            index=True,
            comment="Associated incident ticket ID",
        ),
        sa.Column(
            "model_name",
            sa.String(64),
            nullable=False,
            comment="LLM model name (e.g. llama-3.3-70b-versatile)",
        ),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_estimate", sa.Float(), nullable=False),
        sa.Column(
            "called_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_llm_usage_incident_id", "llm_usage", ["incident_id"])
    op.create_index("ix_llm_usage_called_at", "llm_usage", ["called_at"])


def downgrade() -> None:
    op.drop_table("llm_usage")
    op.drop_column("incidents", "llm_suggested_action")
    op.drop_column("incidents", "llm_confidence")
    op.drop_column("incidents", "llm_explanation")
