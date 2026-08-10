"""Add coding-agent telemetry storage.

Revision ID: e8a7c6b5d4f3
Revises: f0a1b2c3d4e5
Create Date: 2026-08-06 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8a7c6b5d4f3"
down_revision: str | Sequence[str] | None = "f0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the content-free coding-agent telemetry table."""
    op.create_table(
        "agent_telemetry",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("temporality", sa.String(), nullable=True),
        sa.Column("series_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("series_key", sa.String(), nullable=True),
        sa.Column("tool_name", sa.String(), nullable=True),
        sa.Column("decision", sa.String(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("prompt_length", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("session_label", sa.String(), nullable=True),
        sa.Column("dedup_key", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "dedup_key", name="uq_agent_telemetry_source_dedup"),
    )
    op.create_index("ix_agent_telemetry_api_key_id", "agent_telemetry", ["api_key_id"])
    op.create_index("ix_agent_telemetry_user_id", "agent_telemetry", ["user_id"])
    op.create_index("ix_agent_telemetry_timestamp", "agent_telemetry", ["timestamp"])
    op.create_index("ix_agent_telemetry_source", "agent_telemetry", ["source"])
    op.create_index("ix_agent_telemetry_user_id_timestamp", "agent_telemetry", ["user_id", "timestamp"])
    op.create_index("ix_agent_telemetry_series_timestamp", "agent_telemetry", ["series_key", "timestamp"])


def downgrade() -> None:
    """Drop coding-agent telemetry storage."""
    op.drop_table("agent_telemetry")
