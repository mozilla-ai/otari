"""Index whose session each break-glass opening was, so a user's erasure finds its records without a scan.

Revision ID: b7d3e9f1a2c4
Revises: fa0383b859dc
Create Date: 2026-10-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b7d3e9f1a2c4"
down_revision: str | Sequence[str] | None = "fa0383b859dc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_trace_break_glass_reads_session_user_id", "trace_break_glass_reads", ["session_user_id"])


def downgrade() -> None:
    op.drop_index("ix_trace_break_glass_reads_session_user_id", table_name="trace_break_glass_reads")
