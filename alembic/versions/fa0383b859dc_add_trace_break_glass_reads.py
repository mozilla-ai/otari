"""Add the record of each opening of a session's content without its owner's share.

An organization owner, or the deployment's operator where it is turned on, can
open a session's content with a reason (and, for the operator, a reference).
Each opening is recorded before the content opens, kept apart from the session
so it outlives it, and deleted with the organization or after its retention.

Revision ID: fa0383b859dc
Revises: a536c003bcf5
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "fa0383b859dc"
down_revision: str | Sequence[str] | None = "a536c003bcf5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trace_break_glass_reads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("session_user_id", sa.String(), nullable=True),
        sa.Column("reader_user_id", sa.Uuid(), nullable=True),
        sa.Column("reader_email", sa.String(length=320), nullable=False),
        sa.Column("reason", sa.String(length=1000), nullable=False),
        sa.Column("reference", sa.String(length=200), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("kind IN ('organization', 'deployment')", name="check_trace_break_glass_reads_kind"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reader_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_trace_break_glass_reads_organization_read_at", "trace_break_glass_reads", ["organization_id", "read_at"]
    )
    op.create_index("ix_trace_break_glass_reads_read_at", "trace_break_glass_reads", ["read_at"])
    op.create_index("ix_trace_break_glass_reads_reader_user_id", "trace_break_glass_reads", ["reader_user_id"])


def downgrade() -> None:
    op.drop_index("ix_trace_break_glass_reads_reader_user_id", table_name="trace_break_glass_reads")
    op.drop_index("ix_trace_break_glass_reads_read_at", table_name="trace_break_glass_reads")
    op.drop_index("ix_trace_break_glass_reads_organization_read_at", table_name="trace_break_glass_reads")
    op.drop_table("trace_break_glass_reads")
