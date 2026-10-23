"""Add shares of a session's content, and the record of each read through one.

A session's content opens only for its owner. A share lets the organization's
admins, or the deployment's operator, read it until it expires or the owner
revokes it, and every read through it is kept for the owner to see. Both go
with the session.

Revision ID: e1a3c5e7b9d4
Revises: d9f1a3c5e7b2
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1a3c5e7b9d4"
down_revision: str | Sequence[str] | None = "d9f1a3c5e7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trace_content_shares",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("audience", sa.String(length=32), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column("shared_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "audience IN ('organization_admins', 'deployment_operator')", name="check_trace_content_shares_audience"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "trace_id"],
            ["traces.workspace_id", "traces.trace_id"],
            ondelete="CASCADE",
            name="fk_trace_content_shares_trace",
        ),
        sa.ForeignKeyConstraint(["shared_by_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_content_shares_trace", "trace_content_shares", ["workspace_id", "trace_id"])
    op.create_index("ix_trace_content_shares_shared_by_user_id", "trace_content_shares", ["shared_by_user_id"])
    op.create_table(
        "trace_content_reads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("share_id", sa.Uuid(), nullable=False),
        sa.Column("reader_user_id", sa.Uuid(), nullable=True),
        sa.Column("reader_email", sa.String(length=320), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["share_id"], ["trace_content_shares.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reader_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_content_reads_share_id", "trace_content_reads", ["share_id"])
    op.create_index("ix_trace_content_reads_reader_user_id", "trace_content_reads", ["reader_user_id"])


def downgrade() -> None:
    op.drop_index("ix_trace_content_reads_reader_user_id", table_name="trace_content_reads")
    op.drop_index("ix_trace_content_reads_share_id", table_name="trace_content_reads")
    op.drop_table("trace_content_reads")
    op.drop_index("ix_trace_content_shares_shared_by_user_id", table_name="trace_content_shares")
    op.drop_index("ix_trace_content_shares_trace", table_name="trace_content_shares")
    op.drop_table("trace_content_shares")
