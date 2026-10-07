"""Add trace content references, per-session data keys, content reads, and each workspace's capture level.

Content is off in every workspace until an admin turns it on, so the settings
table starts empty and a missing row means "off". Content itself lives in the
object store, sealed with its session's data key; the database keeps a reference
per span, each session's key wrapped, and a record of every read.

Revision ID: 4cc0ae37c36c
Revises: 12d51fa2acc0
Create Date: 2026-10-07 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4cc0ae37c36c"
down_revision: str | Sequence[str] | None = "12d51fa2acc0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ID = sa.String(64)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "trace_span_content",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", _ID, nullable=False),
        sa.Column("span_id", _ID, nullable=False),
        sa.Column("storage_ref", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "trace_id", "span_id"),
        sa.ForeignKeyConstraint(
            ["workspace_id", "trace_id", "span_id"],
            ["trace_spans.workspace_id", "trace_spans.trace_id", "trace_spans.span_id"],
            ondelete="CASCADE",
            name="fk_trace_span_content_span",
        ),
    )
    op.create_index("ix_trace_span_content_created", "trace_span_content", ["created_at"])

    op.create_table(
        "trace_content_keys",
        sa.Column("workspace_id", sa.Uuid(), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trace_id", _ID, nullable=False),
        sa.Column("key_ref", sa.String(), nullable=False),
        sa.Column("wrapped", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "trace_id"),
    )
    op.create_index("ix_trace_content_keys_created_at", "trace_content_keys", ["created_at"])

    op.create_table(
        "trace_content_access",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", _ID, nullable=False),
        sa.Column("span_id", _ID, nullable=False),
        sa.Column("reader_kind", sa.String(16), nullable=False),
        sa.Column("reader", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(500), nullable=True),
        sa.Column("accessed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "reader_kind IN ('owner', 'admin', 'break_glass')", name="ck_trace_content_access_reader_kind"
        ),
    )
    op.create_index("ix_trace_content_access_workspace_at", "trace_content_access", ["workspace_id", "accessed_at"])

    op.create_table(
        "workspace_trace_settings",
        sa.Column("workspace_id", sa.Uuid(), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content_capture", sa.String(), server_default="off", nullable=False),
        sa.Column("admin_content_access", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("updated_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("workspace_trace_settings")
    op.drop_index("ix_trace_content_access_workspace_at", table_name="trace_content_access")
    op.drop_table("trace_content_access")
    op.drop_index("ix_trace_content_keys_created_at", table_name="trace_content_keys")
    op.drop_table("trace_content_keys")
    op.drop_index("ix_trace_span_content_created", table_name="trace_span_content")
    op.drop_table("trace_span_content")
