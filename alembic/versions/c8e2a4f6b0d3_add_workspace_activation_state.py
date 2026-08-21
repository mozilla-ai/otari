"""Add the setup-guide state table, and the index the guide reads usage through.

One row per workspace, holding only what cannot be observed elsewhere: when the
dashboard's first-request setup guide last handed out an API key, which key that
was, and whether someone dismissed it. Whether a workspace has activated is not
a column here, because ``usage_logs`` already records it (the first successful
gateway request in the workspace), so this migration adds the composite index
that makes that lookup a seek rather than a scan of the workspace's traffic. It
covers ``source`` as well as ``status``, because imported usage is excluded from
the question, and on a deployment that imports, most of a workspace's rows are
the wrong source.

``api_key_id`` is ``SET NULL`` rather than cascade: deleting the guide's key from
the Keys page is a legitimate thing to do and must not take the dismissal with
it. ``workspace_id`` cascades, because the row describes a workspace and means
nothing without it.

Revision ID: c8e2a4f6b0d3
Revises: a7c3e5d9b1f4
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8e2a4f6b0d3"
down_revision: str | Sequence[str] | None = "a7c3e5d9b1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_USAGE_INDEX = "ix_usage_logs_workspace_source_status_timestamp"


def upgrade() -> None:
    op.create_table(
        "workspace_activation_state",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("first_presented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_presented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
    )
    op.create_index(
        op.f("ix_workspace_activation_state_api_key_id"),
        "workspace_activation_state",
        ["api_key_id"],
    )
    op.create_index(_USAGE_INDEX, "usage_logs", ["workspace_id", "source", "status", "timestamp"])


def downgrade() -> None:
    op.drop_index(_USAGE_INDEX, table_name="usage_logs")
    op.drop_index(
        op.f("ix_workspace_activation_state_api_key_id"),
        table_name="workspace_activation_state",
    )
    op.drop_table("workspace_activation_state")
