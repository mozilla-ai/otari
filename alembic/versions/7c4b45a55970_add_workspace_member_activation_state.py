"""Add the setup guide's per-member state table.

One row per workspace and person, beside the per-workspace
``workspace_activation_state``. A member's activation is read from
``usage_logs`` through ``ix_usage_logs_user_id_timestamp``, which already
exists, so no index is added there.

Both ``workspace_id`` and ``user_id`` cascade, because the row describes one
person in one workspace and means nothing without either. The primary key
leads with ``workspace_id``, so ``user_id`` gets an index for its cascade.
``api_key_id`` is ``SET NULL`` for the reason it is on the workspace's table.

Revision ID: 7c4b45a55970
Revises: a7c9e1b3d5f8
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c4b45a55970"
down_revision: str | Sequence[str] | None = "a7c9e1b3d5f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_member_activation_state",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("first_presented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_presented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "user_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
    )
    op.create_index(
        op.f("ix_workspace_member_activation_state_user_id"),
        "workspace_member_activation_state",
        ["user_id"],
    )
    op.create_index(
        op.f("ix_workspace_member_activation_state_api_key_id"),
        "workspace_member_activation_state",
        ["api_key_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_workspace_member_activation_state_api_key_id"),
        table_name="workspace_member_activation_state",
    )
    op.drop_index(
        op.f("ix_workspace_member_activation_state_user_id"),
        table_name="workspace_member_activation_state",
    )
    op.drop_table("workspace_member_activation_state")
