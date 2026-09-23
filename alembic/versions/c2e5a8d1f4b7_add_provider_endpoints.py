"""Add the provider endpoints table.

A provider endpoint is a model server a workspace, or one user in it, brings to
the gateway: a base URL, an encrypted API key and default request fields,
reached as ``<name>:<model>``. Its own table rather than owner columns on
``org_provider_keys``, whose rows resolve by provider for a whole organization;
fusing the two would put an owner filter on every query of that table.

The uniqueness pair mirrors ``model_aliases``: a composite constraint for one
row per (workspace, name, user), and a partial index for one workspace-wide row
per (workspace, name), which the composite cannot give while ``user_id`` is
NULL.

Revision ID: c2e5a8d1f4b7
Revises: a9c4e7b2d5f8
Create Date: 2026-09-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2e5a8d1f4b7"
down_revision: str | Sequence[str] | None = "a9c4e7b2d5f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "provider_endpoints"
_USER_INDEX = "ix_provider_endpoints_user_id"
_SHARED_NAME_INDEX = "uq_provider_endpoints_workspace_shared_name"


def upgrade() -> None:
    op.create_table(
        "provider_endpoints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("api_base", sa.String(length=1024), nullable=False),
        sa.Column("encrypted_api_key", sa.String(), nullable=True),
        sa.Column("last4", sa.String(length=8), nullable=True),
        sa.Column("default_params", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "name", "user_id", name="uq_provider_endpoints_workspace_name_user"),
    )
    op.create_index(_USER_INDEX, _TABLE, ["user_id"])
    op.create_index(
        _SHARED_NAME_INDEX,
        _TABLE,
        ["workspace_id", "name"],
        unique=True,
        sqlite_where=sa.text("user_id IS NULL"),
        postgresql_where=sa.text("user_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(_SHARED_NAME_INDEX, table_name=_TABLE)
    op.drop_index(_USER_INDEX, table_name=_TABLE)
    op.drop_table(_TABLE)
