"""Record the workspace a sign-in to an MCP server started from.

A sign-in to an organization's server finishes only for someone who still
reaches the workspace it started from, and the server itself names no
workspace. ``mcp_oauth_flows`` gains a nullable ``workspace_id``: a sign-in
already in flight keeps none and falls back to its server's workspace, and
lives minutes at most.

Revision ID: da9ab6d90202
Revises: 124a7328c1b3
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "da9ab6d90202"
down_revision: str | Sequence[str] | None = "124a7328c1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "mcp_oauth_flows"
_INDEXES = {
    "ix_mcp_oauth_flows_server_id": "server_id",
    "ix_mcp_oauth_flows_user_id": "user_id",
    "ix_mcp_oauth_flows_expires_at": "expires_at",
}
_WORKSPACE_INDEX = "ix_mcp_oauth_flows_workspace_id"
# PostgreSQL's own name for the key, which is what it would have been had the table been created with it.
_WORKSPACE_KEY = "mcp_oauth_flows_workspace_id_fkey"


def _flows_table(*, upgraded: bool) -> sa.Table:
    """``mcp_oauth_flows`` as it stands when its batch rebuild starts, for SQLite.

    ``workspace_id`` is already added, and its key is there only once ``upgraded``. Spelled out
    rather than reflected, as ``dbd76ac9d948`` does. The indexes are dropped before the rebuild and
    created after it.
    """
    return sa.Table(
        _TABLE,
        sa.MetaData(),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("identity_id", sa.Uuid(), nullable=True),
        sa.Column("encrypted_code_verifier", sa.Text(), nullable=False),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("client_auth_method", sa.String(), nullable=False),
        sa.Column("token_endpoint", sa.Text(), nullable=False),
        sa.Column("revocation_endpoint", sa.Text(), nullable=True),
        sa.Column("resource", sa.Text(), nullable=True),
        sa.Column("redirect_uri", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("server_url", sa.Text(), nullable=False),
        sa.Column("issuer", sa.Text(), nullable=True),
        sa.Column("requires_iss", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("return_to", sa.Text(), nullable=True),
        sa.Column("for_check", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("for_shared", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.PrimaryKeyConstraint("state_hash"),
        sa.ForeignKeyConstraint(["server_id"], ["workspace_mcp_servers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        *(
            [sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE", name=_WORKSPACE_KEY)]
            if upgraded
            else []
        ),
    )


def _drop_indexes() -> None:
    for name in _INDEXES:
        op.drop_index(name, table_name=_TABLE)


def _create_indexes() -> None:
    for name, column in _INDEXES.items():
        op.create_index(name, _TABLE, [column])


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(_TABLE, sa.Column("workspace_id", sa.Uuid(), nullable=True))
    _drop_indexes()
    with op.batch_alter_table(_TABLE, copy_from=_flows_table(upgraded=False)) as batch:
        batch.create_foreign_key(_WORKSPACE_KEY, "workspace", ["workspace_id"], ["id"], ondelete="CASCADE")
    _create_indexes()
    op.create_index(_WORKSPACE_INDEX, _TABLE, ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(_WORKSPACE_INDEX, table_name=_TABLE)
    _drop_indexes()
    with op.batch_alter_table(_TABLE, copy_from=_flows_table(upgraded=True)) as batch:
        batch.drop_constraint(_WORKSPACE_KEY, type_="foreignkey")
        batch.drop_column("workspace_id")
    _create_indexes()
