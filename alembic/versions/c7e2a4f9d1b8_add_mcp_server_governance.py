"""Add sign-in, attach-by-default, logos and member requests to a workspace's MCP servers.

On ``workspace_mcp_servers``: ``credential_mode`` defaults to ``shared``, so every
stored server keeps sending its one workspace token; ``attach_by_default`` defaults
to off; ``oauth_client_issuer`` and ``oauth_client_token_endpoint`` start empty, and
are filled by the first sign-in through the server's OAuth client; and ``read_tools``
becomes nullable, null meaning no manager has edited the labels, so they follow the
tools the server marks read-only.

A row with no labels becomes null only where no access rule uses ``@read``: neither the
server's ``everyone_tools`` nor any of its grants' ``tools`` holds it. Where one does, an
empty list admits no tool through ``@read`` today, and null would admit every tool the
server marks read-only, so the upgrade keeps the empty list and widens nobody's access.

The new tables all start empty: each user's own tokens for a per-user server, the one
account a manager signed in to for a shared server, the sign-ins that have started and
not yet come back, the logo a workspace admin sets for a server, and members' requests
for a catalog server. A request copies the catalog entry's name, title and URL rather
than referencing the catalog's tables.

Revision ID: c7e2a4f9d1b8
Revises: b3d6f8a2c5e1
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e2a4f9d1b8"
down_revision: str | Sequence[str] | None = "b3d6f8a2c5e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SERVERS = "workspace_mcp_servers"
_GRANTS = "workspace_mcp_server_grants"
# An access rule's tool patterns hold '@read' as one JSON string, quotes included.
_NAMES_READ = "'%\"@read\"%'"
_PENDING = sa.text("status = 'pending'")


def _connection_columns() -> list[sa.Column[object]]:
    """The tokens of one sign-in and what a refresh sends, shared by a user's connection and a shared one."""
    return [
        sa.Column("encrypted_access_token", sa.Text(), nullable=False),
        sa.Column("encrypted_refresh_token", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("client_auth_method", sa.String(), nullable=False),
        sa.Column("token_endpoint", sa.Text(), nullable=False),
        sa.Column("revocation_endpoint", sa.Text(), nullable=True),
        sa.Column("resource", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    with op.batch_alter_table(_SERVERS) as batch:
        batch.add_column(sa.Column("credential_mode", sa.String(), nullable=False, server_default="shared"))
        batch.add_column(sa.Column("oauth_client", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("oauth_client_issuer", sa.Text(), nullable=True))
        batch.add_column(sa.Column("oauth_client_token_endpoint", sa.Text(), nullable=True))
        batch.add_column(sa.Column("attach_by_default", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.alter_column(
            "read_tools", existing_type=sa.JSON(), nullable=True, existing_server_default="[]", server_default=None
        )
    op.execute(
        sa.text(
            f"UPDATE {_SERVERS} SET read_tools = NULL"
            " WHERE CAST(read_tools AS TEXT) IN ('[]', 'null')"
            f" AND CAST(everyone_tools AS TEXT) NOT LIKE {_NAMES_READ}"
            f" AND id NOT IN (SELECT server_id FROM {_GRANTS} WHERE CAST(tools AS TEXT) LIKE {_NAMES_READ})"
        )
    )

    op.create_table(
        "workspace_mcp_server_logos",
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("luminance", sa.Float(), nullable=True),
        sa.Column("tile", sa.String(), server_default="auto", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["server_id"], [f"{_SERVERS}.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("server_id"),
    )

    op.create_table(
        "mcp_user_connections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        *_connection_columns(),
        sa.ForeignKeyConstraint(["server_id"], [f"{_SERVERS}.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("server_id", "user_id", name="uq_mcp_user_connections_server_user"),
    )
    op.create_index("ix_mcp_user_connections_user_id", "mcp_user_connections", ["user_id"], unique=False)

    op.create_table(
        "mcp_shared_connections",
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("connected_by", sa.Uuid(), nullable=True),
        *_connection_columns(),
        sa.ForeignKeyConstraint(["server_id"], [f"{_SERVERS}.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["connected_by"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("server_id"),
    )
    op.create_index("ix_mcp_shared_connections_connected_by", "mcp_shared_connections", ["connected_by"], unique=False)

    op.create_table(
        "mcp_oauth_flows",
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
        sa.Column("issuer", sa.Text(), nullable=True),
        sa.Column("requires_iss", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("return_to", sa.Text(), nullable=True),
        sa.Column("for_check", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("for_shared", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["server_id"], [f"{_SERVERS}.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state_hash"),
    )
    op.create_index("ix_mcp_oauth_flows_server_id", "mcp_oauth_flows", ["server_id"], unique=False)
    op.create_index("ix_mcp_oauth_flows_user_id", "mcp_oauth_flows", ["user_id"], unique=False)
    op.create_index("ix_mcp_oauth_flows_expires_at", "mcp_oauth_flows", ["expires_at"], unique=False)

    op.create_table(
        "mcp_server_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("catalog_name", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("requester_id", sa.Uuid(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("decided_by", sa.Uuid(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column("server_id", sa.Uuid(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["requester_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["decided_by"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["server_id"], [f"{_SERVERS}.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_mcp_server_requests_pending",
        "mcp_server_requests",
        ["workspace_id", "catalog_name", "requester_id"],
        unique=True,
        postgresql_where=_PENDING,
        sqlite_where=_PENDING,
    )
    op.create_index(
        "ix_mcp_server_requests_workspace_status", "mcp_server_requests", ["workspace_id", "status"], unique=False
    )
    op.create_index("ix_mcp_server_requests_requester_id", "mcp_server_requests", ["requester_id"], unique=False)
    op.create_index("ix_mcp_server_requests_decided_by", "mcp_server_requests", ["decided_by"], unique=False)
    op.create_index("ix_mcp_server_requests_server_id", "mcp_server_requests", ["server_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_mcp_server_requests_server_id", table_name="mcp_server_requests")
    op.drop_index("ix_mcp_server_requests_decided_by", table_name="mcp_server_requests")
    op.drop_index("ix_mcp_server_requests_requester_id", table_name="mcp_server_requests")
    op.drop_index("ix_mcp_server_requests_workspace_status", table_name="mcp_server_requests")
    op.drop_index("uq_mcp_server_requests_pending", table_name="mcp_server_requests")
    op.drop_table("mcp_server_requests")
    op.drop_index("ix_mcp_oauth_flows_expires_at", table_name="mcp_oauth_flows")
    op.drop_index("ix_mcp_oauth_flows_user_id", table_name="mcp_oauth_flows")
    op.drop_index("ix_mcp_oauth_flows_server_id", table_name="mcp_oauth_flows")
    op.drop_table("mcp_oauth_flows")
    op.drop_index("ix_mcp_shared_connections_connected_by", table_name="mcp_shared_connections")
    op.drop_table("mcp_shared_connections")
    op.drop_index("ix_mcp_user_connections_user_id", table_name="mcp_user_connections")
    op.drop_table("mcp_user_connections")
    op.drop_table("workspace_mcp_server_logos")

    op.execute(
        sa.text(
            f"UPDATE {_SERVERS} SET read_tools = '[]' WHERE read_tools IS NULL OR CAST(read_tools AS TEXT) = 'null'"
        )
    )
    with op.batch_alter_table(_SERVERS) as batch:
        batch.alter_column("read_tools", existing_type=sa.JSON(), nullable=False, server_default="[]")
        batch.drop_column("attach_by_default")
        batch.drop_column("oauth_client_token_endpoint")
        batch.drop_column("oauth_client_issuer")
        batch.drop_column("oauth_client")
        batch.drop_column("credential_mode")
