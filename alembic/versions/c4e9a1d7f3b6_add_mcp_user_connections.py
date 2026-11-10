"""Add per-user MCP connections.

``credential_mode`` on workspace_mcp_servers defaults to ``shared``, so every
stored server keeps sending its one workspace token. ``mcp_user_connections``
holds each user's own tokens for a per-user server, and ``mcp_oauth_flows`` the
sign-ins that have started and not yet come back; both start empty.

Revision ID: c4e9a1d7f3b6
Revises: c4f9e2a7d5b8
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e9a1d7f3b6"
down_revision: str | Sequence[str] | None = "c4f9e2a7d5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspace_mcp_servers",
        sa.Column("credential_mode", sa.String(), nullable=False, server_default="shared"),
    )
    op.add_column("workspace_mcp_servers", sa.Column("oauth_client", sa.JSON(), nullable=True))
    op.create_table(
        "mcp_user_connections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
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
        sa.ForeignKeyConstraint(["server_id"], ["workspace_mcp_servers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("server_id", "user_id", name="uq_mcp_user_connections_server_user"),
    )
    op.create_index("ix_mcp_user_connections_user_id", "mcp_user_connections", ["user_id"], unique=False)
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
        sa.Column("return_to", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["server_id"], ["workspace_mcp_servers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state_hash"),
    )
    op.create_index("ix_mcp_oauth_flows_server_id", "mcp_oauth_flows", ["server_id"], unique=False)
    op.create_index("ix_mcp_oauth_flows_user_id", "mcp_oauth_flows", ["user_id"], unique=False)
    op.create_index("ix_mcp_oauth_flows_expires_at", "mcp_oauth_flows", ["expires_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_mcp_oauth_flows_expires_at", table_name="mcp_oauth_flows")
    op.drop_index("ix_mcp_oauth_flows_user_id", table_name="mcp_oauth_flows")
    op.drop_index("ix_mcp_oauth_flows_server_id", table_name="mcp_oauth_flows")
    op.drop_table("mcp_oauth_flows")
    op.drop_index("ix_mcp_user_connections_user_id", table_name="mcp_user_connections")
    op.drop_table("mcp_user_connections")
    op.drop_column("workspace_mcp_servers", "oauth_client")
    op.drop_column("workspace_mcp_servers", "credential_mode")
