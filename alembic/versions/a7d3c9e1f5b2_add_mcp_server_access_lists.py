"""Add access lists to stored MCP servers.

``everyone_tools`` on workspace_mcp_servers defaults to every tool, and
``request_mcp_servers_allowed`` on workspace defaults true, so every stored
server stays reachable by every key of its workspace, and a request may still
name its own servers, as before. ``workspace_mcp_server_grants`` starts empty.

Revision ID: a7d3c9e1f5b2
Revises: d9f1a3c5e7b2
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d3c9e1f5b2"
down_revision: str | Sequence[str] | None = "d9f1a3c5e7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspace_mcp_servers",
        sa.Column("everyone_tools", sa.JSON(), nullable=False, server_default='["*"]'),
    )
    op.add_column(
        "workspace",
        sa.Column("request_mcp_servers_allowed", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_table(
        "workspace_mcp_server_grants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("tools", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(api_key_id IS NULL) <> (user_id IS NULL)", name="ck_workspace_mcp_server_grants_one_subject"
        ),
        sa.ForeignKeyConstraint(["server_id"], ["workspace_mcp_servers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("server_id", "api_key_id", name="uq_workspace_mcp_server_grants_server_key"),
        sa.UniqueConstraint("server_id", "user_id", name="uq_workspace_mcp_server_grants_server_user"),
    )
    op.create_index(
        "ix_workspace_mcp_server_grants_api_key_id", "workspace_mcp_server_grants", ["api_key_id"], unique=False
    )
    op.create_index("ix_workspace_mcp_server_grants_user_id", "workspace_mcp_server_grants", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_workspace_mcp_server_grants_user_id", table_name="workspace_mcp_server_grants")
    op.drop_index("ix_workspace_mcp_server_grants_api_key_id", table_name="workspace_mcp_server_grants")
    op.drop_table("workspace_mcp_server_grants")
    op.drop_column("workspace", "request_mcp_servers_allowed")
    op.drop_column("workspace_mcp_servers", "everyone_tools")
