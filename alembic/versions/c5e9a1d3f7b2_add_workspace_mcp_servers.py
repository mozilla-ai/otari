"""Add the workspace-scoped MCP server table.

One row is one MCP server a workspace has configured: a URL, an optional
bearer token encrypted with ``OTARI_SECRET_KEY``, and the purpose hint and
tool allow-list the tool loop already understands from an inline
``mcp_servers`` entry. A request references them by id with
``mcp_server_ids``, which until now was hybrid-only.

``(workspace_id, name)`` is unique: the name is how an operator and the tool
loop both identify a server, so two rows sharing one within a workspace would
silently hide a server rather than fail.

``workspace_id`` cascades, matching ``workspace_budget_defaults`` and the
provider-key tables: this is a workspace-owned configuration row, not durable
request-plane history like ``usage_logs`` or ``api_keys``.

Revision ID: c5e9a1d3f7b2
Revises: b3d5f7a9c1e6
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c5e9a1d3f7b2"
down_revision: str | Sequence[str] | None = "b3d5f7a9c1e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNIQUE_NAME = "uq_workspace_mcp_servers_workspace_name"


def upgrade() -> None:
    op.create_table(
        "workspace_mcp_servers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("encrypted_token", sa.Text(), nullable=True),
        sa.Column("purpose_hint", sa.Text(), nullable=True),
        sa.Column("allowed_tools", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("workspace_id", "name", name=_UNIQUE_NAME),
    )
    op.create_index(
        op.f("ix_workspace_mcp_servers_workspace_id"),
        "workspace_mcp_servers",
        ["workspace_id"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_workspace_mcp_servers_workspace_id"), table_name="workspace_mcp_servers")
    op.drop_table("workspace_mcp_servers")
