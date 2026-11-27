"""Count each MCP server's shared-account disconnects, so a sign-in started before one cannot store its account.

A shared sign-in consumes its pending flow before the token exchange, so a disconnect that
commits during the exchange leaves nothing for the completion to notice: the server's URL,
mode and client are unchanged. ``workspace_mcp_servers.shared_account_generation`` advances
on every disconnect, and ``mcp_oauth_flows.shared_account_generation`` records the value a
shared sign-in started at, so completion refuses one that a disconnect has overtaken. A
sign-in already in flight records none and is refused, which costs a retry.

Revision ID: ab0af3310ec6
Revises: da9ab6d90202
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "ab0af3310ec6"
down_revision: str | Sequence[str] | None = "da9ab6d90202"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "workspace_mcp_servers",
        sa.Column("shared_account_generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("mcp_oauth_flows", sa.Column("shared_account_generation", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("mcp_oauth_flows", "shared_account_generation")
    op.drop_column("workspace_mcp_servers", "shared_account_generation")
