"""Add attach_by_default to workspace MCP servers.

Defaults to false, so every stored server stays reachable only through a
request that names it, as before.

Revision ID: e5b2d8f4a1c7
Revises: c4e9a1d7f3b6
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5b2d8f4a1c7"
down_revision: str | Sequence[str] | None = "c4e9a1d7f3b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspace_mcp_servers",
        sa.Column("attach_by_default", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("workspace_mcp_servers", "attach_by_default")
