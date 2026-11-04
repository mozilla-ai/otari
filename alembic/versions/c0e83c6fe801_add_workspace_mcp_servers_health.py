"""Add health to workspace_mcp_servers.

Records the last connection check a stored MCP server had: when it ran, and
either how many tools the server offered or why it failed. Nullable with no
default, so existing rows read as never checked.

Revision ID: c0e83c6fe801
Revises: e3a5c7f9b1d4
Create Date: 2026-10-16 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c0e83c6fe801"
down_revision: str | Sequence[str] | None = "e3a5c7f9b1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("workspace_mcp_servers", sa.Column("health", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("workspace_mcp_servers", "health")
