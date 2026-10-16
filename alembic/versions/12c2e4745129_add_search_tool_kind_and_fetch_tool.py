"""Add kind and fetch_tool to search_tool_credentials.

A stored row is now a search or a fetch instance, as ``kind`` says. Every row
written before this column is a search instance, so the server default backfills
them as ``search``. ``fetch_tool`` names the fetch instance that enriches a
search row's results; nullable with no backfill, since null means the fetch
default does.

Revision ID: 12c2e4745129
Revises: c7e1a4d9b3f2
Create Date: 2026-10-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "12c2e4745129"
down_revision: str | Sequence[str] | None = "c7e1a4d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "search_tool_credentials",
        sa.Column("kind", sa.String(), nullable=False, server_default="search"),
    )
    op.add_column("search_tool_credentials", sa.Column("fetch_tool", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema.

    The stored fetch instances go: without ``kind``, the release this returns to
    would read them as search tools, and run searches on a fetch key.
    """
    op.execute(sa.text("DELETE FROM search_tool_credentials WHERE kind = 'fetch'"))
    op.drop_column("search_tool_credentials", "fetch_tool")
    op.drop_column("search_tool_credentials", "kind")
