"""Add tags to usage_logs.

Caller-supplied attribution read from a request's ``metadata`` (``purpose``,
``country``), so spend can be reported by the feature or market that caused it.
Nullable with no backfill: a row written before this column has no tags, and
null reads correctly as "none sent".

Revision ID: a9d3e5f7b1c2
Revises: d4f8b2a6c1e9
Create Date: 2026-10-07 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a9d3e5f7b1c2"
down_revision: str | Sequence[str] | None = "d4f8b2a6c1e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("usage_logs", sa.Column("tags", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "tags")
