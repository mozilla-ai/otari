"""Add cache pricing columns to model_pricing.

Revision ID: c4d6e8f0a2b4
Revises: f3b6d8a1c4e7
Create Date: 2026-07-23 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4d6e8f0a2b4"
down_revision: str | Sequence[str] | None = "f3b6d8a1c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: providers/models without prompt caching leave these unset.
    # The cost calculation in log_usage only prices cache tokens when a rate
    # is configured, so existing rows are unaffected.
    op.add_column(
        "model_pricing",
        sa.Column("cache_read_price_per_million", sa.Float(), nullable=True),
    )
    op.add_column(
        "model_pricing",
        sa.Column("cache_write_price_per_million", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("model_pricing", "cache_write_price_per_million")
    op.drop_column("model_pricing", "cache_read_price_per_million")
