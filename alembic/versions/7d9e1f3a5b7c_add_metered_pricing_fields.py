"""Add metered pricing and usage-audit fields.

Revision ID: 7d9e1f3a5b7c
Revises: f3b6d8a1c4e7
Create Date: 2026-07-22 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7d9e1f3a5b7c"
down_revision: str | Sequence[str] | None = "c4d6e8f0a2b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("model_pricing", sa.Column("cache_write_1h_price_per_million", sa.Float(), nullable=True))
    op.add_column(
        "model_pricing",
        sa.Column("pricing_tiers", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.add_column("usage_logs", sa.Column("cache_write_1h_tokens", sa.Integer(), nullable=True))
    op.add_column("usage_logs", sa.Column("billing_meters", sa.JSON(), nullable=True))
    op.add_column("usage_logs", sa.Column("pricing_breakdown", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "pricing_breakdown")
    op.drop_column("usage_logs", "billing_meters")
    op.drop_column("usage_logs", "cache_write_1h_tokens")
    op.drop_column("model_pricing", "pricing_tiers")
    op.drop_column("model_pricing", "cache_write_1h_price_per_million")
