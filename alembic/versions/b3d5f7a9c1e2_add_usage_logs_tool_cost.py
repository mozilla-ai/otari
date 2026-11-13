"""Add usage_logs.tool_cost, the part of a row's cost that is per-call tool charges.

Revision ID: b3d5f7a9c1e2
Revises: ed6311bc29a0
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3d5f7a9c1e2"
down_revision: str | Sequence[str] | None = "ed6311bc29a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable with no backfill: an older row's tool charges survive only as float
    # amounts on its pricing_breakdown, so it reads as "not recorded".
    op.add_column("usage_logs", sa.Column("tool_cost", sa.Numeric(18, 6), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "tool_cost")
