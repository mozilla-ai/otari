"""Add usage_logs.deployment_paid_cost, the part of a row's cost the deployment's account ran.

Revision ID: a8c0e2f4b6d9
Revises: f6b8d0a2c4e7
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8c0e2f4b6d9"
down_revision: str | Sequence[str] | None = "f6b8d0a2c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable with no backfill: an older row's tool lines carry their payer only as
    # float amounts on its pricing_breakdown, so it reads as "not recorded".
    op.add_column("usage_logs", sa.Column("deployment_paid_cost", sa.Numeric(18, 6), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "deployment_paid_cost")
