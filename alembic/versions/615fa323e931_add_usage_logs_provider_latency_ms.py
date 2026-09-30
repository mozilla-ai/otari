"""Add usage_logs provider_latency_ms column.

Revision ID: 615fa323e931
Revises: a9c4e7b2d5f8
Create Date: 2026-09-11 23:06:18.336380

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "615fa323e931"
down_revision: str | Sequence[str] | None = "a9c4e7b2d5f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: historical rows predate the column, most providers report
    # nothing here, and extraction failure must never block a write.
    op.add_column("usage_logs", sa.Column("provider_latency_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "provider_latency_ms")
