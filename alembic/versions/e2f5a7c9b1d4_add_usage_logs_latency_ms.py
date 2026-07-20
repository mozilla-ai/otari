"""Add usage_logs latency_ms column and status/timestamp index.

Revision ID: e2f5a7c9b1d4
Revises: d1e4b8c6f0a3
Create Date: 2026-07-20 18:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e2f5a7c9b1d4"
down_revision: str | Sequence[str] | None = "d1e4b8c6f0a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: historical rows predate the column, and some write paths (batch
    # jobs, provider-never-reached rejections) have no request duration.
    op.add_column("usage_logs", sa.Column("latency_ms", sa.Integer(), nullable=True))
    # Supports the activity-log viewer's primary "show errors, newest-first"
    # query (status is low-cardinality; model is left unindexed on purpose).
    op.create_index("ix_usage_logs_status_timestamp", "usage_logs", ["status", "timestamp"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_usage_logs_status_timestamp", table_name="usage_logs")
    op.drop_column("usage_logs", "latency_ms")
