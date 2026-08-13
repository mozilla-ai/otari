"""Add the metric-point columns to agent_telemetry.

Revision ID: a1c3e5f7b9d2
Revises: e8a7c6b5d4f3
Create Date: 2026-08-12 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1c3e5f7b9d2"
down_revision: str | Sequence[str] | None = "e8a7c6b5d4f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Widen agent_telemetry with the outcome-metric columns.

    All nullable with no server_default: an existing behavioral row reads back as
    NULL, which is exactly "not a metric row", so nothing has to be backfilled.
    """
    op.add_column("agent_telemetry", sa.Column("kind", sa.String(), nullable=True))
    op.add_column("agent_telemetry", sa.Column("value", sa.Float(), nullable=True))
    op.add_column("agent_telemetry", sa.Column("temporality", sa.String(), nullable=True))
    op.add_column("agent_telemetry", sa.Column("series_start", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_telemetry", sa.Column("series_key", sa.String(), nullable=True))
    # Serves the read-time cumulative-to-delta derivation, which orders one
    # series' points by time inside the requested window.
    op.create_index("ix_agent_telemetry_series_timestamp", "agent_telemetry", ["series_key", "timestamp"])


def downgrade() -> None:
    """Drop the outcome-metric columns; the table itself predates this migration."""
    op.drop_index("ix_agent_telemetry_series_timestamp", table_name="agent_telemetry")
    with op.batch_alter_table("agent_telemetry") as batch_op:
        batch_op.drop_column("series_key")
        batch_op.drop_column("series_start")
        batch_op.drop_column("temporality")
        batch_op.drop_column("value")
        batch_op.drop_column("kind")
