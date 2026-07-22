"""Add persisted upstream pricing snapshots.

Revision ID: d5f7a9b1c3e5
Revises: c4d6e8f0a2b4
Create Date: 2026-07-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5f7a9b1c3e5"
down_revision: str | Sequence[str] | None = "c4d6e8f0a2b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "pricing_snapshots",
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("snapshot", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("source"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("pricing_snapshots")
