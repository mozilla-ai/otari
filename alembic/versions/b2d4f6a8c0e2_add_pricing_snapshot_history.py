"""Keep every accepted genai-prices snapshot, not only the current one.

``pricing_snapshots`` holds one active row per source and one pending row, and
an accept overwrites the active one. Settlement records the snapshot version a
charge was priced under, so the catalog should be able to answer "what did we
think this cost on that date"; it cannot while the previous snapshot is gone.
Each accept now also appends here. Rows are never updated; a downgrade drops
the table and loses only history.

Revision ID: b2d4f6a8c0e2
Revises: c7e9a1b3d5f7
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2d4f6a8c0e2"
down_revision: str | Sequence[str] | None = "c7e9a1b3d5f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "pricing_snapshot_history",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_by", sa.String(length=32), nullable=False),
        sa.Column("model_count", sa.Integer(), nullable=False),
        sa.Column("snapshot", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pricing_snapshot_history_source_accepted_at",
        "pricing_snapshot_history",
        ["source", "accepted_at"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_pricing_snapshot_history_source_accepted_at", table_name="pricing_snapshot_history")
    op.drop_table("pricing_snapshot_history")
