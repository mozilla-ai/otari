"""Drop the stored genai-prices snapshots; the pricing snapshot is models.dev now.

The accepted, pending and poll-claim rows and the accepted-snapshot history were
all genai-prices payloads, which the models.dev price index cannot read. They are
deleted rather than converted: the next refresh stores a models.dev snapshot, and
until one is accepted the gateway prices from the bundled models.dev snapshot.
The downgrade restores nothing, so the genai-prices history is lost.

Revision ID: c4e8a2d6f0b3
Revises: b5d1f3a7c9e2
Create Date: 2026-10-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8a2d6f0b3"
down_revision: str | Sequence[str] | None = "b5d1f3a7c9e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SNAPSHOT_SOURCES = ("genai-prices", "genai-prices-pending", "genai-prices-poll-claim")


def upgrade() -> None:
    """Upgrade schema."""
    snapshots = sa.table("pricing_snapshots", sa.column("source", sa.String))
    history = sa.table("pricing_snapshot_history", sa.column("source", sa.String))
    op.execute(sa.delete(snapshots).where(snapshots.c.source.in_(_SNAPSHOT_SOURCES)))
    op.execute(sa.delete(history).where(history.c.source == "genai-prices"))


def downgrade() -> None:
    """Downgrade schema: the deleted genai-prices rows are not restored."""
