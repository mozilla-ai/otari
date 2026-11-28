"""Record why a request's routing plan looked the way it did.

A routed request's row already says which policy produced it and how the attempt
was selected, but not why the candidates that never ran were left out. Those
reasons depend on the caller (its allow-list, its country) rather than on the
policy alone, so they cannot be recomputed after the fact. ``usage_logs`` gains a
nullable ``routing_plan``: null on an unrouted row, and on a routed one whose plan
dropped nothing and whose router did not decline.

Revision ID: c4b1e7f20a15
Revises: ab0af3310ec6
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4b1e7f20a15"
down_revision: str | Sequence[str] | None = "ab0af3310ec6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "usage_logs"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(_TABLE, sa.Column("routing_plan", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column(_TABLE, "routing_plan")
