"""Index ``trace_user_barriers.erased_at``, which the hourly retention sweep selects on.

A write makes a barrier row for every user whose spans it stores, so the table
holds one row per user who has written a trace since the last sweep. The sweep
locks every row it selects and those users' writers wait on it, so what the
sweep reads has to be an index rather than the table.

Revision ID: f1a93c7e5d28
Revises: e7b2c9d41a60
Create Date: 2026-10-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f1a93c7e5d28"
down_revision: str | Sequence[str] | None = "e7b2c9d41a60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "trace_user_barriers"
_INDEX = "ix_trace_user_barriers_erased_at"


def upgrade() -> None:
    op.create_index(_INDEX, _TABLE, ["erased_at"])


def downgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE)
