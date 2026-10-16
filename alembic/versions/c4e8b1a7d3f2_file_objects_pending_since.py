"""Add ``file_objects.pending_since``, so a row can exist before its bytes do.

A row is set pending when it is reserved and cleared once its bytes land. NULL
is a served row, which every existing row already is, so nothing is backfilled
and adding the column rewrites no rows.

The partial index holds only pending rows, which are what the file sweep reads
when it reclaims an upload that never completed.

Revision ID: c4e8b1a7d3f2
Revises: a9c4e7b2d5f8
Create Date: 2026-09-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8b1a7d3f2"
down_revision: str | Sequence[str] | None = "a9c4e7b2d5f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "file_objects"
_COLUMN = "pending_since"
_INDEX = "ix_file_objects_pending_since"
_PENDING = sa.text("pending_since IS NOT NULL")


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.DateTime(timezone=True), nullable=True))
    op.create_index(_INDEX, _TABLE, [_COLUMN], postgresql_where=_PENDING, sqlite_where=_PENDING)


def downgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE)
    op.drop_column(_TABLE, _COLUMN)
