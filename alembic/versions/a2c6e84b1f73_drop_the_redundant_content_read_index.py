"""Drop ``ix_trace_content_reads_share_id``.

``ix_trace_content_reads_share_read_at`` is on ``(share_id, read_at)``, whose
leading column is the same, so it serves every query and foreign-key check
keyed on ``share_id``. A row is written on each content read through a share,
and one index to maintain there is enough.

Revision ID: a2c6e84b1f73
Revises: b508d2f6ac31
Create Date: 2026-10-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a2c6e84b1f73"
down_revision: str | Sequence[str] | None = "b508d2f6ac31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "trace_content_reads"
_INDEX = "ix_trace_content_reads_share_id"


def upgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE)


def downgrade() -> None:
    op.create_index(_INDEX, _TABLE, ["share_id"])
