"""Record cache and reasoning tokens on each trace span.

Nullable, so spans written before this revision read as unknown rather than zero.

Revision ID: f0f9d05dafc6
Revises: c4e6a8b0d2f4
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f0f9d05dafc6"
down_revision: str | Sequence[str] | None = "c4e6a8b0d2f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def upgrade() -> None:
    for name in _COLUMNS:
        op.add_column("trace_spans", sa.Column(name, sa.Integer(), nullable=True))


def downgrade() -> None:
    for name in reversed(_COLUMNS):
        op.drop_column("trace_spans", name)
