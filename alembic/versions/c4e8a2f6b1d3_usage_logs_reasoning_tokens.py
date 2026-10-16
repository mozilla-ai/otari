"""Record reasoning tokens on each usage row.

A subset of completion_tokens, kept for attribution and never priced on its own.
Nullable, so rows written before this revision read as unknown rather than zero.

Revision ID: c4e8a2f6b1d3
Revises: e3b7d1a5c9f2
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8a2f6b1d3"
down_revision: str | Sequence[str] | None = "e3b7d1a5c9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("usage_logs", sa.Column("reasoning_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("usage_logs", "reasoning_tokens")
