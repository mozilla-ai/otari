"""Add usage_logs ttft_ms column.

Revision ID: 23fbd409eeac
Revises: f1c4a8e2d6b9
Create Date: 2026-09-11 10:33:56.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "23fbd409eeac"
down_revision: str | Sequence[str] | None = "f1c4a8e2d6b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, same reasoning as latency_ms: historical rows predate the
    # column, non-streaming requests have no first chunk, and a stream that
    # failed before yielding anything never reached one.
    op.add_column("usage_logs", sa.Column("ttft_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "ttft_ms")
