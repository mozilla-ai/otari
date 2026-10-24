"""Add when each user's traces were last erased, which spans taken in before it cannot outlive.

An erasure records the time and deletes in one transaction, and a write locks
the row of each user it stores for, so no replica stores spans of an erased
user it was still holding.

Revision ID: a536c003bcf5
Revises: f2b4d6e8a0c3
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a536c003bcf5"
down_revision: str | Sequence[str] | None = "f2b4d6e8a0c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trace_user_barriers",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("erased_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("trace_user_barriers")
