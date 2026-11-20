"""Add ``agent_telemetry_user_barriers``: when each user's telemetry was last erased.

A telemetry write locks the row of the user it stores for, and an erasure sets
``erased_at`` and deletes in one transaction, so an export admitted before the
erasure is refused rather than stored after it. The row goes with the user's.

Revision ID: b4d6f8a0c2e5
Revises: e8c2a4f6b1d3
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b4d6f8a0c2e5"
down_revision: str | Sequence[str] | None = "e8c2a4f6b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_telemetry_user_barriers",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("erased_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("agent_telemetry_user_barriers")
