"""Add the operator's email blocklist.

``blocked_email`` holds one row per blocked address or domain. ``pattern`` is
unique, so an entry is blocked once.

Revision ID: 07802ec50d76
Revises: 90b7e153c41f
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "07802ec50d76"
down_revision: str | Sequence[str] | None = "90b7e153c41f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "blocked_email",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pattern", sa.String(length=255), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pattern"),
    )
    op.create_index("ix_blocked_email_created_by_user_id", "blocked_email", ["created_by_user_id"])


def downgrade() -> None:
    op.drop_index("ix_blocked_email_created_by_user_id", table_name="blocked_email")
    op.drop_table("blocked_email")
