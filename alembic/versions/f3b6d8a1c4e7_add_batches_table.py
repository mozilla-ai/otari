"""Add batches table for idempotent accounting and strict ownership.

Revision ID: f3b6d8a1c4e7
Revises: e2f5a7c9b1d4
Create Date: 2026-07-21 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3b6d8a1c4e7"
down_revision: str | Sequence[str] | None = "e2f5a7c9b1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "batches",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("results_accounted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_batches_user_id", "batches", ["user_id"])
    op.create_index("ix_batches_api_key_id", "batches", ["api_key_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_batches_api_key_id", table_name="batches")
    op.drop_index("ix_batches_user_id", table_name="batches")
    op.drop_table("batches")
