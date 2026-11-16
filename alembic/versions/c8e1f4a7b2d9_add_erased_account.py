"""Add the record of accounts deleted at their own request.

``erased_account`` holds the old id of each such account and when it went, and
nothing that names the person.

Revision ID: c8e1f4a7b2d9
Revises: a6c9fa1da92a
Create Date: 2026-11-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8e1f4a7b2d9"
down_revision: str | Sequence[str] | None = "a6c9fa1da92a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "erased_account",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("erased_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("erased_account")
