"""Add stored_prompts table.

Revision ID: e21e278b55b5
Revises: 62256ffefe75
Create Date: 2026-11-16 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e21e278b55b5"
down_revision: str | Sequence[str] | None = "62256ffefe75"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "stored_prompts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_stored_prompts_name", "stored_prompts", ["name"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_stored_prompts_name", table_name="stored_prompts")
    op.drop_table("stored_prompts")
