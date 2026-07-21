"""Add name to budgets.

Revision ID: c9f3a1b5d7e2
Revises: b7e2d4f6a8c0
Create Date: 2026-07-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9f3a1b5d7e2"
down_revision: str | Sequence[str] | None = "b7e2d4f6a8c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no server_default: an admin-facing label for a budget. Existing
    # budgets stay NULL and are shown by their id, so no behavior changes on
    # upgrade.
    op.add_column("budgets", sa.Column("name", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("budgets", "name")
