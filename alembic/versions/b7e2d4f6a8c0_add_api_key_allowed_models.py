"""Add allowed_models to api_keys.

Revision ID: b7e2d4f6a8c0
Revises: a3f1c5e7b9d0
Create Date: 2026-07-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7e2d4f6a8c0"
down_revision: str | Sequence[str] | None = "a3f1c5e7b9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no server_default: existing keys stay NULL, which means
    # "unrestricted", so no key changes behavior on upgrade. sa.JSON (not JSONB)
    # for SQLite/Postgres parity, matching the existing metadata column.
    op.add_column("api_keys", sa.Column("allowed_models", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("api_keys", "allowed_models")
