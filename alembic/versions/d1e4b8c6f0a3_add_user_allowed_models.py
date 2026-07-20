"""Add allowed_models to users.

Revision ID: d1e4b8c6f0a3
Revises: c9f3a1b5d7e2
Create Date: 2026-07-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d1e4b8c6f0a3"
down_revision: str | Sequence[str] | None = "c9f3a1b5d7e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no server_default: the per-user default model access-list that a
    # user's keys inherit. Existing users stay NULL (unrestricted), so no user
    # changes behavior on upgrade. sa.JSON (not JSONB) for SQLite/Postgres parity,
    # matching api_keys.allowed_models and the metadata column.
    op.add_column("users", sa.Column("allowed_models", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "allowed_models")
