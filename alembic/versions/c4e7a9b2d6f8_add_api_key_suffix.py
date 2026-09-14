"""Add key_suffix to api_keys.

Revision ID: c4e7a9b2d6f8
Revises: a1d4f7c2e8b3
Create Date: 2026-09-14 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e7a9b2d6f8"
down_revision: str | Sequence[str] | None = "a1d4f7c2e8b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no server_default: a plain column add that needs no table rebuild on
    # SQLite or Postgres. Existing keys stay NULL and can never be back-filled (the
    # plaintext suffix cannot be recovered from the stored hash), so they display
    # prefix-only for the life of the row; new keys get a suffix at mint time.
    op.add_column("api_keys", sa.Column("key_suffix", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("api_keys", "key_suffix")
