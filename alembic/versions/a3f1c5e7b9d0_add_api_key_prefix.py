"""Add key_prefix to api_keys.

Revision ID: a3f1c5e7b9d0
Revises: b4d6f8a0c2e4
Create Date: 2026-07-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3f1c5e7b9d0"
down_revision: str | Sequence[str] | None = "b4d6f8a0c2e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no server_default: a plain column add that needs no table rebuild on
    # SQLite or Postgres. Existing keys stay NULL (the plaintext prefix cannot be
    # recovered from the stored hash); new keys get a prefix at mint time.
    op.add_column("api_keys", sa.Column("key_prefix", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("api_keys", "key_prefix")
