"""Add declared_digest to budgets and api_keys.

A digest of the config.yml entry last written over a declared budget or key, so
a start writes an entry back only when config.yml changed it, and a change made
through the API (an emergency shut-off) survives a restart. NULL until a start
applies the entry; the first start after this migration applies every entry
once, as every start did before.

Revision ID: cc92a10774d8
Revises: c7e1a3f5b9d2
Create Date: 2026-10-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cc92a10774d8"
down_revision: str | Sequence[str] | None = "c7e1a3f5b9d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("budgets", sa.Column("declared_digest", sa.String(64), nullable=True))
    op.add_column("api_keys", sa.Column("declared_digest", sa.String(64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("api_keys", "declared_digest")
    op.drop_column("budgets", "declared_digest")
