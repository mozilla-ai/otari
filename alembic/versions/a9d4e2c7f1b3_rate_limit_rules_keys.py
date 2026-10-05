"""Let a stored rate-limit rule be narrowed to API keys.

Adds ``keys``, the API key ids a rule applies to; null covers every key.

Revision ID: a9d4e2c7f1b3
Revises: c6e2a9f4d7b1
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9d4e2c7f1b3"
down_revision: str | Sequence[str] | None = "c6e2a9f4d7b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("rate_limit_rules", sa.Column("keys", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("rate_limit_rules") as batch:
        batch.drop_column("keys")
