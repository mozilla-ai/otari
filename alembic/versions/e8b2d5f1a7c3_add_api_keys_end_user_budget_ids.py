"""Let a service key list the budgets its end users may start on.

Adds ``api_keys.end_user_budget_ids``; null, as every existing key gets, leaves
the key's ``end_user_budget_id`` as the only one.

Revision ID: e8b2d5f1a7c3
Revises: d4f7a2b9e6c1
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8b2d5f1a7c3"
down_revision: str | Sequence[str] | None = "d4f7a2b9e6c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("end_user_budget_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("end_user_budget_ids")
