"""Give a budget per-minute limits for each user on it.

Adds ``rpm_limit`` and ``tpm_limit``; null on both, as every existing budget
gets, limits nothing per minute.

Revision ID: d4f7a2b9e6c1
Revises: c2e5a8d1f4b7
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4f7a2b9e6c1"
down_revision: str | Sequence[str] | None = "c2e5a8d1f4b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("budgets", sa.Column("rpm_limit", sa.Integer(), nullable=True))
    op.add_column("budgets", sa.Column("tpm_limit", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("budgets") as batch:
        batch.drop_column("tpm_limit")
        batch.drop_column("rpm_limit")
