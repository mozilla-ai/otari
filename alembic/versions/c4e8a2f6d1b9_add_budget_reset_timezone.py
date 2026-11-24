"""Let a calendar-aligned budget reset at midnight in a timezone of its own.

``reset_timezone`` names the IANA zone whose midnights ``reset_alignment`` snaps
to. Nullable with no backfill: NULL is UTC, which is where every existing budget
already resets, so the upgrade moves no period.

The check constraint goes on inside ``batch_alter_table``, as in
``b6e8c2a4d7f1``, because SQLite has no ALTER TABLE ADD CONSTRAINT.

Revision ID: c4e8a2f6d1b9
Revises: f6b8d0a2c4e7
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8a2f6d1b9"
down_revision: str | Sequence[str] | None = "f6b8d0a2c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("budgets") as batch_op:
        batch_op.add_column(sa.Column("reset_timezone", sa.String(length=64), nullable=True))
        batch_op.create_check_constraint(
            "ck_budgets_timezone_needs_alignment",
            "reset_timezone IS NULL OR reset_alignment IS NOT NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("budgets") as batch_op:
        batch_op.drop_constraint("ck_budgets_timezone_needs_alignment", type_="check")
        batch_op.drop_column("reset_timezone")
