"""add tag scoped budget fields

Revision ID: f1a2b3c4d5e6
Revises: e0f1a2b3c4d5
Create Date: 2026-05-27 15:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e0f1a2b3c4d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("budgets") as batch_op:
        batch_op.add_column(sa.Column("scope_type", sa.String(), nullable=False, server_default="entity"))
        batch_op.add_column(sa.Column("match_tags", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("spend", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("budget_started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("next_budget_reset_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("blocked", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.create_index(op.f("ix_budgets_scope_type"), ["scope_type"], unique=False)
        batch_op.create_index(op.f("ix_budgets_is_active"), ["is_active"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("budgets") as batch_op:
        batch_op.drop_index(op.f("ix_budgets_is_active"))
        batch_op.drop_index(op.f("ix_budgets_scope_type"))
        batch_op.drop_column("is_active")
        batch_op.drop_column("blocked")
        batch_op.drop_column("next_budget_reset_at")
        batch_op.drop_column("budget_started_at")
        batch_op.drop_column("spend")
        batch_op.drop_column("match_tags")
        batch_op.drop_column("scope_type")
