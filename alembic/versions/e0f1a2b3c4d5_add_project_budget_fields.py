"""add project budget fields

Revision ID: e0f1a2b3c4d5
Revises: d9e0f1a2b3c4
Create Date: 2026-05-27 15:15:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e0f1a2b3c4d5"
down_revision: str | Sequence[str] | None = "d9e0f1a2b3c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(sa.Column("spend", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("budget_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("budget_started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("next_budget_reset_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("blocked", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.create_foreign_key(
            "fk_projects_budget_id_budgets",
            "budgets",
            ["budget_id"],
            ["budget_id"],
        )
        batch_op.create_index(op.f("ix_projects_budget_id"), ["budget_id"], unique=False)

    with op.batch_alter_table("budget_reset_logs") as batch_op:
        batch_op.add_column(sa.Column("project_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_budget_reset_logs_project_id_projects",
            "projects",
            ["project_id"],
            ["project_id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(op.f("ix_budget_reset_logs_project_id"), ["project_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("budget_reset_logs") as batch_op:
        batch_op.drop_index(op.f("ix_budget_reset_logs_project_id"))
        batch_op.drop_constraint("fk_budget_reset_logs_project_id_projects", type_="foreignkey")
        batch_op.drop_column("project_id")

    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_index(op.f("ix_projects_budget_id"))
        batch_op.drop_constraint("fk_projects_budget_id_budgets", type_="foreignkey")
        batch_op.drop_column("blocked")
        batch_op.drop_column("next_budget_reset_at")
        batch_op.drop_column("budget_started_at")
        batch_op.drop_column("budget_id")
        batch_op.drop_column("spend")
