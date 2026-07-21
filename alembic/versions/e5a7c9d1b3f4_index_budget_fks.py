"""Index the budget foreign keys read by the dashboard.

Revision ID: e5a7c9d1b3f4
Revises: d1e4b8c6f0a3
Create Date: 2026-07-21 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5a7c9d1b3f4"
down_revision: str | Sequence[str] | None = "d1e4b8c6f0a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Both columns are foreign keys the dashboard reads on every page load but
    # neither was indexed: the budgets list groups users by users.budget_id to
    # build its rollup, and the reset-log drill-down filters budget_reset_logs by
    # budget_id. Index-only additions, so no data changes and no behavior changes.
    op.create_index("ix_users_budget_id", "users", ["budget_id"])
    op.create_index("ix_budget_reset_logs_budget_id", "budget_reset_logs", ["budget_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_budget_reset_logs_budget_id", table_name="budget_reset_logs")
    op.drop_index("ix_users_budget_id", table_name="users")
