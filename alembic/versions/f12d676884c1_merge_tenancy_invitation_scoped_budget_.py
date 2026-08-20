"""Merge heads: invitations, scoped-budget reset alignment, and user credential columns.

Three PRs each branched off ``a3c7e1b9d5f2`` and merged independently, so the
chain forked into three heads with no schema conflict between them (each adds
its own tables/columns, touching nothing the others do). A no-op merge, not a
real migration.

Revision ID: f12d676884c1
Revises: 7ff4e082eb0c, b6e8c2a4d7f1, f2a4c6d8b0e3
Create Date: 2026-08-20 13:27:16.649266

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "f12d676884c1"
down_revision: str | Sequence[str] | None = ("7ff4e082eb0c", "b6e8c2a4d7f1", "f2a4c6d8b0e3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
