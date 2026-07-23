"""Add dashboard_sessions table for cookie-based dashboard sign-in.

Revision ID: a9c1e3b5d7f9
Revises: 7d9e1f3a5b7c
Create Date: 2026-07-23 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a9c1e3b5d7f9"
down_revision: str | Sequence[str] | None = "7d9e1f3a5b7c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "dashboard_sessions",
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("token_hash"),
    )
    op.create_index("ix_dashboard_sessions_expires_at", "dashboard_sessions", ["expires_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_dashboard_sessions_expires_at", table_name="dashboard_sessions")
    op.drop_table("dashboard_sessions")
