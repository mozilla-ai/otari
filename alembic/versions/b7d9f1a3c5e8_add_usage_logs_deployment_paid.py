"""Add deployment_paid to usage_logs.

Records whether the deployment paid for the credential a row was served on, so
usage can be reconciled against the charges ``BillingPort`` made. Existing rows
backfill to false through the server default, which PostgreSQL applies without
rewriting the table.

Revision ID: b7d9f1a3c5e8
Revises: a3c5e7f9b1d2
Create Date: 2026-10-15 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d9f1a3c5e8"
down_revision: str | Sequence[str] | None = "a3c5e7f9b1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "usage_logs",
        sa.Column("deployment_paid", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "deployment_paid")
