"""Add safety_identifier to usage_logs.

Stores the end-user identifier each provider call carried, so an identifier a
provider reports in an abuse case can be traced back to a workspace, user and
key with one indexed lookup. The identifier is a one-way digest, and the
caller's tag it was derived from is not stored, so without this column it
cannot be traced at all.

Nullable with no backfill: an existing row's identifier was never recorded.

Revision ID: 7e02300efc98
Revises: e8b2d4f6a0c3
Create Date: 2026-10-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7e02300efc98"
down_revision: str | Sequence[str] | None = "e8b2d4f6a0c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("usage_logs", sa.Column("safety_identifier", sa.String(length=64), nullable=True))
    op.create_index("ix_usage_logs_safety_identifier", "usage_logs", ["safety_identifier"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_usage_logs_safety_identifier", table_name="usage_logs")
    op.drop_column("usage_logs", "safety_identifier")
