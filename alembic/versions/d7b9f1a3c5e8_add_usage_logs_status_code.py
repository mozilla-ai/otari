"""Add usage_logs status_code column.

Revision ID: d7b9f1a3c5e8
Revises: c3f7a9d1e5b8
Create Date: 2026-07-31 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d7b9f1a3c5e8"
down_revision: str | Sequence[str] | None = "c3f7a9d1e5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: historical rows predate the column, a successful request has no
    # failure to classify, and some failures carry no HTTP status at all.
    #
    # Deliberately unindexed. Filtering on a code implies status='error' (see
    # ``_usage_filters``) and every aggregate over it is range-bounded, so the
    # existing (status, timestamp) index already narrows to the error rows in the
    # window before the code is applied. usage_logs is append-heavy on the request
    # path, so a further (status, status_code) index would tax every write to save
    # a scan over one window's failures; add it if error volume ever makes that
    # scan the bottleneck.
    op.add_column("usage_logs", sa.Column("status_code", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "status_code")
