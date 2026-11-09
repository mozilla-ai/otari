"""add usage_logs.stream_cut_by

Records who ended a stream before it completed, ``provider`` or ``client``, so a
stream the provider cut can be listed for a refund or a report. Nullable with no
backfill: null reads as "not cut", and rows written before this column cannot say.

Revision ID: c7d2e9a4b6f1
Revises: 5bdadb1401a1
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7d2e9a4b6f1"
down_revision: str | Sequence[str] | None = "5bdadb1401a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("usage_logs", sa.Column("stream_cut_by", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("usage_logs", "stream_cut_by")
