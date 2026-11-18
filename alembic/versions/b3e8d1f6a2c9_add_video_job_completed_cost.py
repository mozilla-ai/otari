"""Keep what the provider charged for a completed video job.

A provider can charge for a video without saying how long it is. The job's
record keeps that charge beside the length it keeps, because a provider that
renders inside the start call has no job to read back, and the charge is what
bills a video whose length nobody states. A job completed before this revision
has none, and is billed as before.

Revision ID: b3e8d1f6a2c9
Revises: a2c6e84b1f73
Create Date: 2026-11-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3e8d1f6a2c9"
down_revision: str | Sequence[str] | None = "a2c6e84b1f73"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("video_jobs", sa.Column("completed_cost", sa.Numeric(18, 6), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("video_jobs", "completed_cost")
