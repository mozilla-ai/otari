"""Add a video job's resolution and aspect ratio, which a price's rules may name.

A video job is billed when it completes, after the request that started it is
gone, so the job keeps the resolution and aspect ratio the request asked for.
A job started before this revision has neither, and is billed at the price
that names no resolution or aspect ratio.

Revision ID: 5bdadb1401a1
Revises: 9e4b7a2c6d1f
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5bdadb1401a1"
down_revision: str | Sequence[str] | None = "9e4b7a2c6d1f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("video_jobs", sa.Column("resolution", sa.String(length=64), nullable=True))
    op.add_column("video_jobs", sa.Column("aspect_ratio", sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("video_jobs", "aspect_ratio")
    op.drop_column("video_jobs", "resolution")
