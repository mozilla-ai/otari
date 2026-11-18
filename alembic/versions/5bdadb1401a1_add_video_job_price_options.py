"""Add the options of a video job that a price's rules may name.

A video job is billed when it completes, after the request that started it is
gone, so the job keeps the resolution, aspect ratio, reference video and sound
track the request asked for. A job started before this revision has none of
them, and is billed as a request that named none of them.

Revision ID: 5bdadb1401a1
Revises: a7d3c9e1f5b2
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5bdadb1401a1"
down_revision: str | Sequence[str] | None = "a7d3c9e1f5b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("video_jobs", sa.Column("resolution", sa.String(length=64), nullable=True))
    op.add_column("video_jobs", sa.Column("aspect_ratio", sa.String(length=64), nullable=True))
    op.add_column("video_jobs", sa.Column("input_video", sa.Boolean(), nullable=True))
    op.add_column("video_jobs", sa.Column("generate_audio", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("video_jobs", "generate_audio")
    op.drop_column("video_jobs", "input_video")
    op.drop_column("video_jobs", "aspect_ratio")
    op.drop_column("video_jobs", "resolution")
