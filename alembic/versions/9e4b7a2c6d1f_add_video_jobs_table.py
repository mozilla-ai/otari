"""Add the video jobs table: who started each video job here, and whether it has been billed.

One row per video generation job a caller starts through the gateway, written
when the provider accepts it. The row anchors ownership, names the provider
instance later calls on the job ask, holds the one-time accounting claim
(``accounted_at``) that bills a completed job once, and remembers what a
completed job reported, for a provider that keeps no job to look up.

Revision ID: 9e4b7a2c6d1f
Revises: d8a3c5e7f9b1
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9e4b7a2c6d1f"
down_revision: str | Sequence[str] | None = "d8a3c5e7f9b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "video_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("seconds", sa.Integer(), nullable=True),
        sa.Column("counts_toward_budget", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accounted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_seconds", sa.Integer(), nullable=True),
        sa.Column("output_url", sa.Text(), nullable=True),
        sa.Column("output_mime_type", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_video_jobs_user_id"), "video_jobs", ["user_id"], unique=False)
    op.create_index(op.f("ix_video_jobs_api_key_id"), "video_jobs", ["api_key_id"], unique=False)
    op.create_index(op.f("ix_video_jobs_workspace_id"), "video_jobs", ["workspace_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_video_jobs_workspace_id"), table_name="video_jobs")
    op.drop_index(op.f("ix_video_jobs_api_key_id"), table_name="video_jobs")
    op.drop_index(op.f("ix_video_jobs_user_id"), table_name="video_jobs")
    op.drop_table("video_jobs")
