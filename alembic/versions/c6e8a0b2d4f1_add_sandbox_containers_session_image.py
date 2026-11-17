"""Add sandbox_containers.session_image, the environment a held sandbox was created in.

Revision ID: c6e8a0b2d4f1
Revises: b3d5f7a9c1e2
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c6e8a0b2d4f1"
down_revision: str | Sequence[str] | None = "b3d5f7a9c1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable with no backfill: a sandbox held before this was created in an image nobody recorded.
    op.add_column("sandbox_containers", sa.Column("session_image", sa.String(255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sandbox_containers", "session_image")
