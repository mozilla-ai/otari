"""add file_objects table

Revision ID: c3d5e7f9a1b3
Revises: b2f4c6d8e0a1
Create Date: 2026-06-04 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d5e7f9a1b3"
down_revision: str | Sequence[str] | None = "b2f4c6d8e0a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the file_objects table backing the /v1/files API."""

    op.create_table(
        "file_objects",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("mime_type", sa.String(), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("storage_ref", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_file_objects_user_id", "file_objects", ["user_id"])
    op.create_index("ix_file_objects_created_at", "file_objects", ["created_at"])
    op.create_index("ix_file_objects_deleted_at", "file_objects", ["deleted_at"])


def downgrade() -> None:
    """Drop the file_objects table."""

    op.drop_index("ix_file_objects_deleted_at", table_name="file_objects")
    op.drop_index("ix_file_objects_created_at", table_name="file_objects")
    op.drop_index("ix_file_objects_user_id", table_name="file_objects")
    op.drop_table("file_objects")
