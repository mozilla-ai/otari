"""Keep the files each saved Playground turn attached.

Adds ``attachments`` to ``playground_message``: the ``file_id``, filename and
size of every file the turn sent, so a resumed transcript draws its chips and
sends the files again. Existing turns attached nothing.

Revision ID: d8f1b3a5c7e9
Revises: c6e2a9f4d7b1
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8f1b3a5c7e9"
down_revision: str | Sequence[str] | None = "c6e2a9f4d7b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``attachments``, empty for every existing turn."""
    op.add_column("playground_message", sa.Column("attachments", sa.JSON(), nullable=False, server_default="[]"))


def downgrade() -> None:
    """Drop ``attachments``; saved turns forget which files they sent."""
    with op.batch_alter_table("playground_message") as batch:
        batch.drop_column("attachments")
