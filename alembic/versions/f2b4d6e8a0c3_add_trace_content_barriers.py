"""Add each workspace's content generation, which content queued before it moved on cannot outlive.

Turning a workspace's content off, or deleting it, moves its generation on, and
a write stores content only when it was captured on the current generation.

Revision ID: f2b4d6e8a0c3
Revises: e1a3c5e7b9d4
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2b4d6e8a0c3"
down_revision: str | Sequence[str] | None = "e1a3c5e7b9d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trace_content_barriers",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("generation", sa.BigInteger(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id"),
    )


def downgrade() -> None:
    op.drop_table("trace_content_barriers")
