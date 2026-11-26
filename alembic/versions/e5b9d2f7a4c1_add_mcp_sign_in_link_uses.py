"""Add the record of MCP sign-in links that have been used.

``mcp_sign_in_link_uses`` holds the hash of each sign-in link whose Continue started a
sign-in, so the link cannot start another, until the link would have expired anyway.

Revision ID: e5b9d2f7a4c1
Revises: a9d3f5b7c1e2
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5b9d2f7a4c1"
down_revision: str | Sequence[str] | None = "a9d3f5b7c1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mcp_sign_in_link_uses",
        sa.Column("ticket_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("ticket_hash"),
    )
    op.create_index("ix_mcp_sign_in_link_uses_expires_at", "mcp_sign_in_link_uses", ["expires_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_mcp_sign_in_link_uses_expires_at", table_name="mcp_sign_in_link_uses")
    op.drop_table("mcp_sign_in_link_uses")
