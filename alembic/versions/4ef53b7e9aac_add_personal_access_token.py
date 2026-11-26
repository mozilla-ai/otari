"""Add the personal access token table.

A person's own credential for reading the management API as them, from an
agent or a script. Only the token's SHA-256 hash is stored, unique so a
presented token resolves on one index lookup.

Revision ID: 4ef53b7e9aac
Revises: 7c4b45a55970
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4ef53b7e9aac"
down_revision: str | Sequence[str] | None = "7c4b45a55970"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "personal_access_token",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("token_suffix", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("organization_id", sa.Uuid(), nullable=True),
        sa.Column("client_id", sa.String(length=128), nullable=True),
        sa.Column("access_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("refresh_token_hash", sa.String(length=64), nullable=True),
        sa.Column("previous_refresh_token_hash", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token_hash"),
        sa.UniqueConstraint("refresh_token_hash"),
    )
    op.create_index(op.f("ix_personal_access_token_user_id"), "personal_access_token", ["user_id"])
    op.create_index(op.f("ix_personal_access_token_organization_id"), "personal_access_token", ["organization_id"])
    op.create_index(
        op.f("ix_personal_access_token_previous_refresh_token_hash"),
        "personal_access_token",
        ["previous_refresh_token_hash"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_personal_access_token_previous_refresh_token_hash"), table_name="personal_access_token")
    op.drop_index(op.f("ix_personal_access_token_organization_id"), table_name="personal_access_token")
    op.drop_index(op.f("ix_personal_access_token_user_id"), table_name="personal_access_token")
    op.drop_table("personal_access_token")
