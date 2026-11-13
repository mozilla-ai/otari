"""Add the organization invitation and membership origin tables.

An invitation row does not need a membership row.
A membership origin row is an audit record of who invited a member, when they were invited, and when they joined.

Revision ID: 3ec751593322
Revises: c6e8a0b2d4f1
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3ec751593322"
down_revision: str | Sequence[str] | None = "c6e8a0b2d4f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNIQUE_NAME = "uq_organization_invitation_organization_email"
_EXPIRES_AT_INDEX = "ix_organization_invitation_expires_at"
_INVITER_INDEX = "ix_organization_invitation_invited_by_user_id"
_TOKEN_HASH_INDEX = "ix_organization_invitation_token_hash"
_MEMBERSHIP_INVITER_INDEX = "ix_organization_membership_origin_invited_by_user_id"


def upgrade() -> None:
    op.create_table(
        "organization_invitation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("invited_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("workspace_assignments", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("declined_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("send_count", sa.Integer(), nullable=False),
        sa.Column("last_send_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("organization_id", "email", name=_UNIQUE_NAME),
    )
    op.create_index(op.f(_EXPIRES_AT_INDEX), "organization_invitation", ["expires_at"])
    op.create_index(op.f(_INVITER_INDEX), "organization_invitation", ["invited_by_user_id"])
    op.create_index(op.f(_TOKEN_HASH_INDEX), "organization_invitation", ["token_hash"], unique=True)

    op.create_table(
        "organization_membership_origin",
        sa.Column("organization_member_id", sa.Uuid(), nullable=False),
        sa.Column("invited_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("invited_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("organization_member_id"),
        sa.ForeignKeyConstraint(["organization_member_id"], ["organization_member.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by_user_id"], ["user.id"], ondelete="SET NULL"),
    )
    op.create_index(op.f(_MEMBERSHIP_INVITER_INDEX), "organization_membership_origin", ["invited_by_user_id"])


def downgrade() -> None:
    op.drop_index(op.f(_MEMBERSHIP_INVITER_INDEX), table_name="organization_membership_origin")
    op.drop_table("organization_membership_origin")
    op.drop_index(op.f(_TOKEN_HASH_INDEX), table_name="organization_invitation")
    op.drop_index(op.f(_INVITER_INDEX), table_name="organization_invitation")
    op.drop_index(op.f(_EXPIRES_AT_INDEX), table_name="organization_invitation")
    op.drop_table("organization_invitation")
