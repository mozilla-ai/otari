"""Make ``last_send_requested_at`` hold a value on every invitation.

A row written before this revision may hold NULL, from before a send request stamped it.
Backfilled from ``created_at``, since creating an invitation is its first send request.

Revision ID: a6c9fa1da92a
Revises: 8171c59fe077
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a6c9fa1da92a"
down_revision: str | Sequence[str] | None = "8171c59fe077"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EXPIRES_AT_INDEX = "ix_organization_invitation_expires_at"
_INVITER_INDEX = "ix_organization_invitation_invited_by_user_id"
_NEXT_SEND_AT_INDEX = "ix_organization_invitation_next_send_at"
_TOKEN_HASH_INDEX = "ix_organization_invitation_token_hash"
_UNIQUE_NAME = "uq_organization_invitation_organization_email"


def _organization_invitation(*, last_send_requested_at_nullable: bool) -> sa.Table:
    """``organization_invitation`` as it stands on either side of this revision.

    Passed as ``copy_from`` so SQLite's rebuild keeps every index and constraint,
    rather than only what reflection would recover on its own.
    """
    table = sa.Table(
        "organization_invitation",
        sa.MetaData(),
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
        sa.Column("last_send_requested_at", sa.DateTime(timezone=True), nullable=last_send_requested_at_nullable),
        sa.Column("next_send_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("organization_id", "email", name=_UNIQUE_NAME),
    )
    sa.Index(_EXPIRES_AT_INDEX, table.c.expires_at)
    sa.Index(_INVITER_INDEX, table.c.invited_by_user_id)
    sa.Index(_NEXT_SEND_AT_INDEX, table.c.next_send_at)
    sa.Index(_TOKEN_HASH_INDEX, table.c.token_hash, unique=True)
    return table


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE organization_invitation SET last_send_requested_at = created_at"
            " WHERE last_send_requested_at IS NULL"
        )
    )
    with op.batch_alter_table(
        "organization_invitation", copy_from=_organization_invitation(last_send_requested_at_nullable=True)
    ) as batch_op:
        batch_op.alter_column("last_send_requested_at", existing_type=sa.DateTime(timezone=True), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table(
        "organization_invitation", copy_from=_organization_invitation(last_send_requested_at_nullable=False)
    ) as batch_op:
        batch_op.alter_column("last_send_requested_at", existing_type=sa.DateTime(timezone=True), nullable=True)
