"""Add the invitation table.

One row per emailed organization-member invitation, pointing at the
``organization_member`` row created at invite time (``status="invited"``). Not
1:1: a membership can be invited, revoked, and re-invited, and each round mints
a fresh row rather than reusing or deleting the cancelled one (see the
column's comment on ``Invitation`` in ``models/tenancy.py``).

No circular foreign key and no batch rebuild needed here: everything this
table points at (``organization``, ``organization_member``, ``user``) already
exists.

Revision ID: 7ff4e082eb0c
Revises: c8f2a6b4e9d3
Create Date: 2026-08-19 00:00:00.000000

Rebased onto ``c8f2a6b4e9d3`` (not the original ``a3c7e1b9d5f2``): that merge
revision landed on main while this one was in flight, rejoining the two other
branches ``a3c7e1b9d5f2`` had forked into. Chaining here instead of adding a
second merge keeps the graph at one head without a redundant no-op revision.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7ff4e082eb0c"
down_revision: str | Sequence[str] | None = "c8f2a6b4e9d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "invitation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("organization_member_id", sa.Uuid(), nullable=False),
        sa.Column("invited_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("workspace_assignments", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_member_id"], ["organization_member.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_invitation_organization_id"), "invitation", ["organization_id"], unique=False)
    # Not unique: a membership can be invited, revoked, and re-invited, each
    # round minting a fresh row rather than reusing or deleting the cancelled
    # one. See the column's comment in models/tenancy.py.
    op.create_index(
        op.f("ix_invitation_organization_member_id"), "invitation", ["organization_member_id"], unique=False
    )
    op.create_index(op.f("ix_invitation_invited_by_user_id"), "invitation", ["invited_by_user_id"], unique=False)
    op.create_index(op.f("ix_invitation_token_hash"), "invitation", ["token_hash"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_invitation_token_hash"), table_name="invitation")
    op.drop_index(op.f("ix_invitation_invited_by_user_id"), table_name="invitation")
    op.drop_index(op.f("ix_invitation_organization_member_id"), table_name="invitation")
    op.drop_index(op.f("ix_invitation_organization_id"), table_name="invitation")
    op.drop_table("invitation")
