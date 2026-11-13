"""Add the columns that record when an invitation email is due.

``next_send_at`` is when the invitation email is next due, and is NULL when no email is due.
``delivery_attempts`` counts the attempts to deliver the invitation email.

Revision ID: 62256ffefe75
Revises: 3ec751593322
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "62256ffefe75"
down_revision: str | Sequence[str] | None = "3ec751593322"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEXT_SEND_AT_INDEX = "ix_organization_invitation_next_send_at"


def upgrade() -> None:
    op.add_column("organization_invitation", sa.Column("next_send_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "organization_invitation", sa.Column("delivery_attempts", sa.Integer(), server_default="0", nullable=False)
    )
    op.create_index(op.f(_NEXT_SEND_AT_INDEX), "organization_invitation", ["next_send_at"])


def downgrade() -> None:
    op.drop_index(op.f(_NEXT_SEND_AT_INDEX), table_name="organization_invitation")
    op.drop_column("organization_invitation", "delivery_attempts")
    op.drop_column("organization_invitation", "next_send_at")
