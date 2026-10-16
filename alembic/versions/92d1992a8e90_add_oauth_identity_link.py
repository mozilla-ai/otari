"""Add the OAuth identity link table.

A link names an account by the provider's ``(provider, subject)`` pair rather than by an email address.
No existing row records a subject, so the table starts empty.
An account gains its link on its next OAuth sign-in.

Revision ID: 92d1992a8e90
Revises: 12d51fa2acc0
Create Date: 2026-10-12 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "92d1992a8e90"
down_revision: str | Sequence[str] | None = "12d51fa2acc0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "oauth_identity_link",
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("email_at_link", sa.String(length=255), nullable=True),
        sa.Column("email_assertion_at_link", sa.String(length=16), nullable=False),
        sa.Column("linked_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("provider", "subject"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "provider", name="uq_oauth_identity_link_user_provider"),
    )


def downgrade() -> None:
    op.drop_table("oauth_identity_link")
