"""Add email_vouched_at to user.

Marks a verification an organization admin vouched for (a first password chosen
through an invitation link the inviter also holds) rather than one the person
proved, so domain auto-join and provider sign-in stop treating it as proof.
Nullable with no backfill: existing invitation claims cannot be told apart from
proven addresses after the fact.

Revision ID: c7e1a4d9b3f2
Revises: a9d3e5f7b1c2
Create Date: 2026-10-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e1a4d9b3f2"
down_revision: str | Sequence[str] | None = "a9d3e5f7b1c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("user", sa.Column("email_vouched_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("user", "email_vouched_at")
