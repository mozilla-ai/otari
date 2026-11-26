"""Replace user.email_vouched_at with user.email_proven_at.

The old column was a negative flag: set meant the address was not proved by its
holder. Every reader of the positive fact had to pair it with ``email_verified_at``
and subtract. The new column records the fact directly, so a proved address is one
column read.

Backfilled from the pair, so no row changes meaning: a row carrying a verification
that nothing contradicted is proved, and one carrying the old flag is not.

Revision ID: e2b7c4d9f1a3
Revises: b4d6f8a0c2e5
Create Date: 2026-11-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e2b7c4d9f1a3"
down_revision: str | Sequence[str] | None = "b4d6f8a0c2e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("user", sa.Column("email_proven_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(
        sa.text(
            'UPDATE "user" SET email_proven_at = email_verified_at '
            "WHERE email_verified_at IS NOT NULL AND email_vouched_at IS NULL"
        )
    )
    op.drop_column("user", "email_vouched_at")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column("user", sa.Column("email_vouched_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(
        sa.text(
            'UPDATE "user" SET email_vouched_at = email_verified_at '
            "WHERE email_verified_at IS NOT NULL AND email_proven_at IS NULL"
        )
    )
    op.drop_column("user", "email_proven_at")
