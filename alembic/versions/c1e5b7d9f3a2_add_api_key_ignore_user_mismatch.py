"""add api_keys.ignore_user_mismatch

Revision ID: c1e5b7d9f3a2
Revises: b5d7f9a1c3e6
Create Date: 2026-08-05 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1e5b7d9f3a2"
down_revision: str | Sequence[str] | None = "b5d7f9a1c3e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the per-key opt-out from the user-mismatch check."""

    # Lands nullable with a server_default so existing rows backfill on ADD COLUMN
    # (both SQLite and Postgres), then is tightened to NOT NULL below.
    op.add_column(
        "api_keys",
        sa.Column("ignore_user_mismatch", sa.Boolean(), nullable=True, server_default=sa.false()),
    )

    conn = op.get_bind()
    conn.execute(
        sa.update(sa.table("api_keys", sa.column("ignore_user_mismatch", sa.Boolean())))
        .where(sa.column("ignore_user_mismatch").is_(None))
        .values(ignore_user_mismatch=False)
    )

    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.alter_column("ignore_user_mismatch", existing_type=sa.Boolean(), nullable=False)


def downgrade() -> None:
    """Drop the per-key opt-out."""

    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.drop_column("ignore_user_mismatch")
