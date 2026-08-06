"""add api_keys.reject_user_mismatch

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
    """Add the per-key override of the deployment-wide user-mismatch check."""

    # Deliberately nullable with no server_default: NULL is the "inherit the
    # gateway setting" state, so existing rows land there on ADD COLUMN and keep
    # behaving exactly as they do today.
    op.add_column("api_keys", sa.Column("reject_user_mismatch", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Drop the per-key override."""

    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.drop_column("reject_user_mismatch")
