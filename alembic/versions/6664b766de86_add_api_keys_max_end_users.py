"""Add max_end_users to api_keys.

The most live end users a service key may have on the budgets it assigns; a
request that would create one more is refused. NULL, every existing key, is no
cap.

Revision ID: 6664b766de86
Revises: 557fc0f5bfda
Create Date: 2026-10-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6664b766de86"
down_revision: str | Sequence[str] | None = "557fc0f5bfda"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("api_keys", sa.Column("max_end_users", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("api_keys", "max_end_users")
