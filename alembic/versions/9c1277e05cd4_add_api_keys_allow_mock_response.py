"""Add allow_mock_response to api_keys.

Whether a chat completion on the key may carry mock_response, which Otari
answers itself without calling the provider, for load tests. False on every
existing key.

Revision ID: 9c1277e05cd4
Revises: 557fc0f5bfda
Create Date: 2026-10-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c1277e05cd4"
down_revision: str | Sequence[str] | None = "557fc0f5bfda"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "api_keys",
        sa.Column("allow_mock_response", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("api_keys", "allow_mock_response")
