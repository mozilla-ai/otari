"""Add sandbox_credentials.

The key a hosted sandbox provider's sandboxes are leased with, stored through
the dashboard so rotating it needs no redeploy. One row per provider.

Revision ID: c3e5a7b9d1f2
Revises: c4f1a9d27e6b
Create Date: 2026-10-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3e5a7b9d1f2"
down_revision: str | Sequence[str] | None = "c4f1a9d27e6b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "sandbox_credentials",
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("encrypted_api_key", sa.String(), nullable=False),
        sa.Column("last4", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("provider"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("sandbox_credentials")
