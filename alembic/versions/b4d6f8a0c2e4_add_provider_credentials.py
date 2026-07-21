"""Add provider_credentials table.

Revision ID: b4d6f8a0c2e4
Revises: a2c4e6f8b0d1
Create Date: 2026-07-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4d6f8a0c2e4"
down_revision: str | Sequence[str] | None = "a2c4e6f8b0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "provider_credentials",
        sa.Column("instance", sa.String(), nullable=False),
        sa.Column("provider_type", sa.String(), nullable=True),
        sa.Column("api_base", sa.String(), nullable=True),
        sa.Column("encrypted_api_key", sa.String(), nullable=True),
        sa.Column("last4", sa.String(), nullable=True),
        sa.Column("client_args", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("instance"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("provider_credentials")
