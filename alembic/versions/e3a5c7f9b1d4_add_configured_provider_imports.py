"""Record the config.yml providers copied onto hosted providers.

Revision ID: e3a5c7f9b1d4
Revises: d1f3a5c7e9b2
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3a5c7f9b1d4"
down_revision: str | Sequence[str] | None = "d1f3a5c7e9b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "configured_provider_imports",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )


def downgrade() -> None:
    op.drop_table("configured_provider_imports")
