"""Add the vendors a hosted provider does not serve.

``hosted_provider_disabled_vendors`` holds one row per hosted provider and
vendor switched off, so dispatch refuses every model of that vendor on that
provider, including ones a later refresh offers.

Revision ID: b7e2c9d4f1a6
Revises: e5b9d2f7a4c1
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e2c9d4f1a6"
down_revision: str | Sequence[str] | None = "e5b9d2f7a4c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "hosted_provider_disabled_vendors",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("vendor", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "vendor", name="uq_hosted_provider_disabled_vendors_provider_vendor"),
    )


def downgrade() -> None:
    op.drop_table("hosted_provider_disabled_vendors")
