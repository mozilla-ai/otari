"""Let a hosted provider and an organization provider key declare zero data retention.

``zero_data_retention`` is a declaration the catalog shows beside each offering
and filters on. Off for every existing row: nothing has been declared yet.

Revision ID: e7c2a4f6b8d0
Revises: b3e5f7a9c1d4
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7c2a4f6b8d0"
down_revision: str | Sequence[str] | None = "b3e5f7a9c1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "hosted_providers",
        sa.Column("zero_data_retention", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "org_provider_keys",
        sa.Column("zero_data_retention", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("org_provider_keys", "zero_data_retention")
    op.drop_column("hosted_providers", "zero_data_retention")
