"""Let a hosted provider link its terms of service and privacy policy.

The public Providers pages show both. Null for every existing row: nothing has
been linked yet.

Revision ID: a9d3f5b7c1e2
Revises: c8e1f4a7b2d9
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9d3f5b7c1e2"
down_revision: str | Sequence[str] | None = "c8e1f4a7b2d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("hosted_providers", sa.Column("terms_url", sa.String(length=1024), nullable=True))
    op.add_column("hosted_providers", sa.Column("privacy_url", sa.String(length=1024), nullable=True))


def downgrade() -> None:
    op.drop_column("hosted_providers", "privacy_url")
    op.drop_column("hosted_providers", "terms_url")
