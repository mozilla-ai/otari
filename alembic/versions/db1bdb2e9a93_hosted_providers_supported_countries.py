"""Let a hosted provider serve only some countries.

``supported_countries`` holds the ISO country codes whose requests the
provider serves; null serves every country.

Revision ID: db1bdb2e9a93
Revises: bea0cc9abb34
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "db1bdb2e9a93"
down_revision: str | Sequence[str] | None = "bea0cc9abb34"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("hosted_providers", sa.Column("supported_countries", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("hosted_providers", "supported_countries")
