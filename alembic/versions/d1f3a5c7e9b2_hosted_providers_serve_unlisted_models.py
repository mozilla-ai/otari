"""Let a hosted provider serve a model its roster does not list.

``serves_unlisted_models`` serves any model a request names, as a
``config.yml`` instance does; a model the roster switches off is still refused.

Revision ID: d1f3a5c7e9b2
Revises: d8b4e2f6a1c3
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d1f3a5c7e9b2"
down_revision: str | Sequence[str] | None = "d8b4e2f6a1c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "hosted_providers",
        sa.Column("serves_unlisted_models", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("hosted_providers", "serves_unlisted_models")
