"""Let a hosted provider serve a model its roster does not list.

A deployment provider stored through the dashboard served any model a request
named, as a ``config.yml`` instance does. ``serves_unlisted_models`` gives a
hosted provider the same reach, so one moved from there keeps serving what it
served; a model its roster lists but switches off is still refused.

Revision ID: d1f3a5c7e9b2
Revises: c4e6a8b0d2f4
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d1f3a5c7e9b2"
down_revision: str | Sequence[str] | None = "c4e6a8b0d2f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "hosted_providers",
        sa.Column("serves_unlisted_models", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("hosted_providers", "serves_unlisted_models")
