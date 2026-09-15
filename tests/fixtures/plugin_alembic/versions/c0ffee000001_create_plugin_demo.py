"""create plugin_demo

Revision ID: c0ffee000001
Revises:
Create Date: 2026-09-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c0ffee000001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "plugin_demo",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("label", sa.String(length=64), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("plugin_demo")
