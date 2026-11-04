"""Add allowed_tools to api_keys and users.

A search tool's access list, apart from the model one. Nullable with no
default: a NULL key or user keeps reading its tools off ``allowed_models``, the
deprecated way, so nothing changes on upgrade.

Revision ID: e8b2d4f6a0c3
Revises: 5ab5a1699e8b
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8b2d4f6a0c3"
down_revision: str | Sequence[str] | None = "5ab5a1699e8b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("allowed_tools", sa.JSON(), nullable=True))
    op.add_column("users", sa.Column("allowed_tools", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "allowed_tools")
    op.drop_column("api_keys", "allowed_tools")
