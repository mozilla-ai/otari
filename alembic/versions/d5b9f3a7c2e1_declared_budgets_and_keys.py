"""Mark the budgets and API keys that config.yml declares.

``budgets.origin`` is ``config`` while config.yml declares the budget, the
vocabulary ``model_pricing.origin`` already uses. ``api_keys.config_name`` is the
name config.yml declares a key under: every start finds the key by it, and its
unique index is what keeps replicas starting at once from inserting two.

Both are nullable with no backfill, because nothing was declared before this.

Revision ID: d5b9f3a7c2e1
Revises: c7e1a4d9b3f2
Create Date: 2026-10-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5b9f3a7c2e1"
down_revision: str | Sequence[str] | None = "c7e1a4d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONFIG_NAME_INDEX = "uq_api_keys_config_name"


def upgrade() -> None:
    """Add the two markers."""
    op.add_column("budgets", sa.Column("origin", sa.String(length=16), nullable=True))
    op.add_column("api_keys", sa.Column("config_name", sa.String(), nullable=True))
    op.create_index(_CONFIG_NAME_INDEX, "api_keys", ["config_name"], unique=True)


def downgrade() -> None:
    """Drop the two markers. The rows they marked stay, as ordinary budgets and keys."""
    op.drop_index(_CONFIG_NAME_INDEX, table_name="api_keys")
    op.drop_column("api_keys", "config_name")
    op.drop_column("budgets", "origin")
