"""Drop the 1-hour cache-write price rate.

No caller ever set a dedicated rate for 1-hour cache writes, so a 1h write
bills at the ordinary ``cache_write_price_per_million``. The 1h token meter on
usage rows stays; only the rate columns and the override table's non-negative
check go.

Revision ID: b5d1f3a7c9e2
Revises: c7e1a4d9b3f2
Create Date: 2026-10-08 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5d1f3a7c9e2"
down_revision: str | Sequence[str] | None = "c7e1a4d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMN = "cache_write_1h_price_per_million"
_CHECK_NAME = "ck_organization_model_pricing_cache_write_1h_non_negative"
_CHECK_SQL = f"{_COLUMN} IS NULL OR {_COLUMN} >= 0"
# The scale ``a7c3e5d9b1f4`` gave every rate column, spelled out so this
# revision keeps its meaning when the application's idea of the scale changes.
_RATE_TYPE = sa.Numeric(18, 8)


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("model_pricing") as batch_op:
        batch_op.drop_column(_COLUMN)
    with op.batch_alter_table("organization_model_pricing") as batch_op:
        batch_op.drop_constraint(_CHECK_NAME, type_="check")
        batch_op.drop_column(_COLUMN)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("model_pricing") as batch_op:
        batch_op.add_column(sa.Column(_COLUMN, _RATE_TYPE, nullable=True))
    with op.batch_alter_table("organization_model_pricing") as batch_op:
        batch_op.add_column(sa.Column(_COLUMN, _RATE_TYPE, nullable=True))
        batch_op.create_check_constraint(_CHECK_NAME, _CHECK_SQL)
