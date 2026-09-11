"""Record what a price is per, and which writer set it.

``input_price_per_million`` has been read four ways: per million tokens for a
model, per million requests for a gateway-run tool or a moderation call, and per
image for image generation. Nothing on the row said which, so every reader that
renders a rate had to know the convention by the key's spelling, and a generic
table put a tool's ``10000.0`` under a column labeled "$ / 1M". ``unit`` is that
convention made a column. Existing rows default to ``tokens``; the one family
the key spelling identifies for certain, the reserved ``otari:`` prefix of a
gateway-run tool, is backfilled to ``requests``.

``origin`` says which path wrote a deployment price: the config file, the API, or
a migration. The three disagree about what a repeat write means, and without a
record of who wrote a row the config loader cannot tell a stale copy of its own
entry from a rate an operator set deliberately. Nullable, with no backfill: a
row written before this column cannot say where it came from, and inventing an
origin would be worse than admitting the gap. The organization override table
takes both columns too, so the two price lists stay one shape.

Revision ID: c7e9a1b3d5f7
Revises: f1c4a8e2d6b9
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e9a1b3d5f7"
down_revision: str | Sequence[str] | None = "f1c4a8e2d6b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("model_pricing", "organization_model_pricing")


def upgrade() -> None:
    """Upgrade schema."""
    for table in _TABLES:
        op.add_column(table, sa.Column("unit", sa.String(length=16), nullable=False, server_default="tokens"))
        op.add_column(table, sa.Column("origin", sa.String(length=16), nullable=True))
    op.execute(sa.text("UPDATE model_pricing SET unit = 'requests' WHERE model_key LIKE 'otari:%'"))


def downgrade() -> None:
    """Downgrade schema."""
    for table in reversed(_TABLES):
        op.drop_column(table, "origin")
        op.drop_column(table, "unit")
