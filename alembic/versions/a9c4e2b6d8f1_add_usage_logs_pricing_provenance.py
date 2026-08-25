"""Add pricing provenance to usage_logs.

Records why a row's ``cost`` is the amount it is: which price list settled it,
which entry in that list was applied, from when that rate was in force, which
revision of the list it came from, and when the pricing happened.

The platform's ``gateway_usage_settlement`` carries this and ``usage_logs``
cannot re-derive it, so without these columns the hosted-usage backfill
(mozilla-ai/otari-ai#1798) drops the provenance of every row it writes and the
truth table cannot answer why an amount is what it is. That backfill is
insert-only, so a row it has already written is never revisited: these columns
have to exist before it runs, not after.

``calculated_at`` is deliberately separate from ``timestamp``. One is when the
amount was priced, the other when the request ran, and usage settled or repriced
later moves them apart.

All five are nullable with no backfill, and the lengths mirror the platform's
settlement columns so a value copied across always fits. The gateway's own
settlement path records no provenance, so null reads correctly as "not
recorded"; inventing a source for a row nobody priced that way would be a lie.

Revision ID: a9c4e2b6d8f1
Revises: d8b3f1c6a4e9
Create Date: 2026-08-25 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a9c4e2b6d8f1"
down_revision: str | Sequence[str] | None = "d8b3f1c6a4e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("usage_logs", sa.Column("pricing_source", sa.String(length=32), nullable=True))
    op.add_column("usage_logs", sa.Column("pricing_reference", sa.String(length=511), nullable=True))
    op.add_column("usage_logs", sa.Column("pricing_effective_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("usage_logs", sa.Column("pricing_version", sa.String(length=255), nullable=True))
    op.add_column("usage_logs", sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "calculated_at")
    op.drop_column("usage_logs", "pricing_version")
    op.drop_column("usage_logs", "pricing_effective_at")
    op.drop_column("usage_logs", "pricing_reference")
    op.drop_column("usage_logs", "pricing_source")
