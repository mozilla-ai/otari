"""Spell every price unit as models.dev does.

The v2 catalog a rate is read from says ``token`` and ``image``, so a rate and
the catalog price it came from never disagree on a row once Otari's own
plurals go. ``request`` is Otari's own and follows the same rule.

Revision ID: d1145eb686f8
Revises: a79a97473478
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d1145eb686f8"
down_revision: str | Sequence[str] | None = "a79a97473478"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("model_pricing", "organization_model_pricing")
_RENAMED = (("tokens", "token"), ("requests", "request"), ("images", "image"))
_PRIMARY_KEY = "model_pricing_pkey"
# The scale ``a7c3e5d9b1f4`` gave every rate column, spelled out so this
# revision keeps its meaning when the application's idea of the scale changes.
_RATE_TYPE = sa.Numeric(18, 8)


def _model_pricing_table(*, unit_default: str) -> sa.Table:
    """``model_pricing`` as it stands, for SQLite's batch rebuild.

    Spelled out rather than reflected, as ``a7c3e5d9b1f4`` explains: batch mode
    has mis-reflected this table's composite primary key before.
    """
    return sa.Table(
        "model_pricing",
        sa.MetaData(),
        sa.Column("kind", sa.String(length=16), nullable=False, server_default="model"),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_price_per_million", _RATE_TYPE, nullable=False),
        sa.Column("output_price_per_million", _RATE_TYPE, nullable=False),
        sa.Column("cache_read_price_per_million", _RATE_TYPE, nullable=True),
        sa.Column("cache_write_price_per_million", _RATE_TYPE, nullable=True),
        sa.Column("pricing_tiers", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("unit", sa.String(length=16), nullable=False, server_default=unit_default),
        sa.Column("origin", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("kind", "model_key", "effective_at", name=_PRIMARY_KEY),
    )


def _rename(pairs: tuple[tuple[str, str], ...], *, default: str, was: str) -> None:
    for table in _TABLES:
        for old, new in pairs:
            op.execute(
                sa.text(f"UPDATE {table} SET unit = :new WHERE unit = :old").bindparams(new=new, old=old)  # noqa: S608
            )
    with op.batch_alter_table("model_pricing", copy_from=_model_pricing_table(unit_default=was)) as batch:
        batch.alter_column("unit", existing_type=sa.String(length=16), existing_nullable=False, server_default=default)
    with op.batch_alter_table("organization_model_pricing") as batch:
        batch.alter_column("unit", existing_type=sa.String(length=16), existing_nullable=False, server_default=default)


def upgrade() -> None:
    """Upgrade schema."""
    _rename(_RENAMED, default="token", was="tokens")


def downgrade() -> None:
    """Downgrade schema."""
    _rename(tuple((new, old) for old, new in _RENAMED), default="tokens", was="token")
