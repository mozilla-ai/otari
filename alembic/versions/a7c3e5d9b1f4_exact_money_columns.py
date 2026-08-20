"""store rates and settled costs as exact numerics

Revision ID: a7c3e5d9b1f4
Revises: f7a2c4e6b8d1
Create Date: 2026-08-20 10:00:00.000000

The rate columns on ``model_pricing`` and ``organization_model_pricing``, and
``usage_logs.cost``, were binary floating point. They become ``NUMERIC``:
rates at scale 8, costs at scale 6, matching ``gateway.models.money`` and the
rounding ``gateway.core.metered_pricing`` defines. mozilla-ai/otari#661; the
cost column's scale is what mozilla-ai/otari-ai#1751 settled for the accounting
truth.

**No value moves.** PostgreSQL casts ``double precision`` to ``numeric``
through the float's shortest round-trip decimal representation, which is the
decimal the operator typed: ``0.075`` was stored as the float nearest ``0.075``
and comes back as exactly ``0.075``. A rate would only shift if it had been
written with more than 8 significant decimals after the point, which no
published rate has.

**A settled cost does move, and the move is one-way.** ``usage_logs.cost``
keeps six decimals, so an existing amount is rounded to the micro-dollar as it
converts: ``0.1234567`` becomes ``0.123457``, and anything below half a
micro-dollar (a handful of embedding or moderation tokens on a cheap model)
becomes ``0.000000``. The downgrade restores the column type and cannot restore
those digits, so a deployment's historical spend total can shift very slightly
downward on upgrade. That is the scale mozilla-ai/otari-ai#1751 chose for the
accounting truth; it is recorded here because nothing else would tell an
operator why last month's total changed.

**On PostgreSQL this rewrites ``usage_logs``.** ``double precision`` to
``numeric`` is not binary-coercible, so the ALTER copies the whole table under
an ACCESS EXCLUSIVE lock and needs transient space for a second copy. On a
large usage table that is real downtime, and it runs at startup when
``auto_migrate`` is set. Schedule it rather than discovering it.

SQLite gets the same DDL through a table rebuild, though there the change is
declarative: SQLite has no numeric storage class, so the values stay REAL and
exactness at rest needs PostgreSQL. What the rebuild buys is that a SQLite
database migrated to head and one created from the models describe the same
schema.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c3e5d9b1f4"
down_revision: str | Sequence[str] | None = "f7a2c4e6b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RATE_COLUMNS = (
    ("input_price_per_million", False),
    ("output_price_per_million", False),
    ("cache_read_price_per_million", True),
    ("cache_write_price_per_million", True),
    ("cache_write_1h_price_per_million", True),
)
_PRICING_TABLES = ("model_pricing", "organization_model_pricing")

# Spelled out rather than imported from ``gateway.models.money``: a migration
# describes the schema at one point in the chain, so it must not change meaning
# when the application's idea of the scale does.
_RATE_TYPE = sa.Numeric(18, 8)
_COST_TYPE = sa.Numeric(18, 6)
_FLOAT_TYPE = sa.Float()


def _rate_columns(numeric: bool) -> list[sa.Column[object]]:
    column_type: sa.types.TypeEngine[object] = _RATE_TYPE if numeric else _FLOAT_TYPE
    return [sa.Column(name, column_type, nullable=nullable) for name, nullable in _RATE_COLUMNS]


def _model_pricing_table(numeric: bool) -> sa.Table:
    """``model_pricing`` as it stands, for SQLite's batch rebuild.

    Spelled out rather than reflected because the table carries a composite
    primary key that batch mode has mis-reflected here before (the
    ``effective_at`` revision left a warning about exactly that), and a rebuild
    that drops half a primary key is not something a type change should risk.
    """
    return sa.Table(
        "model_pricing",
        sa.MetaData(),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        *_rate_columns(numeric),
        sa.Column("pricing_tiers", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_key", "effective_at", name="model_pricing_pkey"),
    )


def _organization_model_pricing_table(numeric: bool) -> sa.Table:
    """``organization_model_pricing`` as it stands, for SQLite's batch rebuild.

    SQLite cannot reflect a CHECK constraint, so a rebuild driven by reflection
    would silently drop the five that keep a negative rate out of the table.
    ``copy_from`` is what carries them across.
    """
    return sa.Table(
        "organization_model_pricing",
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        *_rate_columns(numeric),
        sa.Column("pricing_tiers", sa.JSON(), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_organization_model_pricing_period_ordered",
        ),
        sa.CheckConstraint(
            "input_price_per_million >= 0",
            name="ck_organization_model_pricing_input_non_negative",
        ),
        sa.CheckConstraint(
            "output_price_per_million >= 0",
            name="ck_organization_model_pricing_output_non_negative",
        ),
        sa.CheckConstraint(
            "cache_read_price_per_million IS NULL OR cache_read_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_read_non_negative",
        ),
        sa.CheckConstraint(
            "cache_write_price_per_million IS NULL OR cache_write_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_non_negative",
        ),
        sa.CheckConstraint(
            "cache_write_1h_price_per_million IS NULL OR cache_write_1h_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_1h_non_negative",
        ),
        sa.Index(
            "uq_organization_model_pricing_period_start",
            "organization_id",
            "model_key",
            "effective_from",
            unique=True,
        ),
    )


def _copy_from(table: str, numeric: bool) -> sa.Table:
    if table == "model_pricing":
        return _model_pricing_table(numeric)
    return _organization_model_pricing_table(numeric)


def _convert(to_numeric: bool) -> None:
    """Retype every money column, in whichever direction is being run."""
    is_sqlite = op.get_bind().dialect.name == "sqlite"
    rate_type: sa.types.TypeEngine[object] = _RATE_TYPE if to_numeric else _FLOAT_TYPE
    cost_type: sa.types.TypeEngine[object] = _COST_TYPE if to_numeric else _FLOAT_TYPE
    existing_rate_type: sa.types.TypeEngine[object] = _FLOAT_TYPE if to_numeric else _RATE_TYPE
    existing_cost_type: sa.types.TypeEngine[object] = _FLOAT_TYPE if to_numeric else _COST_TYPE

    for table in _PRICING_TABLES:
        if is_sqlite:
            # The rebuild starts from the table as it is now, so ``copy_from``
            # describes the *old* column types.
            with op.batch_alter_table(table, copy_from=_copy_from(table, not to_numeric)) as batch_op:
                for name, nullable in _RATE_COLUMNS:
                    batch_op.alter_column(
                        name,
                        existing_type=existing_rate_type,
                        type_=rate_type,
                        existing_nullable=nullable,
                    )
            continue
        for name, nullable in _RATE_COLUMNS:
            op.alter_column(
                table,
                name,
                existing_type=existing_rate_type,
                type_=rate_type,
                existing_nullable=nullable,
                postgresql_using=f"{name}::{'numeric' if to_numeric else 'double precision'}",
            )

    if is_sqlite:
        # usage_logs carries no CHECK constraint, so reflection describes it
        # fully and the rebuild needs no hand-written copy.
        with op.batch_alter_table("usage_logs") as batch_op:
            batch_op.alter_column("cost", existing_type=existing_cost_type, type_=cost_type, existing_nullable=True)
        return
    op.alter_column(
        "usage_logs",
        "cost",
        existing_type=existing_cost_type,
        type_=cost_type,
        existing_nullable=True,
        postgresql_using=f"cost::{'numeric' if to_numeric else 'double precision'}",
    )


def upgrade() -> None:
    """Retype the rate and cost columns as exact numerics."""
    _convert(to_numeric=True)


def downgrade() -> None:
    """Return the rate and cost columns to binary floating point.

    Lossy in principle and not in practice: a value that fits 8 decimals in
    numeric round-trips through a double, and every stored rate and micro-dollar
    cost does.
    """
    _convert(to_numeric=False)
