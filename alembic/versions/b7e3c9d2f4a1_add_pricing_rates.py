"""Store every price as the models.dev cost block it is.

A price row held four per-million rates, ``pricing_tiers`` and a ``unit``: one
price per direction, in one unit, for one modality. models.dev v2 publishes a
price per modality and direction, each with its unit, cache and overrides, a
web search price beside the modalities, and multipliers. ``rates`` holds that
block whole, in the document's own shape, and is the truth of the row from
here on; the six columns stay as the text-token view an older reader has.
Every existing row gets the block its columns say, in the slot its unit
chooses, so no row is without one. Text rather than the engine's JSON type,
because that reads every number as a float and this column holds money.

The transliteration is spelled out here rather than imported, as this
revision keeps its meaning when the application's changes; the SQLite chain
test pins the two to each other on this date.

Revision ID: b7e3c9d2f4a1
Revises: e7c2a4f6b8d0
Create Date: 2026-11-04
"""

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "b7e3c9d2f4a1"
down_revision: str | Sequence[str] | None = "e7c2a4f6b8d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("model_pricing", "organization_model_pricing")
_PRIMARY_KEY = "model_pricing_pkey"
_RATE_TYPE = sa.Numeric(18, 8)
_MILLION = Decimal(1_000_000)
# What one published price buys, per unit, as ``gateway.types.price_unit.PRICE_PER`` says on this date.
_PER = {
    "token": 1_000_000,
    "request": 1,
    "image": 1,
    "character": 1_000_000,
    "megapixel": 1,
    "second": 1,
    "minute": 1,
    "web_search": 1_000,
}
# The plural names a row written by an older writer may still carry.
_RETIRED = {"tokens": "token", "requests": "request", "images": "image"}
# The slot a unit's columns are written into: the modality (``None`` for the
# ``web_search`` field), the direction the input column prices, and the one
# the output column prices (``None``: not carried).
_SLOTS: dict[str, tuple[str | None, str, str | None]] = {
    "token": ("text", "input", "output"),
    "character": ("text", "input", "output"),
    "request": ("text", "input", "output"),
    "image": ("image", "output", None),
    "megapixel": ("image", "output", None),
    "second": ("audio", "input", "output"),
    "minute": ("audio", "input", "output"),
    "web_search": (None, "input", None),
}
_TIER_RATES = {
    "input": "input_price_per_million",
    "output": "output_price_per_million",
    "cache_read": "cache_read_price_per_million",
    "cache_write": "cache_write_price_per_million",
}


def _model_pricing_table(*, rates: sa.Column[Any] | None) -> sa.Table:
    """``model_pricing`` as it stands, for SQLite's batch rebuild.

    Spelled out rather than reflected, as ``a7c3e5d9b1f4`` explains: batch mode
    has mis-reflected this table's composite primary key before.
    """
    columns: list[sa.SchemaItem] = [
        sa.Column("kind", sa.String(length=16), nullable=False, server_default="model"),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_price_per_million", _RATE_TYPE, nullable=False),
        sa.Column("output_price_per_million", _RATE_TYPE, nullable=False),
        sa.Column("cache_read_price_per_million", _RATE_TYPE, nullable=True),
        sa.Column("cache_write_price_per_million", _RATE_TYPE, nullable=True),
        sa.Column("pricing_tiers", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("unit", sa.String(length=16), nullable=False, server_default="token"),
        sa.Column("origin", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("kind", "model_key", "effective_at", name=_PRIMARY_KEY),
    ]
    if rates is not None:
        columns.insert(9, rates)
    return sa.Table("model_pricing", sa.MetaData(), *columns)


def _rate_columns(table: str) -> sa.TableClause:
    return sa.table(
        table,
        sa.column("input_price_per_million", _RATE_TYPE),
        sa.column("output_price_per_million", _RATE_TYPE),
        sa.column("cache_read_price_per_million", _RATE_TYPE),
        sa.column("cache_write_price_per_million", _RATE_TYPE),
        sa.column("pricing_tiers", sa.JSON()),
        sa.column("unit", sa.String(length=16)),
        sa.column("rates", sa.Text()),
    )


def _published(column: object, per: Decimal) -> Decimal:
    return Decimal(str(column)) * per


def _block(unit: str, row: Mapping[Any, Any]) -> str:
    """The cost block ``row``'s six columns say, as ``gateway.types.price_rates.rates_from_columns`` writes it."""
    per = Decimal(_PER[unit]) / _MILLION
    modality, given_direction, made_direction = _SLOTS[unit]
    given_overrides: list[dict[str, Any]] = []
    made_overrides: list[dict[str, Any]] = []
    for tier in row["pricing_tiers"] or []:
        when = {"context": {"gte": tier["min_input_tokens"]}}
        override: dict[str, Any] = {"when": when}
        if (price := tier.get(_TIER_RATES["input"])) is not None:
            override["price"] = _published(price, per)
        cache = _cache(tier.get(_TIER_RATES["cache_read"]), tier.get(_TIER_RATES["cache_write"]), per)
        if cache:
            override["cache"] = cache
        if len(override) > 1:
            given_overrides.append(override)
        if made_direction is not None and (price := tier.get(_TIER_RATES["output"])) is not None:
            made_overrides.append({"when": when, "price": _published(price, per)})
    given: dict[str, Any] = {"price": _published(row["input_price_per_million"], per), "unit": unit}
    if cache := _cache(row["cache_read_price_per_million"], row["cache_write_price_per_million"], per):
        given["cache"] = cache
    if given_overrides:
        given["overrides"] = given_overrides
    if modality is None:
        return _dump({"web_search": given})
    prices = {given_direction: given}
    if made_direction is not None:
        made: dict[str, Any] = {"price": _published(row["output_price_per_million"], per), "unit": unit}
        if made_overrides:
            made["overrides"] = made_overrides
        prices[made_direction] = made
    return _dump({modality: {direction: prices[direction] for direction in ("input", "output") if direction in prices}})


def _cache(read: object, write: object, per: Decimal) -> dict[str, Decimal]:
    cache = {}
    if read is not None:
        cache["read"] = _published(read, per)
    if write is not None:
        cache["write"] = _published(write, per)
    return cache


def _dump(value: object) -> str:
    """JSON with every number in plain digits, in the order given: the column's own encoding."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(Decimal(repr(value)).normalize(), "f")
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{json.dumps(str(key))}:{_dump(item)}" for key, item in value.items()) + "}"
    if isinstance(value, Sequence):
        return "[" + ",".join(_dump(item) for item in value) + "]"
    raise TypeError(f"rates cannot hold a {type(value).__name__}")


def _backfill(connection: sa.Connection, table: str, key: tuple[str, ...]) -> None:
    rates = _rate_columns(table)
    keys: list[sa.ColumnClause[Any]] = [sa.column(name) for name in key]
    rows = connection.execute(sa.select(*keys, rates)).mappings().all()
    for row in rows:
        unit = _RETIRED.get(row["unit"], row["unit"])
        connection.execute(
            sa.update(rates).where(*(sa.column(name) == row[name] for name in key)).values(rates=_block(unit, row))
        )


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("rates", sa.Text(), nullable=True))
    connection = op.get_bind()
    _backfill(connection, "model_pricing", ("kind", "model_key", "effective_at"))
    _backfill(connection, "organization_model_pricing", ("id",))
    added = sa.Column("rates", sa.Text(), nullable=True)
    with op.batch_alter_table("model_pricing", copy_from=_model_pricing_table(rates=added)) as batch:
        batch.alter_column("rates", existing_type=sa.Text(), nullable=False)
    with op.batch_alter_table("organization_model_pricing") as batch:
        batch.alter_column("rates", existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("organization_model_pricing") as batch:
        batch.drop_column("rates")
    kept = sa.Column("rates", sa.Text(), nullable=False)
    with op.batch_alter_table("model_pricing", copy_from=_model_pricing_table(rates=kept)) as batch:
        batch.drop_column("rates")
