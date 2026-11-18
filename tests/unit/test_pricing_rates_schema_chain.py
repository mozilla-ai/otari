"""The rates revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default and the integration suite migrates
PostgreSQL only, so this is that engine's coverage for the revision. Three
things are pinned: every existing row gets the block its columns say, spelled
exactly as the application spells it, on both tables; the column lands NOT
NULL with the model's type, so a raw writer cannot leave it empty; and the
downgrade drops it and leaves the columns it was written from intact.
"""

import uuid
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.types.price_rates import RateColumns, dump_rates, rates_from_columns
from gateway.types.price_unit import PriceUnit

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "b7e3c9d2f4a1"
_BEFORE = "e7c2a4f6b8d0"
_TABLES = ("model_pricing", "organization_model_pricing")
_AT = "2026-05-20 00:00:00.000000"
_ORGANIZATION = uuid.uuid4()
_TIERS = '[{"min_input_tokens": 200000, "input_price_per_million": 5.0, "output_price_per_million": 15.0}]'

# One row per shape the transliteration has: a unit, its two rates, the cache rates and the tiers.
_ROWS: dict[str, tuple[PriceUnit, str, str, str | None, str | None, str]] = {
    "openai:gpt-4o": ("token", "2.5", "10", "1.25", "3.75", _TIERS),
    "openai:omni-moderation": ("request", "5000", "0", None, None, "[]"),
    "openai:gpt-image-1": ("image", "40000", "0", None, None, "[]"),
    "otari:web_search": ("web_search", "2000", "0", None, None, "[]"),
    "openai:whisper": ("second", "6000", "0", None, None, "[]"),
    "openai:tts": (
        "character",
        "15",
        "0",
        "1",
        None,
        '[{"min_input_tokens": 1000, "cache_read_price_per_million": 2}]',
    ),  # noqa: E501
}


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A database at the revision before this one, holding one rate per shape on each table."""
    database_url = f"sqlite:///{tmp_path / 'rates.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO organization (id, name, slug, created_at, updated_at) "
                "VALUES (:id, 'Acme', 'acme', :at, :at)"
            ),
            {"id": _ORGANIZATION.hex, "at": _AT},
        )
        for key, (unit, given, made, read, write, tiers) in _ROWS.items():
            values = {"key": key, "unit": unit, "in": given, "out": made, "read": read, "write": write, "tiers": tiers}
            connection.execute(
                text(
                    "INSERT INTO model_pricing (kind, model_key, effective_at, input_price_per_million, "
                    "output_price_per_million, cache_read_price_per_million, cache_write_price_per_million, "
                    "pricing_tiers, unit, created_at, updated_at) "
                    "VALUES ('model', :key, :at, :in, :out, :read, :write, :tiers, :unit, :at, :at)"
                ),
                {**values, "at": _AT},
            )
            connection.execute(
                text(
                    "INSERT INTO organization_model_pricing (id, organization_id, kind, model_key, effective_from, "
                    "input_price_per_million, output_price_per_million, cache_read_price_per_million, "
                    "cache_write_price_per_million, pricing_tiers, unit, created_at, updated_at) "
                    "VALUES (:id, :org, 'model', :key, :at, :in, :out, :read, :write, :tiers, :unit, :at, :at)"
                ),
                {**values, "id": uuid.uuid4().hex, "org": _ORGANIZATION.hex, "at": _AT},
            )
    try:
        yield config, engine
    finally:
        engine.dispose()


def _expected(key: str) -> str:
    """The block the application writes for the row's columns."""
    unit, given, made, read, write, tiers = _ROWS[key]
    import json

    columns = RateColumns(
        unit=unit,
        input_price_per_million=Decimal(given),
        output_price_per_million=Decimal(made),
        cache_read_price_per_million=None if read is None else Decimal(read),
        cache_write_price_per_million=None if write is None else Decimal(write),
        pricing_tiers=tuple(json.loads(tiers)),
    )
    return dump_rates(rates_from_columns(columns))


def _rates(engine: Engine, table: str) -> dict[str, Any]:
    with engine.connect() as connection:
        rows = connection.execute(text(f"SELECT model_key, rates FROM {table}")).all()  # noqa: S608
    return {key: rates for key, rates in rows}


def test_every_row_gets_the_block_its_columns_say(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before

    command.upgrade(config, _REVISION)

    for table in _TABLES:
        rates = _rates(engine, table)
        assert {key: rates[key] for key in _ROWS} == {key: _expected(key) for key in _ROWS}, table
    # The free per-request row ``9964faffc86b`` wrote gets its block too, so no row is without one.
    assert _rates(engine, "model_pricing")["builtin_fetch"] == (
        '{"text":{"input":{"price":0,"unit":"request"},"output":{"price":0,"unit":"request"}}}'
    )


def test_the_column_lands_not_null_with_the_model_s_type(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    for table in _TABLES:
        columns = {column["name"]: column for column in inspect(engine).get_columns(table)}
        assert str(columns["rates"]["type"]) == "TEXT", table
        assert columns["rates"]["nullable"] is False, table
        assert set(columns) == set(SQLModel.metadata.tables[table].columns.keys()), table
    assert inspector_keeps_the_composite_key(engine)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_pricing (kind, model_key, effective_at, input_price_per_million, "
                "output_price_per_million, pricing_tiers, created_at, updated_at) "
                "VALUES ('model', 'openai:unwritten', :at, 1, 2, '[]', :at, :at)"
            ),
            {"at": _AT},
        )


def inspector_keeps_the_composite_key(engine: Engine) -> bool:
    return inspect(engine).get_pk_constraint("model_pricing")["constrained_columns"] == [
        "kind",
        "model_key",
        "effective_at",
    ]


def test_the_revision_round_trips(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    for table in _TABLES:
        columns = {column["name"] for column in inspect(engine).get_columns(table)}
        assert "rates" not in columns, table
    with engine.connect() as connection:
        rate = connection.execute(
            text("SELECT input_price_per_million FROM model_pricing WHERE model_key = 'openai:gpt-image-1'")
        ).scalar_one()
    assert Decimal(str(rate)) == Decimal(40000)
    assert inspector_keeps_the_composite_key(engine)

    command.upgrade(config, _REVISION)
    assert _rates(engine, "model_pricing")["openai:gpt-image-1"] == _expected("openai:gpt-image-1")
