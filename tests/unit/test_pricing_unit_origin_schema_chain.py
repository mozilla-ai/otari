"""The unit-and-origin revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default and the integration suite migrates
PostgreSQL only, so this is that engine's coverage for the revision. Three
things are pinned: the columns land on both price tables with the model's
types and defaults, the one backfill the key spelling justifies happens
(``otari:`` tool rows become ``requests``) and nothing else is guessed at, and
the downgrade leaves the rates it was added beside intact.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine.interfaces import ReflectedColumn
from sqlalchemy.orm import Session
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.entities import ModelPricing

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c7e9a1b3d5f7"
_BEFORE = "f1c4a8e2d6b9"
_TABLES = ("model_pricing", "organization_model_pricing")
_EFFECTIVE_AT = datetime(2026, 5, 20, 0, 0, tzinfo=UTC)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A database at the revision before this one, so the upgrade can be watched."""
    database_url = f"sqlite:///{tmp_path / 'units.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _columns(engine: Engine, table: str) -> dict[str, ReflectedColumn]:
    return {column["name"]: column for column in inspect(engine).get_columns(table)}


# SQLite has no timestamp type; SQLAlchemy stores one as this string, so a raw
# insert has to spell it the same way for the mapped class to find the row.
_EFFECTIVE_AT_SQLITE = "2026-05-20 00:00:00.000000"


def _insert_legacy_row(engine: Engine, model_key: str) -> None:
    # Raw SQL, because the mapped class already declares the columns this
    # database does not yet have.
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_pricing (model_key, effective_at, input_price_per_million, "
                "output_price_per_million, pricing_tiers, created_at, updated_at) "
                "VALUES (:key, :at, 10000, 0, '[]', :at, :at)"
            ),
            {"key": model_key, "at": _EFFECTIVE_AT_SQLITE},
        )


def test_the_columns_land_on_both_tables_with_the_model_s_shape(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before

    command.upgrade(config, _REVISION)

    for table in _TABLES:
        columns = _columns(engine, table)
        assert str(columns["unit"]["type"]) == "VARCHAR(16)", table
        assert columns["unit"]["nullable"] is False, table
        assert str(columns["origin"]["type"]) == "VARCHAR(16)", table
        assert columns["origin"]["nullable"] is True, table
        assert set(columns) == set(SQLModel.metadata.tables[table].columns.keys()), table


def test_existing_rows_read_as_tokens_with_no_origin_except_a_tool_row(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    _insert_legacy_row(engine, "openai:gpt-4o")
    _insert_legacy_row(engine, "otari:web_search")

    command.upgrade(config, _REVISION)

    with Session(engine) as session:
        model = session.get(ModelPricing, ("openai:gpt-4o", _EFFECTIVE_AT))
        tool = session.get(ModelPricing, ("otari:web_search", _EFFECTIVE_AT))
    assert model is not None and tool is not None
    assert model.unit == "tokens"
    # The reserved prefix is the one spelling that says "per request" for certain.
    assert tool.unit == "requests"
    # Nobody knows which path wrote a pre-existing row, and the migration does not pretend to.
    assert model.origin is None
    assert tool.origin is None


def test_a_new_row_defaults_to_tokens_and_records_its_writer(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    with Session(engine) as session:
        session.add(
            ModelPricing(
                model_key="anthropic:claude-sonnet-4-6",
                effective_at=_EFFECTIVE_AT,
                input_price_per_million=Decimal("3"),
                output_price_per_million=Decimal("15"),
                pricing_tiers=[],
                origin="api",
            )
        )
        session.commit()
        row = session.get(ModelPricing, ("anthropic:claude-sonnet-4-6", _EFFECTIVE_AT))

    assert row is not None
    assert row.unit == "tokens"
    assert row.origin == "api"


def test_the_revision_round_trips(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    _insert_legacy_row(engine, "otari:web_search")
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    for table in _TABLES:
        assert {"unit", "origin"}.isdisjoint(_columns(engine, table)), table
    with engine.connect() as connection:
        rate = connection.execute(
            text("SELECT input_price_per_million FROM model_pricing WHERE model_key = 'otari:web_search'")
        ).scalar_one()
    assert Decimal(str(rate)) == Decimal("10000")

    command.upgrade(config, _REVISION)
    with Session(engine) as session:
        row = session.get(ModelPricing, ("otari:web_search", _EFFECTIVE_AT))
    assert row is not None
    assert row.unit == "requests"
