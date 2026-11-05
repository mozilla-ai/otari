"""The singular-unit revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default and the integration suite migrates
PostgreSQL only, so this is that engine's coverage for the revision. Three
things are pinned: every row in a retired name reads back in the current one
and a row in a newer unit is left alone, the column's default follows on both
tables, and the SQLite rebuild the default change costs keeps the override
table's constraints and the price table's composite key.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "d1145eb686f8"
_BEFORE = "a79a97473478"
_TABLES = ("model_pricing", "organization_model_pricing")
_AT = "2026-05-20 00:00:00.000000"
_ORGANIZATION = uuid.uuid4()


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A database at the revision before this one, holding one rate per unit on each table."""
    database_url = f"sqlite:///{tmp_path / 'units.db'}"
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
        for unit in ("tokens", "requests", "images", "second"):
            connection.execute(
                text(
                    "INSERT INTO model_pricing (kind, model_key, effective_at, input_price_per_million, "
                    "output_price_per_million, pricing_tiers, unit, created_at, updated_at) "
                    "VALUES ('model', :key, :at, 1, 2, '[]', :unit, :at, :at)"
                ),
                {"key": f"openai:{unit}", "unit": unit, "at": _AT},
            )
            connection.execute(
                text(
                    "INSERT INTO organization_model_pricing (id, organization_id, kind, model_key, effective_from, "
                    "input_price_per_million, output_price_per_million, pricing_tiers, unit, created_at, updated_at) "
                    "VALUES (:id, :org, 'model', :key, :at, 1, 2, '[]', :unit, :at, :at)"
                ),
                {"id": uuid.uuid4().hex, "org": _ORGANIZATION.hex, "key": f"openai:{unit}", "unit": unit, "at": _AT},
            )
    try:
        yield config, engine
    finally:
        engine.dispose()


def _units(engine: Engine, table: str) -> dict[str, str]:
    """The unit of every row by key: the seeded rates and, on ``model_pricing``, the one ``9964faffc86b`` wrote."""
    with engine.connect() as connection:
        rows = connection.execute(text(f"SELECT model_key, unit FROM {table}")).all()  # noqa: S608
    return {key: unit for key, unit in rows}


_RENAMED = {"openai:tokens": "token", "openai:requests": "request", "openai:images": "image", "openai:second": "second"}
_RETIRED = {
    "openai:tokens": "tokens",
    "openai:requests": "requests",
    "openai:images": "images",
    "openai:second": "second",
}
# The built-in fetcher's price, which the earlier revision wrote in the retired name.
_WRITTEN_BEFORE = {"model_pricing": "builtin_fetch"}


def _expected(table: str, *, retired: bool) -> dict[str, str]:
    seeded = _RETIRED if retired else _RENAMED
    written = _WRITTEN_BEFORE.get(table)
    return seeded if written is None else {**seeded, written: "requests" if retired else "request"}


def test_every_row_reads_in_the_current_name_and_a_newer_unit_is_left_alone(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before

    command.upgrade(config, _REVISION)

    for table in _TABLES:
        assert _units(engine, table) == _expected(table, retired=False), table


def test_the_default_follows_on_both_tables(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    for table in _TABLES:
        columns = {column["name"]: column for column in inspect(engine).get_columns(table)}
        assert columns["unit"]["default"] == "'token'", table
        assert columns["unit"]["nullable"] is False, table
        # The model has since gained ``rates``, which this revision predates.
        assert set(columns) == set(SQLModel.metadata.tables[table].columns.keys()) - {"rates"}, table
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_pricing (kind, model_key, effective_at, input_price_per_million, "
                "output_price_per_million, pricing_tiers, created_at, updated_at) "
                "VALUES ('model', 'openai:defaulted', :at, 1, 2, '[]', :at, :at)"
            ),
            {"at": _AT},
        )
    assert _units(engine, "model_pricing")["openai:defaulted"] == "token"


def test_the_rebuild_keeps_what_sqlite_cannot_alter_in_place(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    inspector = inspect(engine)
    model = SQLModel.metadata.tables["organization_model_pricing"]
    assert {check["name"] for check in inspector.get_check_constraints("organization_model_pricing")} == {
        constraint.name
        for constraint in model.constraints
        if constraint.name and str(constraint.name).startswith("ck_")
    }
    assert {index["name"] for index in inspector.get_indexes("organization_model_pricing")} >= {
        index.name for index in model.indexes
    }
    (foreign_key,) = inspector.get_foreign_keys("organization_model_pricing")
    assert (foreign_key["referred_table"], foreign_key["options"].get("ondelete")) == ("organization", "CASCADE")
    assert inspector.get_pk_constraint("model_pricing")["constrained_columns"] == ["kind", "model_key", "effective_at"]


def test_the_revision_round_trips(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    for table in _TABLES:
        assert _units(engine, table) == _expected(table, retired=True), table
        columns = {column["name"]: column for column in inspect(engine).get_columns(table)}
        assert columns["unit"]["default"] == "'tokens'", table

    command.upgrade(config, _REVISION)
    for table in _TABLES:
        assert _units(engine, table) == _expected(table, retired=False), table
