"""The pricing-provenance revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default and nothing else in the suite migrates it,
so this is the only coverage of that path for this revision. Three things are
worth pinning beyond "the columns appear". The revision is hand-written, so
nothing else would notice it and the model drifting apart. The lengths are
deliberate (they mirror the platform's ``gateway_usage_settlement`` columns, so a
value copied across by the hosted-usage backfill always fits), and a silently
unbounded column would only be found by the value that overflowed the platform's
own. And the downgrade drops five columns from a table that holds accounting
rows, so the round trip has to show those rows survive it.

The PostgreSQL half is ``tests/integration/test_pricing_provenance_columns.py``.
"""

import uuid
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
from gateway.models.entities import UsageLog

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_PROVENANCE_REVISION = "a9c4e2b6d8f1"
_BEFORE_PROVENANCE = "d8b3f1c6a4e9"

_TABLE = "usage_logs"
# The platform's settlement column types, which are what the backfill copies from.
_EXPECTED_TYPES = {
    "pricing_source": "VARCHAR(32)",
    "pricing_reference": "VARCHAR(511)",
    "pricing_effective_at": "DATETIME",
    "pricing_version": "VARCHAR(255)",
    "calculated_at": "DATETIME",
}

_RAN_AT = datetime(2026, 8, 1, 9, 30, tzinfo=UTC)
# Later than _RAN_AT on purpose: an amount can be settled or repriced well after
# the request it prices, which is why this is not `timestamp`.
_PRICED_AT = datetime(2026, 8, 3, 17, 5, tzinfo=UTC)
_EFFECTIVE_AT = datetime(2026, 5, 20, 0, 0, tzinfo=UTC)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'provenance.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _columns(engine: Engine) -> dict[str, ReflectedColumn]:
    return {column["name"]: column for column in inspect(engine).get_columns(_TABLE)}


def _utc(value: datetime | None) -> datetime | None:
    """Stamp UTC on a naive read-back.

    ``timezone=True`` is a no-op on SQLite (it has no timestamp type), so a value
    written aware reads back naive, exactly as this table's ``timestamp`` column
    already does on that engine.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


_PROVENANCE = {
    "pricing_source": "genai_prices",
    "pricing_reference": "openai:gpt-4o-mini",
    "pricing_effective_at": _EFFECTIVE_AT,
    "pricing_version": "0.0.30",
    "calculated_at": _PRICED_AT,
}


def _write_settled_row(engine: Engine, **provenance: object) -> None:
    with Session(engine) as session:
        session.add(
            UsageLog(
                id="settled-1",
                workspace_id=uuid.UUID(int=0),
                timestamp=_RAN_AT,
                model="openai:gpt-4o-mini",
                provider="openai",
                endpoint="/v1/chat/completions",
                status="success",
                prompt_tokens=1_000,
                completion_tokens=500,
                cost=Decimal("0.000450"),
                **provenance,
            )
        )
        session.commit()


def test_the_columns_land_nullable_with_the_platform_s_lengths(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head

    columns = _columns(engine)

    for name, expected_type in _EXPECTED_TYPES.items():
        assert str(columns[name]["type"]) == expected_type, name
        assert columns[name]["nullable"] is True, name


def test_the_migrated_table_matches_the_model(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head

    declared = set(SQLModel.metadata.tables[_TABLE].columns.keys())

    assert set(_columns(engine)) == declared


def test_a_settled_row_records_where_its_amount_came_from(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    _write_settled_row(engine, **_PROVENANCE)

    with Session(engine) as session:
        row = session.get(UsageLog, "settled-1")

    assert row is not None
    assert row.pricing_source == "genai_prices"
    assert row.pricing_reference == "openai:gpt-4o-mini"
    assert row.pricing_version == "0.0.30"
    assert _utc(row.pricing_effective_at) == _EFFECTIVE_AT
    # When the amount was priced, not when the request ran.
    assert _utc(row.calculated_at) == _PRICED_AT
    assert _utc(row.timestamp) == _RAN_AT


def test_a_row_priced_by_the_gateway_leaves_the_provenance_null(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Nothing in this edition records provenance, so the columns have to stay optional."""
    _, engine = sqlite_at_head
    _write_settled_row(engine)

    with Session(engine) as session:
        row = session.get(UsageLog, "settled-1")

    assert row is not None
    assert row.cost == Decimal("0.000450")
    assert all(getattr(row, name) is None for name in _EXPECTED_TYPES)


def test_the_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head
    _write_settled_row(engine, **_PROVENANCE)

    command.downgrade(config, _BEFORE_PROVENANCE)

    assert set(_EXPECTED_TYPES).isdisjoint(_columns(engine))
    with engine.connect() as connection:
        # Raw SQL: the mapped class tracks the current schema, and this database
        # is pinned to the revision before the columns it declares.
        cost = connection.execute(text("SELECT cost FROM usage_logs WHERE id = 'settled-1'")).scalar_one()
    assert Decimal(str(cost)) == Decimal("0.000450")

    command.upgrade(config, _PROVENANCE_REVISION)

    columns = _columns(engine)
    assert set(_EXPECTED_TYPES) <= set(columns)
    with Session(engine) as session:
        row = session.get(UsageLog, "settled-1")
    assert row is not None
    # The provenance went with the columns; nothing backfills it.
    assert all(getattr(row, name) is None for name in _EXPECTED_TYPES)
