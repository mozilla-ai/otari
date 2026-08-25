"""Pricing provenance on PostgreSQL: the column types, and the round trip.

The revision's SQLite half is ``tests/unit/test_pricing_provenance_schema_chain.py``.
PostgreSQL is where the two things that engine cannot show land: a ``varchar``
length is enforced rather than advisory, so the ceilings copied from the
platform's ``gateway_usage_settlement`` are real, and ``timezone=True`` is
honored, so a provenance timestamp reads back UTC-aware.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import DateTime, String, inspect, select
from sqlalchemy.exc import DataError
from sqlalchemy.orm import Session

from conftest import seed_workspace_id
from gateway.models.entities import UsageLog

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_PROVENANCE_REVISION = "a9c4e2b6d8f1"
_BEFORE_PROVENANCE = "d8b3f1c6a4e9"

_RAN_AT = datetime(2026, 8, 1, 9, 30, tzinfo=UTC)
# Later than _RAN_AT on purpose: an amount can be settled or repriced well after
# the request it prices, which is why this is not ``timestamp``.
_PRICED_AT = datetime(2026, 8, 3, 17, 5, tzinfo=UTC)
_EFFECTIVE_AT = datetime(2026, 5, 20, 0, 0, tzinfo=UTC)

_EXPECTED_LENGTHS = {
    "pricing_source": 32,
    "pricing_reference": 511,
    "pricing_version": 255,
}
_TIMESTAMP_COLUMNS = ("pricing_effective_at", "calculated_at")
_PROVENANCE_COLUMNS = (*_EXPECTED_LENGTHS, *_TIMESTAMP_COLUMNS)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _provenance_columns(test_db: Session) -> set[str]:
    reflected = {column["name"] for column in inspect(test_db.get_bind()).get_columns("usage_logs")}
    return reflected & set(_PROVENANCE_COLUMNS)


def _settled_row(test_db: Session, **provenance: object) -> UsageLog:
    return UsageLog(
        id="settled-1",
        workspace_id=seed_workspace_id(test_db),
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


def test_the_columns_are_nullable_with_the_platform_s_types(test_db: Session) -> None:
    columns = {column["name"]: column for column in inspect(test_db.get_bind()).get_columns("usage_logs")}

    for name in _PROVENANCE_COLUMNS:
        assert columns[name]["nullable"] is True, name
    for name, length in _EXPECTED_LENGTHS.items():
        string_type = columns[name]["type"]
        assert isinstance(string_type, String), name
        assert string_type.length == length, name
    for name in _TIMESTAMP_COLUMNS:
        # ``timestamptz``. A naive column would read back offset-less and a
        # reader would take a UTC value for local time.
        timestamp_type = columns[name]["type"]
        assert isinstance(timestamp_type, DateTime), name
        assert timestamp_type.timezone is True, name


def test_a_settled_row_keeps_its_provenance_utc_aware(test_db: Session) -> None:
    test_db.add(
        _settled_row(
            test_db,
            pricing_source="organization",
            pricing_reference="7f3a1c9e-0000-4000-8000-000000000001",
            pricing_effective_at=_EFFECTIVE_AT,
            pricing_version="2026-05-20T00:00:00+00:00",
            calculated_at=_PRICED_AT,
        )
    )
    test_db.commit()
    test_db.expunge_all()

    row = test_db.get(UsageLog, "settled-1")

    assert row is not None
    assert row.pricing_source == "organization"
    assert row.pricing_reference == "7f3a1c9e-0000-4000-8000-000000000001"
    assert row.pricing_version == "2026-05-20T00:00:00+00:00"
    assert row.pricing_effective_at == _EFFECTIVE_AT
    # When the amount was priced, not when the request ran.
    assert row.calculated_at == _PRICED_AT
    assert row.timestamp == _RAN_AT


def test_a_row_priced_by_the_gateway_leaves_the_provenance_null(test_db: Session) -> None:
    """Nothing in this edition records provenance, so the columns have to stay optional."""
    test_db.add(_settled_row(test_db))
    test_db.commit()
    test_db.expunge_all()

    row = test_db.get(UsageLog, "settled-1")

    assert row is not None
    assert row.cost == Decimal("0.000450")
    assert all(getattr(row, name) is None for name in _PROVENANCE_COLUMNS)


def test_a_source_longer_than_the_platform_s_column_is_refused(test_db: Session) -> None:
    """The ceiling is the platform's own, so a value it could hold cannot be truncated here."""
    test_db.add(_settled_row(test_db, pricing_source="x" * 33))

    with pytest.raises(DataError):
        test_db.commit()


def test_the_revision_round_trips_on_postgresql_with_rows_in_the_table(
    test_db: Session, postgres_url: str
) -> None:
    """Drop the five columns and add them back, with data, on the engine CI migrates.

    The SQLite half is ``tests/unit/test_pricing_provenance_schema_chain.py``.
    This is the PostgreSQL half, and it is the one that runs against a real
    deployment's accounting rows.
    """
    test_db.add(
        _settled_row(
            test_db,
            pricing_source="managed",
            pricing_reference="7f3a1c9e-0000-4000-8000-000000000001",
            pricing_effective_at=_EFFECTIVE_AT,
            pricing_version="2026-05-20T00:00:00+00:00",
            calculated_at=_PRICED_AT,
        )
    )
    test_db.commit()
    # Released before the ALTERs: an idle transaction would block them.
    test_db.close()

    config = _alembic_config(postgres_url)
    try:
        command.downgrade(config, _BEFORE_PROVENANCE)

        assert _provenance_columns(test_db) == set()
        # One column rather than the whole entity: the mapped class tracks the
        # current schema and this database is pinned a revision behind it.
        settled_cost = test_db.execute(select(UsageLog.cost).where(UsageLog.id == "settled-1")).scalar_one()
        assert settled_cost == Decimal("0.000450")

        test_db.close()
        command.upgrade(config, _PROVENANCE_REVISION)

        assert _provenance_columns(test_db) == set(_PROVENANCE_COLUMNS)
        row = test_db.get(UsageLog, "settled-1")
        assert row is not None
        # The provenance went with the columns; nothing backfills it.
        assert all(getattr(row, name) is None for name in _PROVENANCE_COLUMNS)
    finally:
        # Put the database back at head on the failure path as much as the happy
        # one: the integration fixtures migrate once per session and share the
        # database, so leaving it a revision behind would hand every later test a
        # schema its models no longer match. Released first for the same reason
        # the downgrade is.
        test_db.close()
        command.upgrade(config, "head")
