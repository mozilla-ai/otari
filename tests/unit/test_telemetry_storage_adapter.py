"""Unit coverage for the core TelemetryStoragePort adapter's batch-insert path.

Run against a real SQLite database rather than a mocked session, because what
matters is what the insert leaves in the table: a replayed or overlapping export
stores each record once and reports the rest as duplicates, without failing.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
from gateway.models.usage import AgentTelemetry
from gateway.ports.telemetry_storage_port import TelemetryFilter, TelemetryRecord

_TS = datetime(2026, 8, 6, tzinfo=UTC)


def _record(dedup_key: str, source: str = "claude_code") -> TelemetryRecord:
    return TelemetryRecord(
        name="tool_result",
        timestamp=_TS,
        source=source,
        dedup_key=dedup_key,
        tool_name="Bash",
    )


@asynccontextmanager
async def _session(*, foreign_keys: bool = False) -> AsyncIterator[AsyncSession]:
    """A session on a fresh in-memory SQLite schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    if foreign_keys:

        def _enforce(dbapi_connection: Any, _record: Any) -> None:
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

        event.listen(engine.sync_engine, "connect", _enforce)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        # aiosqlite runs each connection on its own thread, so an undisposed
        # engine leaks one per test.
        await engine.dispose()


async def _stored(db: AsyncSession) -> list[tuple[str, str]]:
    rows = await db.execute(select(AgentTelemetry.source, AgentTelemetry.dedup_key).order_by(AgentTelemetry.dedup_key))
    return [(source, key) for source, key in rows.all()]


@pytest.mark.asyncio
async def test_record_stores_every_new_record() -> None:
    async with _session() as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(
            api_key_id="key-1",
            user_id="alice",
            records=tuple(_record(f"dedup-{i}") for i in range(5)),
        )

        assert (result.accepted, result.duplicate) == (5, 0)
        assert len(await _stored(db)) == 5


@pytest.mark.asyncio
async def test_record_attributes_every_row_to_the_exporting_key_and_user() -> None:
    async with _session() as db:
        await DatabaseTelemetryStorageAdapter(db).record(
            api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),)
        )

        row = (await db.execute(select(AgentTelemetry))).scalar_one()
        assert (row.api_key_id, row.user_id, row.name) == ("key-1", "alice", "tool_result")


@pytest.mark.asyncio
async def test_record_keys_duplicates_by_source() -> None:
    """The same dedup key from two sources is two records."""
    async with _session() as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(
            api_key_id="key-1",
            user_id="alice",
            records=(_record("dedup-0", "claude_code"), _record("dedup-0", "codex")),
        )

        assert (result.accepted, result.duplicate) == (2, 0)
        assert await _stored(db) == [("claude_code", "dedup-0"), ("codex", "dedup-0")]


@pytest.mark.asyncio
async def test_record_reports_a_replayed_export_as_duplicates() -> None:
    """An exporter resending a batch stores nothing twice and does not fail."""
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        records = tuple(_record(f"dedup-{i}") for i in range(3))
        await adapter.record(api_key_id="key-1", user_id="alice", records=records)

        result = await adapter.record(api_key_id="key-1", user_id="alice", records=records)

        assert (result.accepted, result.duplicate) == (0, 3)
        assert len(await _stored(db)) == 3


@pytest.mark.asyncio
async def test_record_stores_only_the_new_part_of_an_overlapping_export() -> None:
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        await adapter.record(api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))

        result = await adapter.record(
            api_key_id="key-1",
            user_id="alice",
            records=(_record("dedup-0"), _record("dedup-1"), _record("dedup-2")),
        )

        assert (result.accepted, result.duplicate) == (2, 1)
        assert [key for _, key in await _stored(db)] == ["dedup-0", "dedup-1", "dedup-2"]


@pytest.mark.asyncio
async def test_record_drops_in_export_repeats_before_the_insert() -> None:
    async with _session() as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(
            api_key_id="key-1",
            user_id="alice",
            records=(_record("dedup-0"), _record("dedup-1"), _record("dedup-0")),
        )

        assert (result.accepted, result.duplicate) == (2, 1)
        assert len(await _stored(db)) == 2


@pytest.mark.asyncio
async def test_record_counts_rows_the_database_refuses_instead_of_failing_the_export() -> None:
    """A refusal other than the unique key (here a missing user) falls back to row at a time."""
    async with _session(foreign_keys=True) as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(
            api_key_id="no-such-key",
            user_id="nobody",
            records=(_record("dedup-0"), _record("dedup-1")),
        )

        assert (result.accepted, result.duplicate) == (0, 2)
        assert await _stored(db) == []


@pytest.mark.asyncio
async def test_record_stores_nothing_for_an_empty_export() -> None:
    async with _session() as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(api_key_id="key-1", user_id="alice", records=())

        assert (result.accepted, result.duplicate, result.rejected) == (0, 0, 0)
        assert (await db.execute(select(func.count()).select_from(AgentTelemetry))).scalar_one() == 0


@pytest.mark.asyncio
async def test_adapter_without_a_session_refuses_rather_than_storing_nothing() -> None:
    """Every surface behind this port is standalone-only, so a session-less
    request is a wiring mistake and must not look like a successful no-op."""
    adapter = DatabaseTelemetryStorageAdapter(None)

    with pytest.raises(RuntimeError, match="database session"):
        await adapter.record(api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))
    with pytest.raises(RuntimeError, match="database session"):
        await adapter.count(filters=TelemetryFilter())
