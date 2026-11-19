"""Unit coverage for the core TelemetryStoragePort adapter's batch-insert path.

Run against a real SQLite database rather than a mocked session, because what
matters is what the insert leaves in the table: a replayed or overlapping export
stores each record once and reports the rest as duplicates, without failing.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.adapters.telemetry_storage_adapter import ERASURE_MARGIN, DatabaseTelemetryStorageAdapter
from gateway.models.usage import AgentTelemetry, AgentTelemetryUserBarrier
from gateway.ports.telemetry_storage_port import TelemetryFilter, TelemetryRecord

_TS = datetime(2026, 8, 6, tzinfo=UTC)
# When the exports were let in: after any erasure the tests below make, by more than its margin.
_ADMITTED = datetime.now(UTC) + timedelta(hours=1)


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
            admitted_at=_ADMITTED,
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
            admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),)
        )

        row = (await db.execute(select(AgentTelemetry))).scalar_one()
        assert (row.api_key_id, row.user_id, row.name) == ("key-1", "alice", "tool_result")


@pytest.mark.asyncio
async def test_record_keys_duplicates_by_source() -> None:
    """The same dedup key from two sources is two records."""
    async with _session() as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(
            admitted_at=_ADMITTED,
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
        await adapter.record(admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=records)

        result = await adapter.record(admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=records)

        assert (result.accepted, result.duplicate) == (0, 3)
        assert len(await _stored(db)) == 3


@pytest.mark.asyncio
async def test_record_stores_only_the_new_part_of_an_overlapping_export() -> None:
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        await adapter.record(admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))

        result = await adapter.record(
            admitted_at=_ADMITTED,
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
            admitted_at=_ADMITTED,
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
            admitted_at=_ADMITTED,
            api_key_id="no-such-key",
            user_id="nobody",
            records=(_record("dedup-0"), _record("dedup-1")),
        )

        assert (result.accepted, result.duplicate) == (0, 2)
        assert await _stored(db) == []


@pytest.mark.asyncio
async def test_record_stores_nothing_for_an_empty_export() -> None:
    async with _session() as db:
        result = await DatabaseTelemetryStorageAdapter(db).record(
            admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=()
        )

        assert (result.accepted, result.duplicate, result.rejected) == (0, 0, 0)
        assert (await db.execute(select(func.count()).select_from(AgentTelemetry))).scalar_one() == 0


@pytest.mark.asyncio
async def test_adapter_without_a_session_refuses_rather_than_storing_nothing() -> None:
    """Every surface behind this port has a session, so a session-less
    request is a wiring mistake and must not look like a successful no-op."""
    adapter = DatabaseTelemetryStorageAdapter(None)

    with pytest.raises(RuntimeError, match="database session"):
        await adapter.record(admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))
    with pytest.raises(RuntimeError, match="database session"):
        await adapter.count(filters=TelemetryFilter())


@pytest.mark.asyncio
async def test_record_refuses_an_export_admitted_before_its_users_erasure() -> None:
    """An export let in before the erasure and written after it stores nothing, and says so."""
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        admitted_at = datetime.now(UTC)
        await adapter.purge_user(user_id="alice")

        result = await adapter.record(
            admitted_at=admitted_at,
            api_key_id="key-1",
            user_id="alice",
            records=(_record("dedup-0"), _record("dedup-1", "codex")),
        )

        assert (result.accepted, result.duplicate, result.rejected) == (0, 0, 2)
        assert await _stored(db) == []


@pytest.mark.asyncio
async def test_record_refuses_an_export_admitted_within_the_margin_after_the_erasure() -> None:
    """The margin absorbs two replicas' clocks disagreeing about which came first."""
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        await adapter.purge_user(user_id="alice")

        result = await adapter.record(
            admitted_at=datetime.now(UTC) + ERASURE_MARGIN / 2,
            api_key_id="key-1",
            user_id="alice",
            records=(_record("dedup-0"),),
        )

        assert result.rejected == 1
        assert await _stored(db) == []


@pytest.mark.asyncio
async def test_record_stores_an_export_admitted_after_the_erasure() -> None:
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        await adapter.purge_user(user_id="alice")

        result = await adapter.record(
            admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),)
        )

        assert (result.accepted, result.rejected) == (1, 0)


@pytest.mark.asyncio
async def test_an_erasure_refuses_only_its_own_users_exports() -> None:
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        admitted_at = datetime.now(UTC)
        await adapter.purge_user(user_id="alice")

        result = await adapter.record(
            admitted_at=admitted_at, api_key_id="key-1", user_id="bob", records=(_record("dedup-0"),)
        )

        assert (result.accepted, result.rejected) == (1, 0)


@pytest.mark.asyncio
async def test_purge_user_deletes_the_users_rows_and_records_when() -> None:
    async with _session() as db:
        adapter = DatabaseTelemetryStorageAdapter(db)
        await adapter.record(admitted_at=_ADMITTED, api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))
        await adapter.record(admitted_at=_ADMITTED, api_key_id="key-2", user_id="bob", records=(_record("dedup-1"),))
        before = datetime.now(UTC)

        assert await adapter.purge_user(user_id="alice") == 1

        assert await _stored(db) == [("claude_code", "dedup-1")]
        erased_at = (
            await db.execute(
                select(AgentTelemetryUserBarrier.erased_at).where(AgentTelemetryUserBarrier.user_id == "alice")
            )
        ).scalar_one()
        assert erased_at is not None
        assert erased_at >= before


@pytest.mark.asyncio
async def test_purge_user_of_a_user_with_no_row_erases_nothing() -> None:
    """With no user row, no telemetry names them and the foreign key refuses any that would."""
    async with _session(foreign_keys=True) as db:
        assert await DatabaseTelemetryStorageAdapter(db).purge_user(user_id="nobody") == 0
