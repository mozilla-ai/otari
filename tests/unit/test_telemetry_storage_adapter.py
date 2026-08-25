"""Unit coverage for the core TelemetryStoragePort adapter's batch-insert path.

The escalation this exercises (one bulk insert, then survivors only, then
row-at-a-time) is what turns a uniqueness collision into a reported duplicate
rather than a failed export, so it is asserted on the round trips it issues and
not only on the counts it returns.
"""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
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


def _adapter(db: AsyncMock) -> DatabaseTelemetryStorageAdapter:
    return DatabaseTelemetryStorageAdapter(cast(AsyncSession, db))


def _existing(*dedup_keys: str) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(dedup_keys)
    return result


@pytest.mark.asyncio
async def test_record_batches_non_colliding_records_in_one_bulk_insert() -> None:
    """N non-colliding records issue one add_all + commit(), not N nested savepoints."""
    db = AsyncMock()
    db.add_all = MagicMock()

    result = await _adapter(db).record(
        api_key_id="key-1", user_id="alice", records=tuple(_record(f"dedup-{i}") for i in range(5))
    )

    assert result.accepted == 5
    assert result.duplicate == 0
    db.add_all.assert_called_once()
    assert len(db.add_all.call_args[0][0]) == 5
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_attributes_every_row_to_the_exporting_key_and_user() -> None:
    """Attribution comes from the port call, not from the record."""
    db = AsyncMock()
    db.add_all = MagicMock()

    await _adapter(db).record(api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))

    row = db.add_all.call_args[0][0][0]
    assert (row.api_key_id, row.user_id) == ("key-1", "alice")
    # The dedup-only fields are not columns; the key derived from them is what is stored.
    assert row.dedup_key == "dedup-0"
    assert not hasattr(row, "tool_use_id")


@pytest.mark.asyncio
async def test_record_batches_each_source_separately() -> None:
    """The uniqueness constraint is (source, dedup_key), so batches are per source."""
    db = AsyncMock()
    db.add_all = MagicMock()

    result = await _adapter(db).record(
        api_key_id="key-1",
        user_id="alice",
        records=(_record("dedup-0", "claude_code"), _record("dedup-0", "codex")),
    )

    assert result.accepted == 2
    assert db.add_all.call_count == 2


@pytest.mark.asyncio
async def test_record_retries_only_survivors_after_a_collision() -> None:
    """A collision with an already-stored row is reported as a duplicate, not a
    failure, and the retry bulk-inserts only the surviving records."""
    db = AsyncMock()
    db.add_all = MagicMock()
    db.execute.side_effect = [_existing("dedup-0")]
    db.commit.side_effect = [IntegrityError("insert", {}, Exception("dup")), None]

    result = await _adapter(db).record(
        api_key_id="key-1", user_id="alice", records=(_record("dedup-0"), _record("dedup-1"))
    )

    assert result.accepted == 1
    assert result.duplicate == 1
    assert db.add_all.call_count == 2
    survivors = db.add_all.call_args_list[1][0][0]
    assert [row.dedup_key for row in survivors] == ["dedup-1"]
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_reports_all_duplicates_when_nothing_survives_the_first_collision() -> None:
    """Every record in the batch already exists: no retry insert is attempted,
    the whole batch is reported as duplicate."""
    db = AsyncMock()
    db.add_all = MagicMock()
    db.execute.side_effect = [_existing("dedup-0")]
    db.commit.side_effect = [IntegrityError("insert", {}, Exception("dup"))]

    result = await _adapter(db).record(api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))

    assert result.accepted == 0
    assert result.duplicate == 1
    db.add_all.assert_called_once()
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_falls_back_to_row_at_a_time_after_a_second_collision() -> None:
    """When the retry bulk insert also collides (a still-racing writer landed more
    rows in the window), the still-colliding remainder is inserted row-at-a-time
    and only the rows that still fail are reported as duplicate."""
    db = AsyncMock()
    db.add_all = MagicMock()
    db.add = MagicMock()
    db.execute.side_effect = [_existing("dedup-0")]
    db.commit.side_effect = [
        IntegrityError("insert", {}, Exception("dup")),  # whole-batch insert
        IntegrityError("insert", {}, Exception("dup")),  # retry bulk insert of survivors
        None,  # row-at-a-time: dedup-1 succeeds
        IntegrityError("insert", {}, Exception("dup")),  # row-at-a-time: dedup-2 still collides
    ]

    result = await _adapter(db).record(
        api_key_id="key-1",
        user_id="alice",
        records=(_record("dedup-0"), _record("dedup-1"), _record("dedup-2")),
    )

    assert result.accepted == 1
    assert result.duplicate == 2
    assert db.add_all.call_count == 2
    assert db.add.call_count == 2
    assert db.rollback.await_count == 3


@pytest.mark.asyncio
async def test_record_drops_in_export_repeats_before_the_insert() -> None:
    """A record repeated inside one export never reaches the database.

    Letting it through fails the whole bulk insert on the uniqueness constraint and
    drops the batch into the row-at-a-time fallback, so a single repeated record
    turns an N-row export into N commits.
    """
    db = AsyncMock()
    db.add_all = MagicMock()

    result = await _adapter(db).record(
        api_key_id="key-1",
        user_id="alice",
        records=(_record("dedup-0"), _record("dedup-1"), _record("dedup-0")),
    )

    assert result.accepted == 2
    assert result.duplicate == 1
    db.add_all.assert_called_once()
    assert [row.dedup_key for row in db.add_all.call_args[0][0]] == ["dedup-0", "dedup-1"]
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_stores_nothing_for_an_empty_export() -> None:
    db = AsyncMock()
    db.add_all = MagicMock()

    result = await _adapter(db).record(api_key_id="key-1", user_id="alice", records=())

    assert (result.accepted, result.duplicate, result.rejected) == (0, 0, 0)
    db.add_all.assert_not_called()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_adapter_without_a_session_refuses_rather_than_storing_nothing() -> None:
    """Every surface behind this port is standalone-only, so a session-less
    request is a wiring mistake and must not look like a successful no-op."""
    adapter = DatabaseTelemetryStorageAdapter(None)

    with pytest.raises(RuntimeError, match="database session"):
        await adapter.record(api_key_id="key-1", user_id="alice", records=(_record("dedup-0"),))
    with pytest.raises(RuntimeError, match="database session"):
        await adapter.count(filters=TelemetryFilter())
