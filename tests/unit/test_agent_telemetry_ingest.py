"""Unit coverage for agent_telemetry_service.ingest()'s batch-insert path (FR-008)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from gateway.models.entities import APIKey
from gateway.services.agent_telemetry_service import TelemetryRecord, ingest

_TS = datetime(2026, 8, 6, tzinfo=UTC)


def _record(dedup_key: str) -> TelemetryRecord:
    return TelemetryRecord(
        name="tool_result",
        timestamp=_TS,
        source="claude_code",
        dedup_key=dedup_key,
        tool_name="Bash",
    )


def _user_exists_result(user_id: str) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = user_id
    return result


@pytest.mark.asyncio
async def test_ingest_batches_non_colliding_records_in_one_bulk_insert() -> None:
    """N non-colliding records issue one add_all + commit(), not N nested savepoints."""
    db = AsyncMock()
    db.add_all = MagicMock()
    db.execute.return_value = _user_exists_result("alice")

    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")
    records = [_record(f"dedup-{i}") for i in range(5)]

    result = await ingest(db, records, api_key=api_key)

    assert result.accepted == 5
    assert result.duplicate == 0
    db.add_all.assert_called_once()
    assert len(db.add_all.call_args[0][0]) == 5
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_retries_only_survivors_after_a_collision() -> None:
    """A collision with an already-stored row is reported as a duplicate, not a
    failure, and the retry bulk-inserts only the surviving records."""
    db = AsyncMock()
    db.add_all = MagicMock()

    existing_result = MagicMock()
    existing_result.scalars.return_value.all.return_value = ["dedup-0"]
    db.execute.side_effect = [_user_exists_result("alice"), existing_result]
    db.commit.side_effect = [IntegrityError("insert", {}, Exception("dup")), None]

    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")
    records = [_record("dedup-0"), _record("dedup-1")]

    result = await ingest(db, records, api_key=api_key)

    assert result.accepted == 1
    assert result.duplicate == 1
    assert db.add_all.call_count == 2
    survivors = db.add_all.call_args_list[1][0][0]
    assert [row.dedup_key for row in survivors] == ["dedup-1"]
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_reports_all_duplicates_when_nothing_survives_the_first_collision() -> None:
    """Every record in the batch already exists: no retry insert is attempted,
    the whole batch is reported as duplicate."""
    db = AsyncMock()
    db.add_all = MagicMock()

    existing_result = MagicMock()
    existing_result.scalars.return_value.all.return_value = ["dedup-0"]
    db.execute.side_effect = [_user_exists_result("alice"), existing_result]
    db.commit.side_effect = [IntegrityError("insert", {}, Exception("dup"))]

    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")
    records = [_record("dedup-0")]

    result = await ingest(db, records, api_key=api_key)

    assert result.accepted == 0
    assert result.duplicate == 1
    db.add_all.assert_called_once()
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_falls_back_to_row_at_a_time_after_a_second_collision() -> None:
    """When the retry bulk insert also collides (a still-racing writer landed more
    rows in the window), the still-colliding remainder is inserted row-at-a-time
    and only the rows that still fail are reported as duplicate."""
    db = AsyncMock()
    db.add_all = MagicMock()
    db.add = MagicMock()

    existing_result = MagicMock()
    existing_result.scalars.return_value.all.return_value = ["dedup-0"]
    db.execute.side_effect = [_user_exists_result("alice"), existing_result]
    db.commit.side_effect = [
        IntegrityError("insert", {}, Exception("dup")),  # whole-batch insert
        IntegrityError("insert", {}, Exception("dup")),  # retry bulk insert of survivors
        None,  # row-at-a-time: dedup-1 succeeds
        IntegrityError("insert", {}, Exception("dup")),  # row-at-a-time: dedup-2 still collides
    ]

    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")
    records = [_record("dedup-0"), _record("dedup-1"), _record("dedup-2")]

    result = await ingest(db, records, api_key=api_key)

    assert result.accepted == 1
    assert result.duplicate == 2
    assert db.add_all.call_count == 2
    assert db.add.call_count == 2
    assert db.rollback.await_count == 3


@pytest.mark.asyncio
async def test_ingest_drops_in_export_repeats_before_the_insert() -> None:
    """A record repeated inside one export never reaches the database.

    Letting it through fails the whole bulk insert on the uniqueness constraint and
    drops the batch into the row-at-a-time fallback, so a single repeated record
    turns an N-row export into N commits.
    """
    db = AsyncMock()
    db.add_all = MagicMock()
    db.execute.return_value = _user_exists_result("alice")

    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")
    records = [_record("dedup-0"), _record("dedup-1"), _record("dedup-0")]

    result = await ingest(db, records, api_key=api_key)

    assert result.accepted == 2
    assert result.duplicate == 1
    db.add_all.assert_called_once()
    assert [row.dedup_key for row in db.add_all.call_args[0][0]] == ["dedup-0", "dedup-1"]
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()
