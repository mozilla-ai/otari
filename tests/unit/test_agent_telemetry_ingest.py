"""Unit coverage for agent_telemetry_service.ingest()'s caller-side gate (FR-008).

What the service still owns after telemetry storage moved behind a port: the
active-user gate, and handing everything that passes it to whichever store this
build bound. The batch-insert behavior it used to hold is covered in
``test_telemetry_storage_adapter.py``.
"""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import APIKey
from gateway.ports.telemetry_storage_port import IngestResult, TelemetryRecord, TelemetryStoragePort
from gateway.services.agent_telemetry_service import ingest

_TS = datetime(2026, 8, 6, tzinfo=UTC)


def _record(dedup_key: str) -> TelemetryRecord:
    return TelemetryRecord(
        name="tool_result",
        timestamp=_TS,
        source="claude_code",
        dedup_key=dedup_key,
        tool_name="Bash",
    )


def _db(user_id: str | None) -> AsyncSession:
    """A session whose active-user lookup resolves to ``user_id`` (None = gone)."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = user_id
    db = AsyncMock()
    db.execute.return_value = result
    return cast(AsyncSession, db)


def _storage(result: IngestResult) -> AsyncMock:
    storage = AsyncMock()
    storage.record.return_value = result
    return storage


@pytest.mark.asyncio
async def test_ingest_hands_the_export_to_storage_with_its_attribution() -> None:
    """The key and user the export authenticated as travel with the records."""
    storage = _storage(IngestResult(accepted=2))
    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")
    records = [_record("dedup-0"), _record("dedup-1")]

    result = await ingest(_db("alice"), records, api_key=api_key, storage=cast(TelemetryStoragePort, storage))

    assert result.accepted == 2
    storage.record.assert_awaited_once_with(api_key_id="key-1", user_id="alice", records=tuple(records))


@pytest.mark.asyncio
async def test_ingest_rejects_an_export_from_a_soft_deleted_user() -> None:
    """A soft-deleted user's exporter can still hold a live key; its events are
    rejected rather than stored, and storage is never asked."""
    storage = _storage(IngestResult())
    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")

    result = await ingest(
        _db(None), [_record("dedup-0")], api_key=api_key, storage=cast(TelemetryStoragePort, storage)
    )

    assert (result.accepted, result.rejected) == (0, 1)
    storage.record.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_rejects_an_export_from_a_key_with_no_user() -> None:
    storage = _storage(IngestResult())
    api_key = APIKey(id="key-1", user_id=None, key_hash="h")

    result = await ingest(
        _db("alice"), [_record("dedup-0")], api_key=api_key, storage=cast(TelemetryStoragePort, storage)
    )

    assert result.rejected == 1
    storage.record.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_short_circuits_an_empty_export_before_the_gate() -> None:
    """Nothing to store is not worth a user lookup, let alone a storage round trip."""
    storage = _storage(IngestResult())
    db = _db("alice")
    api_key = APIKey(id="key-1", user_id="alice", key_hash="h")

    result = await ingest(db, [], api_key=api_key, storage=cast(TelemetryStoragePort, storage))

    assert (result.accepted, result.duplicate, result.rejected) == (0, 0, 0)
    storage.record.assert_not_awaited()
    cast(AsyncMock, db).execute.assert_not_awaited()
