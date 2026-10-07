"""Each session's content is sealed with that session's own key, and a reader can always open what was sealed.

Two workers are two ``ContentKeys`` over one key table: the first to seal a
session's content mints its key, the other unwraps the stored one, and a purge
on either is seen by the other once its cached key is checked again.
"""

import asyncio
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.exc import IntegrityError

from gateway.adapters.data_key_adapter import SecretBoxDataKeys
from gateway.exceptions.traces_exceptions import TraceContentKeysUnavailableError
from gateway.ports.data_key_port import (
    DataKey,
    DataKeyContext,
    DataKeyContextMismatchError,
    DataKeyUnavailableError,
)
from gateway.ports.trace_storage_port import SpanRecord, StoredContent, TraceWrite
from gateway.services.traces._content_keys import ContentKeys, ContentKeyStoreError, ContentUnreadableError

_WORKSPACE = uuid.uuid4()


class _KeyTable:
    """``trace_content_keys`` as the key repository sees it: one row per session."""

    def __init__(self) -> None:
        self.rows: dict[tuple[uuid.UUID, str], SimpleNamespace] = {}
        self.finds = 0
        self.fail_add: Exception | None = None

    async def add_if_absent(self, *, workspace_id: uuid.UUID, trace_id: str, key_ref: str, wrapped: bytes) -> None:
        if self.fail_add is not None:
            raise self.fail_add
        self.rows.setdefault((workspace_id, trace_id), SimpleNamespace(wrapped=wrapped, created_at=datetime.now(UTC)))

    async def find(self, workspace_id: uuid.UUID, trace_id: str) -> SimpleNamespace | None:
        self.finds += 1
        return self.rows.get((workspace_id, trace_id))

    async def delete_for_workspace(self, workspace_id: uuid.UUID) -> int:
        doomed = [key for key in self.rows if key[0] == workspace_id]
        for key in doomed:
            del self.rows[key]
        return len(doomed)

    async def created_before(self, before: datetime, *, limit: int) -> list[tuple[uuid.UUID, str]]:
        return [key for key, row in self.rows.items() if row.created_at < before][:limit]

    async def delete_sessions(self, keys: set[tuple[uuid.UUID, str]]) -> int:
        for key in keys:
            self.rows.pop(key, None)
        return len(keys)


class _TraceTable:
    def __init__(self, stored: set[tuple[uuid.UUID, str]]) -> None:
        self.stored = stored

    async def existing(self, keys: list[tuple[uuid.UUID, str]]) -> set[tuple[uuid.UUID, str]]:
        return {key for key in keys if key in self.stored}


class _Uow:
    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


@asynccontextmanager
async def _open() -> AsyncIterator[Any]:
    yield _Uow()


class _CountingKeys(SecretBoxDataKeys):
    """Secret-box keys that count generations and unwraps, taking ``delay`` seconds to generate."""

    def __init__(self, delay: float = 0.0) -> None:
        self.generated = 0
        self.unwrapped = 0
        self.delay = delay

    async def generate(self, context: DataKeyContext) -> DataKey:
        self.generated += 1
        await asyncio.sleep(self.delay)
        return await super().generate(context)

    async def unwrap(self, wrapped: bytes, context: DataKeyContext) -> bytes:
        self.unwrapped += 1
        return await super().unwrap(wrapped, context)


@pytest.fixture(autouse=True)
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", Fernet.generate_key().decode())


def _worker(
    table: _KeyTable, data_keys: SecretBoxDataKeys | None = None, *, traces: _TraceTable | None = None, **kwargs: Any
) -> ContentKeys:
    def tables(_uow: Any) -> Any:
        return SimpleNamespace(keys=table, traces=traces or _TraceTable(set()))

    return ContentKeys(data_keys or SecretBoxDataKeys(), _open, tables, **kwargs)


def _write(trace_id: str = "t1") -> TraceWrite:
    span = SpanRecord(span_id="s1", kind="tool", origin="gateway", name="Bash", outcome="ok")
    return TraceWrite(
        workspace_id=_WORKSPACE, trace_id=trace_id, user_id=None, api_key_id=None, session_source="none", spans=(span,)
    )


async def _seal(keys: ContentKeys, trace_id: str = "t1") -> StoredContent:
    sealed = (await keys.seal(_write(trace_id), {"s1": {"result": "secret"}})).spans[0].sealed
    assert sealed is not None
    return StoredContent(workspace_id=_WORKSPACE, owner_user_id=None, sealed=sealed)


@pytest.mark.asyncio
async def test_each_session_gets_a_key_of_its_own() -> None:
    table = _KeyTable()
    keys = _worker(table)

    await _seal(keys, "t1")
    await _seal(keys, "t1")
    await _seal(keys, "t2")

    assert set(table.rows) == {(_WORKSPACE, "t1"), (_WORKSPACE, "t2")}


@pytest.mark.asyncio
async def test_a_second_worker_seals_with_the_key_the_first_stored() -> None:
    table = _KeyTable()
    first, second = _worker(table), _worker(table)

    by_first = await _seal(first)
    by_second = await _seal(second)

    assert len(table.rows) == 1
    for stored in (by_first, by_second):
        assert await first.open(stored, trace_id="t1", span_id="s1") == {"result": "secret"}


@pytest.mark.asyncio
async def test_a_purge_on_another_worker_is_seen_once_the_cached_key_is_rechecked() -> None:
    table = _KeyTable()
    sealing, purging = _worker(table, recheck_s=0.0), _worker(table)
    await _seal(sealing)

    await purging.destroy_workspace(_WORKSPACE)
    after = await _seal(sealing)

    assert (_WORKSPACE, "t1") in table.rows, "a fresh key is stored for content sealed after the purge"
    assert await purging.open(after, trace_id="t1", span_id="s1") == {"result": "secret"}


@pytest.mark.asyncio
async def test_a_cached_key_is_trusted_within_its_window_and_rechecked_without_kms_after() -> None:
    table = _KeyTable()
    data_keys = _CountingKeys()
    keys = _worker(table, data_keys, recheck_s=0.0)
    await _seal(keys)
    finds = table.finds

    await _seal(keys)

    assert table.finds == finds + 1, "past its window the cached key is checked against its row"
    assert data_keys.unwrapped == 0, "and confirming it needs no unwrap"


@pytest.mark.asyncio
async def test_concurrent_first_requests_mint_one_key() -> None:
    table = _KeyTable()
    data_keys = _CountingKeys(delay=0.05)
    keys = _worker(table, data_keys)

    await asyncio.gather(*(_seal(keys) for _ in range(10)))

    assert data_keys.generated == 1
    assert len(table.rows) == 1


@pytest.mark.asyncio
async def test_a_key_that_is_not_ready_in_time_fails_the_seal() -> None:
    keys = _worker(_KeyTable(), _CountingKeys(delay=1.0), deadline_s=0.05)

    with pytest.raises(ContentKeyStoreError):
        await _seal(keys)


@pytest.mark.asyncio
async def test_a_failed_key_insert_carries_no_bound_parameters() -> None:
    table = _KeyTable()
    table.fail_add = IntegrityError("INSERT INTO trace_content_keys", {"wrapped": "WRAPPED-KEY-BYTES"}, Exception())

    with pytest.raises(ContentKeyStoreError) as caught:
        await _seal(_worker(table))

    logged = "".join(traceback.format_exception(caught.value))
    assert "WRAPPED-KEY-BYTES" not in logged
    assert "IntegrityError" in logged


class _Unwrap(SecretBoxDataKeys):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def unwrap(self, wrapped: bytes, context: DataKeyContext) -> bytes:
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DataKeyContextMismatchError("other context"), ContentUnreadableError),
        (DataKeyUnavailableError("Set OTARI_SECRET_KEY"), TraceContentKeysUnavailableError),
        (RuntimeError("botocore said no"), TraceContentKeysUnavailableError),
    ],
    ids=["mismatch", "unavailable", "backend error"],
)
async def test_a_key_store_error_on_read_maps_to_unreadable_or_unavailable(
    error: Exception, expected: type[Exception]
) -> None:
    table = _KeyTable()
    stored = await _seal(_worker(table))

    with pytest.raises(expected):
        await _worker(table, _Unwrap(error)).open(stored, trace_id="t1", span_id="s1")


@pytest.mark.asyncio
async def test_content_does_not_open_under_another_span_or_session() -> None:
    table = _KeyTable()
    keys = _worker(table)
    stored = await _seal(keys, "t1")
    await _seal(keys, "t2")

    with pytest.raises(ContentUnreadableError):
        await keys.open(stored, trace_id="t1", span_id="another span")
    with pytest.raises(ContentUnreadableError):
        await keys.open(stored, trace_id="t2", span_id="s1")


@pytest.mark.asyncio
async def test_a_destroyed_session_key_makes_its_content_unreadable() -> None:
    table = _KeyTable()
    keys = _worker(table)
    stored = await _seal(keys)

    await keys.destroy_workspace(_WORKSPACE)

    with pytest.raises(ContentUnreadableError):
        await keys.open(stored, trace_id="t1", span_id="s1")


@pytest.mark.asyncio
async def test_the_sweep_destroys_only_keys_whose_session_is_gone() -> None:
    table = _KeyTable()
    keys = _worker(table, traces=_TraceTable({(_WORKSPACE, "kept")}))
    await _seal(keys, "kept")
    await _seal(keys, "dropped")

    destroyed = await keys.destroy_orphaned(datetime.now(UTC) + timedelta(seconds=1))

    assert destroyed == 1
    assert set(table.rows) == {(_WORKSPACE, "kept")}
