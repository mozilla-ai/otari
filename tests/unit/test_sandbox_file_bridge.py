"""The bridge between the ``/v1/files`` store and a sandbox session.

Covers what the sandbox backend's own tests stub out: that a produced file is
streamed into the store, that an empty one leaves nothing behind, and that a
row which fails to land takes its blob with it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.services import file_service
from gateway.services.files import SandboxFileBridge


class _MemoryStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    async def put(self, file_id: str, data: bytes) -> str:
        self.blobs[file_id] = data
        return file_id

    async def get(self, storage_ref: str) -> bytes:
        return self.blobs[storage_ref]

    async def put_stream(self, file_id: str, chunks: AsyncIterator[bytes]) -> tuple[str, int]:
        data = bytearray()
        async for chunk in chunks:
            data.extend(chunk)
        self.blobs[file_id] = bytes(data)
        return file_id, len(data)

    async def get_stream(self, storage_ref: str) -> Any:
        yield self.blobs[storage_ref]

    async def delete(self, storage_ref: str) -> None:
        self.blobs.pop(storage_ref, None)


class _FakeDb:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, record: Any) -> None:
        self.added.append(record)

    async def flush(self) -> None:
        return None


class _FakeUnitOfWork:
    """Enough of a Unit of Work for ``session_for``: the session and an open block."""

    def __init__(self, db: Any) -> None:
        self._session = db
        self._depth = 0

    async def __aenter__(self) -> _FakeUnitOfWork:
        self._depth += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._depth -= 1


class _FailingUnitOfWork(_FakeUnitOfWork):
    """A Unit of Work whose commit fails the way a connect timeout does: a bare ``TimeoutError``."""

    async def __aexit__(self, *exc: object) -> None:
        raise TimeoutError("connect timed out")


class _CommittingUnitOfWork(_FakeUnitOfWork):
    pass


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def _bridge(store: _MemoryStore, uow: Any = None, **config: Any) -> SandboxFileBridge:
    return SandboxFileBridge(
        file_store=store,
        config=GatewayConfig(**config),
        uow=cast(UnitOfWork, uow if uow is not None else _CommittingUnitOfWork(_FakeDb())),
        user_id="u1",
        workspace_id=uuid.uuid4(),
        inputs=[],
    )


@pytest.mark.asyncio
async def test_store_output_streams_the_file_in_and_writes_its_row() -> None:
    store = _MemoryStore()
    db = _FakeDb()

    file_id = await _bridge(store, _CommittingUnitOfWork(db)).store_output("chart.png", _chunks(b"\x89PNG", b"..."))

    assert file_id is not None and file_id.startswith("file-")
    assert store.blobs == {file_id: b"\x89PNG..."}
    (record,) = db.added
    assert (record.id, record.filename, record.bytes, record.purpose) == (
        file_id,
        "chart.png",
        7,
        file_service.CODE_EXECUTION_OUTPUT_PURPOSE,
    )


@pytest.mark.asyncio
async def test_an_empty_output_leaves_no_blob_and_no_row() -> None:
    store = _MemoryStore()
    db = _FakeDb()

    assert await _bridge(store, _CommittingUnitOfWork(db)).store_output("empty.txt", _chunks()) is None
    assert store.blobs == {}
    assert db.added == []


@pytest.mark.asyncio
async def test_a_row_that_fails_to_land_takes_its_blob_with_it() -> None:
    store = _MemoryStore()

    with pytest.raises(TimeoutError):
        await _bridge(store, _FailingUnitOfWork(_FakeDb())).store_output("out.csv", _chunks(b"a,b\n"))
    # Nothing references the bytes any more, and the sweep only sees rows, so
    # leaving them would be a leak nothing reclaims.
    assert store.blobs == {}


def test_the_output_budget_never_exceeds_the_upload_cap() -> None:
    store = _MemoryStore()
    assert _bridge(store, files_output_max_bytes=1 << 30, files_max_bytes=1 << 20).max_output_bytes == 1 << 20
    assert _bridge(store, files_output_max_bytes=1 << 10, files_max_bytes=1 << 20).max_output_bytes == 1 << 10
    assert _bridge(store, files_output_max_files=3).max_output_files == 3
