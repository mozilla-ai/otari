"""The bridge between the ``/v1/files`` store and a sandbox session.

Covers what the sandbox backend's own tests stub out: that a produced file is
streamed into the store, that an empty one leaves nothing behind, and that a
row which fails to land takes its blob with it. Also covers copying the files a
provider's own sandbox produced, with the provider's client stubbed.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Collection
from typing import Any, cast

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.services import file_service
from gateway.services.files import ProviderFile, SandboxFileBridge
from gateway.services.files.provider_files import FileOverBudgetError, ProviderFileUnavailableError


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


class _FailingSecondBlock(_FakeUnitOfWork):
    """A Unit of Work whose first block commits and whose later ones fail, as a row insert can."""

    def __init__(self, db: Any) -> None:
        super().__init__(db)
        self._blocks = 0

    async def __aexit__(self, *exc: object) -> None:
        await super().__aexit__(*exc)
        self._blocks += 1
        if self._blocks > 1:
            raise TimeoutError("connect timed out")


class _StubProviderClient:
    """Serves ``files`` by ID the way ``ProviderFileClient`` does, budget included."""

    provider = "anthropic"
    provider_instance = "anthropic-eu"

    def __init__(self, files: dict[str, bytes | Exception], delay: float = 0.0) -> None:
        self._files = files
        self._delay = delay
        self.reads: list[str] = []

    async def get_filename(self, file_id: str) -> str | None:
        if file_id == "file_01nameless":
            raise RuntimeError("metadata failed")
        return f"{file_id}.png"

    async def read(self, file: ProviderFile, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        self.reads.append(file.file_id)
        await asyncio.sleep(self._delay)
        body = self._files[file.file_id]
        if isinstance(body, Exception):
            raise body
        if len(body) > budget_bytes:
            raise FileOverBudgetError
        yield body


def _stub_provider(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes | Exception],
    known: Collection[str] = (),
    delay: float = 0.0,
) -> _StubProviderClient:
    client = _StubProviderClient(files, delay)
    monkeypatch.setattr(
        "gateway.services.files.sandbox_bridge.ProviderFileClient.for_run", lambda *args, **kwargs: client
    )

    async def _known(uow: Any, file_ids: Collection[str]) -> set[str]:
        return set(file_ids) & set(known)

    monkeypatch.setattr("gateway.services.files.sandbox_bridge.existing_file_ids", _known)
    return client


async def _copy(bridge: SandboxFileBridge, *file_ids: str) -> None:
    await bridge.copy_provider_files(
        [ProviderFile(file_id=file_id) for file_id in file_ids], provider="anthropic", provider_instance="anthropic-eu"
    )


@pytest.mark.asyncio
async def test_a_provider_file_is_copied_under_the_providers_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01chart": b"\x89PNG..."})
    store = _MemoryStore()
    db = _FakeDb()

    await _copy(_bridge(store, _CommittingUnitOfWork(db)), "file_01chart")

    (record,) = db.added
    assert (record.id, record.filename, record.mime_type, record.bytes) == (
        "file_01chart",
        "file_01chart.png",
        "image/png",
        7,
    )
    assert (record.provider, record.provider_instance, record.purpose) == (
        "anthropic",
        "anthropic-eu",
        file_service.CODE_EXECUTION_OUTPUT_PURPOSE,
    )
    # The blob key is Otari's own, never the provider's ID.
    assert record.storage_ref != "file_01chart"
    assert store.blobs == {record.storage_ref: b"\x89PNG..."}


@pytest.mark.asyncio
async def test_a_file_already_recorded_is_not_copied_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a", "file_01b": b"b"}, known={"file_01a"})
    db = _FakeDb()

    await _copy(_bridge(_MemoryStore(), _CommittingUnitOfWork(db)), "file_01a", "file_01b", "file_01b")

    assert [record.id for record in db.added] == ["file_01b"]


@pytest.mark.asyncio
async def test_one_reply_copies_at_most_max_output_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a", "file_01b": b"b"})
    db = _FakeDb()

    await _copy(_bridge(_MemoryStore(), _CommittingUnitOfWork(db), files_output_max_files=1), "file_01a", "file_01b")

    assert [record.id for record in db.added] == ["file_01a"]


@pytest.mark.asyncio
async def test_one_reply_copies_at_most_max_output_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"aaaa", "file_01big": b"bbbbbbb", "file_01c": b"cc"})
    store = _MemoryStore()
    db = _FakeDb()
    bridge = _bridge(store, _CommittingUnitOfWork(db), files_output_max_bytes=7)

    await _copy(bridge, "file_01a", "file_01big", "file_01c")

    # The big file would take the reply past its budget; the one after it still fits.
    assert [record.id for record in db.added] == ["file_01a", "file_01c"]
    assert sorted(store.blobs.values()) == [b"aaaa", b"cc"]


@pytest.mark.asyncio
async def test_the_caps_hold_across_calls_in_one_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream copies once per event that names a file, and every call draws on the same caps."""
    _stub_provider(monkeypatch, {"file_01a": b"aaaa", "file_01b": b"bbbb", "file_01c": b"c"})
    db = _FakeDb()
    bridge = _bridge(_MemoryStore(), _CommittingUnitOfWork(db), files_output_max_files=2, files_output_max_bytes=5)

    await _copy(bridge, "file_01a")
    await _copy(bridge, "file_01b")
    await _copy(bridge, "file_01c")

    # file_01b is past the bytes left, and file_01c is past the file count.
    assert [record.id for record in db.added] == ["file_01a"]


@pytest.mark.asyncio
async def test_a_file_the_provider_refuses_does_not_stop_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    gone = ProviderFileUnavailableError("anthropic could not serve file file_01gone")
    _stub_provider(monkeypatch, {"file_01gone": gone, "file_01b": b"b"})
    db = _FakeDb()

    await _copy(_bridge(_MemoryStore(), _CommittingUnitOfWork(db)), "file_01gone", "file_01b")

    assert [record.id for record in db.added] == ["file_01b"]


@pytest.mark.asyncio
async def test_a_copied_file_whose_row_fails_takes_its_blob_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a"})
    store = _MemoryStore()

    await _copy(_bridge(store, _FailingSecondBlock(_FakeDb())), "file_01a")

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_no_credential_copies_nothing_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_credential(*args: Any, **kwargs: Any) -> Any:
        raise LookupError("no credential configured for provider 'anthropic'")

    monkeypatch.setattr("gateway.services.files.sandbox_bridge.ProviderFileClient.for_run", _no_credential)
    store = _MemoryStore()

    await _copy(_bridge(store), "file_01a")

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_a_provider_otari_cannot_read_from_is_not_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a"})
    db = _FakeDb()

    await _bridge(_MemoryStore(), _CommittingUnitOfWork(db)).copy_provider_files(
        [ProviderFile(file_id="file_01a")], provider="nebius", provider_instance="nebius"
    )

    assert db.added == []


@pytest.mark.asyncio
async def test_a_file_cited_again_in_one_request_is_tried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream names one file in several events, and a failed copy must not be retried or recharged."""
    gone = ProviderFileUnavailableError("anthropic could not serve file file_01gone")
    client = _stub_provider(monkeypatch, {"file_01gone": gone, "file_01b": b"b"})
    db = _FakeDb()
    bridge = _bridge(_MemoryStore(), _CommittingUnitOfWork(db), files_output_max_files=2)

    for _ in range(3):
        await _copy(bridge, "file_01gone")
    await _copy(bridge, "file_01b")

    assert client.reads == ["file_01gone", "file_01b"]
    assert [record.id for record in db.added] == ["file_01b"]


@pytest.mark.asyncio
async def test_the_copy_stops_at_the_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.services.files.sandbox_bridge._PROVIDER_COPY_SECONDS", 0.05)
    _stub_provider(monkeypatch, {"file_01slow": b"a", "file_01next": b"b"}, delay=1.0)
    store = _MemoryStore()
    db = _FakeDb()

    await _copy(_bridge(store, _CommittingUnitOfWork(db)), "file_01slow", "file_01next")

    assert db.added == []
    assert store.blobs == {}


@pytest.mark.asyncio
async def test_a_blob_goes_when_its_row_cannot_be_built(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01nameless": b"a"})
    store = _MemoryStore()
    db = _FakeDb()

    await _copy(_bridge(store, _CommittingUnitOfWork(db)), "file_01nameless")

    assert db.added == []
    assert store.blobs == {}
