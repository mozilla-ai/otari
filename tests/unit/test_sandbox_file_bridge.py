"""The bridge between the ``/v1/files`` store and a sandbox session.

Covers what the sandbox backend's own tests stub out: that a produced file is
streamed into the store, that its row is recorded before its bytes, and that a
row which will not land stops the write rather than following it. Also covers
copying the files a provider's own sandbox produced, with the provider's client
stubbed.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Collection
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from any_llm import LLMProvider
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import FileOverBudgetError, FileStorageError, ProviderFileUnavailableError
from gateway.models.files import FileObject
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.repositories.files import FileProviderCopyRepository, FileRepositories, FileRepository
from gateway.services.files import (
    CODE_EXECUTION_OUTPUT_PURPOSE,
    FileBackends,
    FileService,
    ProviderFile,
    SandboxFileBridge,
)
from gateway.types.provider_account import ResolvedCredential


class _MemoryStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    async def allocate(self, file_id: str) -> str:
        return file_id

    async def put(self, storage_ref: str, data: bytes) -> None:
        self.blobs[storage_ref] = data

    async def get(self, storage_ref: str) -> bytes:
        return self.blobs[storage_ref]

    async def put_stream(self, storage_ref: str, chunks: AsyncIterator[bytes]) -> int:
        data = bytearray()
        async for chunk in chunks:
            data.extend(chunk)
        self.blobs[storage_ref] = bytes(data)
        return len(data)

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


class _AcknowledgmentLostUnitOfWork(_FakeUnitOfWork):
    """The database commits, but the client never receives the acknowledgment."""

    def __init__(self, db: Any) -> None:
        super().__init__(db)
        self.committed: list[Any] = []

    async def __aexit__(self, *exc: object) -> None:
        await super().__aexit__(*exc)
        self.committed.extend(self._session.added)
        raise ConnectionError("commit acknowledgment lost")


class _CancellingUnitOfWork(_FakeUnitOfWork):
    """A Unit of Work cancelled as its block ends, so the commit's outcome is unknown."""

    async def __aexit__(self, *exc: object) -> None:
        raise asyncio.CancelledError


class _CommittingUnitOfWork(_FakeUnitOfWork):
    pass


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


class _StubFiles(FileRepository):
    """The repository's real write path over a fake session, with the ID lookup answered from a set.

    ``existing_ids`` runs a query the fake session cannot serve.
    """

    def __init__(
        self,
        db: Any,
        *,
        known: Collection[str] = (),
        error: Exception | None = None,
        record_error: BaseException | None = None,
    ) -> None:
        super().__init__(cast(UnitOfWork, db))
        self._rows: list[FileObject] = db.added
        self._known = set(known)
        self._error = error
        self._record_error = record_error

    async def existing_ids(self, file_ids: Collection[str]) -> set[str]:
        if self._error is not None:
            raise self._error
        return set(file_ids) & (self._known | {row.id for row in self._rows})

    async def add(self, record: FileObject) -> None:
        if self._record_error is not None:
            raise self._record_error
        await super().add(record)

    async def mark_stored(self, record: FileObject, size: int) -> bool:
        record.bytes = size
        record.pending_since = None
        return True

    async def remove_all(self, file_ids: Collection[str]) -> None:
        gone = set(file_ids)
        self._rows[:] = [row for row in self._rows if row.id not in gone]


def _bridge(
    store: _MemoryStore,
    uow: Any = None,
    *,
    known: Collection[str] = (),
    lookup_error: Exception | None = None,
    record_error: BaseException | None = None,
    **config: Any,
) -> SandboxFileBridge:
    uow = uow if uow is not None else _CommittingUnitOfWork(_FakeDb())
    settings = GatewayConfig(**config)
    backends = FileBackends(storage=store, provider_files=cast(ProviderFilePort, _PROVIDER_FILES))
    return SandboxFileBridge(
        backends=backends,
        config=settings,
        files=FileService(
            cast(UnitOfWork, uow),
            FileRepositories(
                files=_StubFiles(uow._session, known=known, error=lookup_error, record_error=record_error),
                provider_copies=cast(FileProviderCopyRepository, None),
            ),
            backends,
            settings,
            AsyncMock(side_effect=AssertionError("Workspace resolution is not expected")),
        ),
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
    (record,) = db.added
    assert store.blobs == {record.storage_ref: b"\x89PNG..."}
    assert (record.id, record.filename, record.bytes, record.purpose) == (
        file_id,
        "chart.png",
        7,
        CODE_EXECUTION_OUTPUT_PURPOSE,
    )


@pytest.mark.asyncio
async def test_an_empty_output_gives_its_reservation_back() -> None:
    store = _MemoryStore()
    db = _FakeDb()

    assert await _bridge(store, _CommittingUnitOfWork(db)).store_output("empty.txt", _chunks()) is None
    assert store.blobs == {}
    assert db.added == []


@pytest.mark.asyncio
async def test_an_output_whose_row_will_not_land_writes_no_bytes() -> None:
    store = _MemoryStore()

    with pytest.raises(FileStorageError):
        await _bridge(store, record_error=TimeoutError()).store_output("out.csv", _chunks(b"a,b\n"))
    assert store.blobs == {}


@pytest.mark.asyncio
async def test_a_lost_reservation_acknowledgment_writes_no_bytes() -> None:
    """The row may have landed pending, which the sweep reclaims, so no bytes follow it."""
    store = _MemoryStore()
    uow = _AcknowledgmentLostUnitOfWork(_FakeDb())

    with pytest.raises(ConnectionError, match="commit acknowledgment lost"):
        await _bridge(store, uow).store_output("out.csv", _chunks(b"a,b\n"))

    (record,) = uow.committed
    assert record.pending_since is not None
    assert store.blobs == {}


def test_the_output_budget_never_exceeds_the_upload_cap() -> None:
    store = _MemoryStore()
    assert _bridge(store, files_output_max_bytes=1 << 30, files_max_bytes=1 << 20).max_output_bytes == 1 << 20
    assert _bridge(store, files_output_max_bytes=1 << 10, files_max_bytes=1 << 20).max_output_bytes == 1 << 10
    assert _bridge(store, files_output_max_files=3).max_output_files == 3


class _FailingBlock(_FakeUnitOfWork):
    """Every block up to ``failing`` succeeds, and that one has an unknown outcome."""

    def __init__(self, db: Any, failing: int) -> None:
        super().__init__(db)
        self._failing = failing
        self._blocks = 0

    async def __aexit__(self, *exc: object) -> None:
        await super().__aexit__(*exc)
        self._blocks += 1
        if self._blocks == self._failing:
            raise TimeoutError("connect timed out")


class _StubProviderClient:
    """Serves ``files`` by ID the way a provider file session does, budget included."""

    def __init__(self, files: dict[str, bytes | Exception], delay: float = 0.0) -> None:
        self._files = files
        self._delay = delay
        self.reads: list[str] = []
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def filename_of(self, file_id: str) -> str | None:
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


class _StubProviderFiles:
    """The provider file port, reaching Anthropic and OpenAI through ``client`` once a test sets one."""

    def __init__(self) -> None:
        self.client: _StubProviderClient | None = None
        self.opened: list[tuple[LLMProvider, str]] = []

    def serves(self, provider: LLMProvider) -> bool:
        return provider in (LLMProvider.ANTHROPIC, LLMProvider.OPENAI)

    def open_session(self, *, provider: LLMProvider, instance: str, credential: ResolvedCredential) -> Any:
        self.opened.append((provider, instance))
        if self.client is None:
            raise LookupError("no provider files stubbed")
        return self.client


_PROVIDER_FILES = _StubProviderFiles()


def _stub_provider(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes | Exception],
    delay: float = 0.0,
) -> _StubProviderClient:
    client = _StubProviderClient(files, delay)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(_PROVIDER_FILES, "client", client)
    return client


async def _copy(bridge: SandboxFileBridge, *file_ids: str) -> None:
    await bridge.copy_provider_files(
        [ProviderFile(file_id=file_id) for file_id in file_ids], provider="anthropic", provider_instance="anthropic-eu"
    )


@pytest.mark.asyncio
async def test_a_provider_file_is_copied_under_the_providers_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _stub_provider(monkeypatch, {"file_01chart": b"\x89PNG..."})
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
        CODE_EXECUTION_OUTPUT_PURPOSE,
    )
    # The blob key is Otari's own, never the provider's ID.
    assert record.storage_ref != "file_01chart"
    assert store.blobs == {record.storage_ref: b"\x89PNG..."}
    # The client holds the provider connection, so the copy owns closing it.
    assert client.closed


@pytest.mark.asyncio
async def test_a_database_failure_setting_up_the_copy_still_releases_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _stub_provider(monkeypatch, {"file_01chart": b"\x89PNG..."})

    bridge = _bridge(_MemoryStore(), _CommittingUnitOfWork(_FakeDb()), lookup_error=SQLAlchemyError())
    await _copy(bridge, "file_01chart")

    assert client.closed


@pytest.mark.asyncio
async def test_a_file_already_recorded_is_not_copied_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a", "file_01b": b"b"})
    db = _FakeDb()

    bridge = _bridge(_MemoryStore(), _CommittingUnitOfWork(db), known={"file_01a"})

    await _copy(bridge, "file_01a", "file_01b", "file_01b")

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
async def test_an_empty_provider_file_is_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty copy is a refusal here too, as it is on the other two write paths."""
    _stub_provider(monkeypatch, {"file_01empty": b""})
    store = _MemoryStore()
    db = _FakeDb()

    await _copy(_bridge(store, _CommittingUnitOfWork(db)), "file_01empty")

    assert db.added == []
    assert store.blobs == {}


@pytest.mark.asyncio
async def test_a_file_a_copy_could_not_take_is_left_for_a_later_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed copy gives its row back, so the ID it claimed does not block the next request."""
    gone = ProviderFileUnavailableError("anthropic could not serve file file_01a")
    files: dict[str, bytes | Exception] = {"file_01a": gone}
    _stub_provider(monkeypatch, files)
    store = _MemoryStore()
    db = _FakeDb()

    await _copy(_bridge(store, _CommittingUnitOfWork(db)), "file_01a")
    assert db.added == []

    files["file_01a"] = b"chart"
    await _copy(_bridge(store, _CommittingUnitOfWork(db)), "file_01a")

    assert [record.id for record in db.added] == ["file_01a"]
    assert list(store.blobs.values()) == [b"chart"]


@pytest.mark.asyncio
async def test_a_copied_file_whose_row_will_not_land_writes_no_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a"})
    store = _MemoryStore()

    await _copy(_bridge(store, record_error=TimeoutError()), "file_01a")

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_a_copy_whose_reservation_is_uncertain_writes_no_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a"})
    store = _MemoryStore()

    await _copy(_bridge(store, _FailingBlock(_FakeDb(), failing=2)), "file_01a")

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_a_copy_whose_stamp_is_uncertain_keeps_its_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """The row may have been served, so the bytes stay rather than stranding it."""
    _stub_provider(monkeypatch, {"file_01a": b"a"})
    store = _MemoryStore()

    await _copy(_bridge(store, _FailingBlock(_FakeDb(), failing=3)), "file_01a")

    assert list(store.blobs.values()) == [b"a"]


@pytest.mark.asyncio
async def test_no_credential_copies_nothing_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch, {"file_01a": b"a"})
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
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
    monkeypatch.setattr("gateway.services.files._sandbox_bridge._PROVIDER_COPY_SECONDS", 0.05)
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


@pytest.mark.asyncio
async def test_a_cancelled_reservation_writes_no_bytes() -> None:
    """Cancelled as the reservation commits, so its outcome is unknown and no bytes follow."""
    store = _MemoryStore()

    with pytest.raises(asyncio.CancelledError):
        await _bridge(store, _CancellingUnitOfWork(_FakeDb())).store_output("out.csv", _chunks(b"a,b\n"))

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_an_output_cancelled_inside_its_reservation_writes_no_bytes() -> None:
    """The other side of the previous test: cancelled inside the block, no row can have landed."""
    store = _MemoryStore()
    bridge = _bridge(store, _CommittingUnitOfWork(_FakeDb()), record_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await bridge.store_output("chart.png", _chunks(b"\x89PNG"))

    assert store.blobs == {}
