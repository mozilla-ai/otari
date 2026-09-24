"""The files service's error paths, which the HTTP tests cannot reach.

Covers what happens when the store and the row disagree: an upload whose row
will not land, one that carries nothing, one past the deployment's ceiling, a
page token this gateway never issued, and a file whose bytes will not read back
or will not delete.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import (
    EmptyUploadError,
    FileNotServedError,
    FileStorageError,
    UnknownPageCursorError,
    UploadTooLargeError,
)
from gateway.models.files import FileObject
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FilePageQuery, FileRepositories, FileRepository
from gateway.services.files import FileDialect, FileListing, FileScope, FileService, NewFile

_WORKSPACE = uuid.uuid4()
_DEFAULT_WORKSPACE = uuid.uuid4()


class _MemoryStore:
    def __init__(self, *, delete_error: Exception | None = None, read_error: Exception | None = None) -> None:
        self.blobs: dict[str, bytes] = {}
        self._delete_error = delete_error
        self._read_error = read_error

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
        if self._read_error is not None:
            raise self._read_error
        yield self.blobs[storage_ref]

    async def delete(self, storage_ref: str) -> None:
        if self._delete_error is not None:
            raise self._delete_error
        self.blobs.pop(storage_ref, None)


class _FakeUnitOfWork:
    """Enough of a Unit of Work for the service: a block that opens and closes."""

    async def __aenter__(self) -> _FakeUnitOfWork:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubFiles:
    """The file repository as the service uses it, answering from memory."""

    def __init__(
        self,
        *,
        add_error: Exception | None = None,
        delete_error: Exception | None = None,
        rows: list[FileObject] | None = None,
    ) -> None:
        self._add_error = add_error
        self._delete_error = delete_error
        self._rows = rows or []
        self.added: list[FileObject] = []
        self.discarded: list[str] = []

    async def add(self, record: FileObject) -> None:
        if self._add_error is not None:
            raise self._add_error
        self.added.append(record)

    async def page(self, query: FilePageQuery) -> list[FileObject]:
        return self._rows[: query.limit]

    async def any_owned(self, *args: object, **kwargs: object) -> FileObject | None:
        return self._rows[0] if self._rows else None

    async def live(self, file_id: str, *args: object, **kwargs: object) -> FileObject | None:
        return next((row for row in self._rows if row.id == file_id), None)

    async def soft_delete(self, record: FileObject, at: datetime) -> None:
        if self._delete_error is not None:
            raise self._delete_error
        self.discarded.append(record.id)


def _row(file_id: str = "file-1", *, storage_ref: str | None = "blob-1") -> FileObject:
    return FileObject(
        id=file_id,
        user_id="u1",
        workspace_id=_WORKSPACE,
        filename="report.csv",
        mime_type="text/csv",
        bytes=4,
        purpose="user_data",
        storage_ref=storage_ref,
        created_at=datetime.now(UTC),
    )


def _service(
    store: _MemoryStore, files: _StubFiles, *, workspace_error: BaseException | None = None, **config: Any
) -> FileService:
    async def _default_workspace() -> uuid.UUID:
        if workspace_error is not None:
            raise workspace_error
        return _DEFAULT_WORKSPACE

    return FileService(
        cast(UnitOfWork, _FakeUnitOfWork()),
        FileRepositories(files=cast(FileRepository, files)),
        cast(FileStoragePort, store),
        GatewayConfig(**config),
        _default_workspace,
    )


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def _upload(*parts: bytes, workspace_id: uuid.UUID | None = _WORKSPACE) -> NewFile:
    return NewFile(
        user_id="u1",
        workspace_id=workspace_id,
        filename="report.csv",
        content_type="text/csv",
        purpose="user_data",
        chunks=_chunks(*parts),
    )


@pytest.mark.asyncio
async def test_an_upload_whose_row_will_not_land_takes_its_blob_with_it() -> None:
    store = _MemoryStore()
    service = _service(store, _StubFiles(add_error=SQLAlchemyError()))

    with pytest.raises(FileStorageError):
        await service.store(_upload(b"a,b\n"))

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_an_empty_upload_is_refused_and_leaves_no_blob() -> None:
    store = _MemoryStore()

    with pytest.raises(EmptyUploadError):
        await _service(store, _StubFiles()).store(_upload())

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_an_upload_past_the_ceiling_is_refused() -> None:
    with pytest.raises(UploadTooLargeError):
        await _service(_MemoryStore(), _StubFiles(), files_max_bytes=4).store(_upload(b"12345"))


@pytest.mark.asyncio
async def test_a_master_key_upload_lands_in_the_default_workspace() -> None:
    files = _StubFiles()

    record = await _service(_MemoryStore(), files).store(_upload(b"a,b\n", workspace_id=None))

    assert record.workspace_id == _DEFAULT_WORKSPACE
    assert files.added == [record]


@pytest.mark.asyncio
async def test_a_page_token_this_gateway_did_not_issue_is_refused() -> None:
    listing = FileListing(
        scope=FileScope(user_id="u1", workspace_id=_WORKSPACE),
        dialect=FileDialect.ANTHROPIC,
        limit=10,
        cursor="not-a-token",
    )

    with pytest.raises(UnknownPageCursorError):
        await _service(_MemoryStore(), _StubFiles()).page(listing)


@pytest.mark.asyncio
async def test_a_cleanup_that_fails_does_not_replace_the_refusal_it_follows() -> None:
    """An orphaned blob is a smaller failure than losing why the upload was refused."""
    store = _MemoryStore(delete_error=OSError("read-only store"))

    with pytest.raises(EmptyUploadError):
        await _service(store, _StubFiles()).store(_upload())

    with pytest.raises(FileStorageError):
        await _service(store, _StubFiles(add_error=SQLAlchemyError())).store(_upload(b"a,b\n"))


@pytest.mark.asyncio
async def test_a_file_whose_row_will_not_change_is_still_served() -> None:
    files = _StubFiles(delete_error=SQLAlchemyError(), rows=[_row()])
    store = _MemoryStore()
    store.blobs["blob-1"] = b"a,b\n"

    with pytest.raises(FileStorageError):
        await _service(store, files).discard("file-1", FileScope(user_id="u1", workspace_id=_WORKSPACE))

    # The bytes stay, because the caller is still served the file.
    assert store.blobs == {"blob-1": b"a,b\n"}


@pytest.mark.asyncio
async def test_a_blob_that_will_not_delete_does_not_undo_a_completed_discard() -> None:
    files = _StubFiles(rows=[_row()])

    await _service(_MemoryStore(delete_error=OSError("read-only store")), files).discard(
        "file-1", FileScope(user_id="u1", workspace_id=_WORKSPACE)
    )

    assert files.discarded == ["file-1"]


@pytest.mark.asyncio
async def test_a_row_holding_no_bytes_is_not_served() -> None:
    files = _StubFiles(rows=[_row(storage_ref=None)])

    with pytest.raises(FileNotServedError):
        await _service(_MemoryStore(), files).content("file-1", FileScope(user_id="u1", workspace_id=_WORKSPACE))


@pytest.mark.asyncio
async def test_bytes_that_will_not_read_back_are_a_storage_failure() -> None:
    files = _StubFiles(rows=[_row()])
    store = _MemoryStore(read_error=OSError("blob missing"))

    with pytest.raises(FileStorageError):
        await _service(store, files).content("file-1", FileScope(user_id="u1", workspace_id=_WORKSPACE))


@pytest.mark.asyncio
async def test_an_upload_whose_workspace_will_not_resolve_takes_its_blob_with_it() -> None:
    """The workspace is resolved after the bytes are written, so its failure owns them too."""
    store = _MemoryStore()
    service = _service(store, _StubFiles(), workspace_error=RuntimeError("no default workspace"))

    with pytest.raises(RuntimeError, match="no default workspace"):
        await service.store(_upload(b"a,b\n", workspace_id=None))

    assert store.blobs == {}


@pytest.mark.asyncio
async def test_an_upload_cancelled_before_its_commit_takes_its_blob_with_it() -> None:
    """Cancellation inside the block is knowably pre-commit, so no row can be stranded."""
    store = _MemoryStore()
    service = _service(store, _StubFiles(), workspace_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await service.store(_upload(b"a,b\n", workspace_id=None))

    assert store.blobs == {}
