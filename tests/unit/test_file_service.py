"""The files service's error paths, which the HTTP tests cannot reach.

Covers what happens when the store and the row disagree: an upload whose row
will not land, one that carries nothing, one past the deployment's ceiling, and
a page token this gateway never issued.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import (
    EmptyUploadError,
    FilesDisabledError,
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


class _FakeUnitOfWork:
    """Enough of a Unit of Work for the service: a block that opens and closes."""

    async def __aenter__(self) -> _FakeUnitOfWork:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubFiles:
    """The file repository as the service uses it, answering from memory."""

    def __init__(self, *, add_error: Exception | None = None, rows: list[FileObject] | None = None) -> None:
        self._add_error = add_error
        self._rows = rows or []
        self.added: list[FileObject] = []

    async def add(self, record: FileObject) -> FileObject:
        if self._add_error is not None:
            raise self._add_error
        self.added.append(record)
        return record

    async def page(self, query: FilePageQuery) -> list[FileObject]:
        return self._rows[: query.limit]

    async def any_owned(self, *args: object, **kwargs: object) -> FileObject | None:
        return self._rows[0] if self._rows else None


def _service(store: _MemoryStore, files: _StubFiles, **config: Any) -> FileService:
    async def _default_workspace() -> uuid.UUID:
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
async def test_a_deployment_that_does_not_serve_files_refuses_every_verb() -> None:
    service = _service(_MemoryStore(), _StubFiles(), files_enabled=False)

    with pytest.raises(FilesDisabledError):
        await service.store(_upload(b"a,b\n"))
    with pytest.raises(FilesDisabledError):
        await service.stored_file("file-1", FileScope(user_id="u1"))
