"""File consumers leave transactions and repository access to the service."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import FileStorageError
from gateway.models.files import FileObject
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FileRepositories, FileRepository
from gateway.services.files import FileService, NewOutput, SweepBatch, _sweeper


class _Transactions:
    def __init__(self, *, commit_error: BaseException | None = None) -> None:
        self.depth = 0
        self.blocks = 0
        self.commit_error = commit_error

    async def __aenter__(self) -> _Transactions:
        self.depth += 1
        self.blocks += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.depth -= 1
        if exc[0] is None and self.commit_error is not None:
            raise self.commit_error


def _service(uow: _Transactions, repo: Mock, store: Mock) -> FileService:
    return FileService(
        cast(UnitOfWork, uow),
        FileRepositories(files=repo),
        store,
        GatewayConfig(),
        AsyncMock(side_effect=AssertionError("Workspace resolution is not expected")),
    )


def _output() -> NewOutput:
    return NewOutput("file-1", "user-1", uuid.uuid4(), "out.txt", "text/plain", "user_data", None)


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def _store() -> Mock:
    store = Mock(spec=FileStoragePort)
    store.allocate.return_value = "blob-1"
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_error", [None, FileNotFoundError(), PermissionError()])
async def test_sweep_transfers_outside_transactions(delete_error: OSError | None) -> None:
    uow = _Transactions()
    repo = Mock(spec=FileRepository)
    store = Mock(spec=FileStoragePort)
    now = datetime.now(UTC)

    async def candidates(**kwargs: object) -> list[FileObject]:
        assert uow.depth == 1
        return [FileObject(id="file-1", storage_ref="blob-1", created_at=now)]

    async def delete(storage_ref: str) -> None:
        assert uow.depth == 0
        assert storage_ref == "blob-1"
        if delete_error is not None:
            raise delete_error

    async def remove(file_ids: list[str]) -> None:
        assert uow.depth == 1
        assert file_ids == ["file-1"]

    repo.reclaimable.side_effect = candidates
    repo.remove_all.side_effect = remove
    store.delete.side_effect = delete
    result = await _service(uow, repo, store).sweep(batch_size=2)
    blocked = isinstance(delete_error, PermissionError)
    assert result == SweepBatch(reclaimed=0 if blocked else 1, seen=1, cursor=(now, "file-1"))
    assert uow.depth == 0
    assert uow.blocks == (1 if blocked else 2)
    if blocked:
        repo.remove_all.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_commit", [False, True])
@pytest.mark.parametrize(
    "error_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit, RuntimeError, ConnectionError]
)
async def test_a_reservation_that_fails_writes_no_bytes(during_commit: bool, error_type: type[BaseException]) -> None:
    """The row comes before the bytes, so a failed reservation has nothing to undo."""
    error = error_type()
    uow = _Transactions(commit_error=error if during_commit else None)
    repo = Mock(spec=FileRepository)
    store = _store()

    async def add(record: FileObject) -> None:
        assert uow.depth == 1
        if not during_commit:
            raise error

    repo.add.side_effect = add
    with pytest.raises(error_type) as raised:
        await _service(uow, repo, store).store_produced(_output(), _chunks(b"x"))
    assert raised.value is error
    assert uow.depth == 0
    store.put_stream.assert_not_awaited()
    store.delete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [SQLAlchemyError, TimeoutError])
async def test_a_reservation_the_database_refuses_is_a_storage_error(error_type: type[BaseException]) -> None:
    uow = _Transactions()
    repo = Mock(spec=FileRepository)
    store = _store()
    repo.add.side_effect = error_type()

    with pytest.raises(FileStorageError):
        await _service(uow, repo, store).store_produced(_output(), _chunks(b"x"))

    assert uow.depth == 0
    store.put_stream.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [None, "anthropic"])
async def test_a_produced_file_is_reserved_from_its_output(provider: str | None) -> None:
    uow = _Transactions()
    repo = Mock(spec=FileRepository)
    store = _store()
    store.put_stream.return_value = 3
    repo.mark_stored.return_value = True
    workspace_id = uuid.uuid4()
    expires_at = datetime.now(UTC)
    instance = "primary" if provider else None
    container = "container-1" if provider else None
    output = NewOutput(
        file_id="file-1",
        user_id="user-1",
        workspace_id=workspace_id,
        filename="out.txt",
        mime_type="text/plain",
        purpose="user_data",
        expires_at=expires_at,
        provider=provider,
        provider_instance=instance,
        provider_container_id=container,
    )
    added: list[FileObject] = []

    async def add(record: FileObject) -> None:
        assert uow.depth == 1
        added.append(record)

    repo.add.side_effect = add
    size = await _service(uow, repo, store).store_produced(output, _chunks(b"abc"))

    (record,) = added
    assert size == 3
    assert (record.id, record.user_id, record.workspace_id) == ("file-1", "user-1", workspace_id)
    assert (record.filename, record.mime_type, record.purpose) == ("out.txt", "text/plain", "user_data")
    assert (record.storage_ref, record.bytes, record.expires_at) == ("blob-1", 0, expires_at)
    assert (record.provider, record.provider_instance, record.provider_container_id) == (provider, instance, container)
    assert record.pending_since is not None
    assert uow.depth == 0
    store.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_builds_one_service_per_job_without_opening_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    uow = _Transactions()
    service = Mock(spec=FileService)
    service.sweep = AsyncMock(return_value=SweepBatch(reclaimed=0, seen=0, cursor=None))

    @asynccontextmanager
    async def job() -> AsyncIterator[UnitOfWork]:
        yield cast(UnitOfWork, uow)

    def build(actual: UnitOfWork) -> FileService:
        assert actual is cast(UnitOfWork, uow)
        assert uow.depth == 0
        return cast(FileService, service)

    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError()])
    monkeypatch.setattr(_sweeper, "create_unit_of_work", job)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await _sweeper.run_file_sweeper(1, build)
    service.sweep.assert_awaited_once_with(batch_size=200, after=None)
    assert uow.blocks == 0
