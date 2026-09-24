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

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.files import FileObject
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FileRepositories, FileRepository, OutputFileRow
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
    "error_type",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit, RuntimeError, ConnectionError, TimeoutError],
)
async def test_output_failure_cleanup(during_commit: bool, error_type: type[BaseException]) -> None:
    error = error_type()
    uow = _Transactions(commit_error=error if during_commit else None)
    repo = Mock(spec=FileRepository)
    store = Mock(spec=FileStoragePort)

    async def record(row: OutputFileRow) -> None:
        assert uow.depth == 1
        if not during_commit:
            raise error

    async def delete(storage_ref: str) -> None:
        assert uow.depth == 0
        assert storage_ref == "blob-1"

    repo.record_output.side_effect = record
    store.delete.side_effect = delete
    output = NewOutput("file-1", "user-1", uuid.uuid4(), "out.txt", "text/plain", 1, "user_data", "blob-1", None)
    with pytest.raises(error_type) as raised:
        await _service(uow, repo, store).record_output(output)
    assert raised.value is error
    assert uow.depth == 0
    if during_commit:
        store.delete.assert_not_awaited()
    else:
        store.delete.assert_awaited_once_with("blob-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [None, "anthropic"])
async def test_output_maps_service_input_to_repository_row(provider: str | None) -> None:
    uow = _Transactions()
    repo = Mock(spec=FileRepository)
    store = Mock(spec=FileStoragePort)
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
        bytes=42,
        purpose="user_data",
        storage_ref="blob-1",
        expires_at=expires_at,
        provider=provider,
        provider_instance=instance,
        provider_container_id=container,
    )

    async def record(row: OutputFileRow) -> None:
        assert uow.depth == 1
        assert type(row) is OutputFileRow
        assert row == OutputFileRow(
            file_id="file-1",
            user_id="user-1",
            workspace_id=workspace_id,
            filename="out.txt",
            mime_type="text/plain",
            bytes=42,
            purpose="user_data",
            storage_ref="blob-1",
            expires_at=expires_at,
            provider=provider,
            provider_instance=instance,
            provider_container_id=container,
        )

    repo.record_output.side_effect = record
    await _service(uow, repo, store).record_output(output)

    repo.record_output.assert_awaited_once()
    assert uow.blocks == 1
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
