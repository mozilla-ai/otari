"""Output cleanup preserves failures and survives caller cancellation."""

import asyncio
from typing import cast
from unittest.mock import Mock

import pytest

from gateway.log_config import logger
from gateway.ports.file_storage_port import FileStoragePort
from gateway.services.files import _cleanup
from gateway.services.files._cleanup import discard_output_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, PermissionError("denied"), RuntimeError("storage unavailable")])
async def test_cleanup_waits_for_deletion_and_suppresses_ordinary_errors(
    error: Exception | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning = Mock()
    monkeypatch.setattr(logger, "warning", warning)
    store = Mock(spec=FileStoragePort)
    store.delete.side_effect = error

    await discard_output_bytes(cast(FileStoragePort, store), "blob-1")

    store.delete.assert_awaited_once_with("blob-1")
    assert not _cleanup._PENDING
    if error is None:
        warning.assert_not_called()
    else:
        warning.assert_called_once_with("Could not remove unregistered output blob %s: %s", "blob-1", error)


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_original_error() -> None:
    store = Mock(spec=FileStoragePort)
    store.delete.side_effect = PermissionError("denied")
    original = ValueError("output registration failed")

    with pytest.raises(ValueError) as raised:
        try:
            raise original
        except ValueError:
            await discard_output_bytes(cast(FileStoragePort, store), "blob-1")
            raise

    assert raised.value is original
    assert not _cleanup._PENDING


@pytest.mark.asyncio
async def test_cancelled_deletion_releases_task_without_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = Mock()
    monkeypatch.setattr(logger, "warning", warning)
    store = Mock(spec=FileStoragePort)
    store.delete.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await discard_output_bytes(cast(FileStoragePort, store), "blob-1")

    assert not _cleanup._PENDING
    warning.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, PermissionError("denied")])
async def test_cleanup_continues_after_caller_cancellation(
    error: Exception | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning = Mock()
    monkeypatch.setattr(logger, "warning", warning)
    started = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()
    store = Mock(spec=FileStoragePort)

    async def delete(storage_ref: str) -> None:
        assert storage_ref == "blob-1"
        started.set()
        await release.wait()
        deleted.set()
        if error is not None:
            raise error

    store.delete.side_effect = delete
    cleanup = asyncio.create_task(discard_output_bytes(cast(FileStoragePort, store), "blob-1"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert not deleted.is_set()
        assert len(_cleanup._PENDING) == 1
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*_cleanup._PENDING, return_exceptions=True), timeout=1)
    assert deleted.is_set()
    assert not _cleanup._PENDING
    if error is None:
        warning.assert_not_called()
    else:
        warning.assert_called_once_with("Could not remove unregistered output blob %s: %s", "blob-1", error)
