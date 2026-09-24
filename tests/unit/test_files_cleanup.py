"""Output cleanup preserves failures and survives caller cancellation."""

import asyncio
from typing import cast
from unittest.mock import Mock

import pytest

from gateway.ports.file_storage_port import FileStoragePort
from gateway.services.files._cleanup import discard_output_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, PermissionError("denied"), RuntimeError("storage unavailable")])
async def test_cleanup_waits_for_deletion_and_suppresses_ordinary_errors(error: Exception | None) -> None:
    store = Mock(spec=FileStoragePort)
    store.delete.side_effect = error

    await discard_output_bytes(cast(FileStoragePort, store), "blob-1")

    store.delete.assert_awaited_once_with("blob-1")


@pytest.mark.asyncio
async def test_cleanup_continues_after_caller_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()
    store = Mock(spec=FileStoragePort)

    async def delete(storage_ref: str) -> None:
        assert storage_ref == "blob-1"
        started.set()
        await release.wait()
        deleted.set()

    store.delete.side_effect = delete
    cleanup = asyncio.create_task(discard_output_bytes(cast(FileStoragePort, store), "blob-1"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert not deleted.is_set()
    finally:
        release.set()
        await asyncio.wait_for(deleted.wait(), timeout=1)
