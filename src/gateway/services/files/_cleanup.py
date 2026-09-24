"""Best-effort removal of unregistered output blobs."""

import asyncio
import contextlib

from gateway.log_config import logger
from gateway.ports.file_storage_port import FileStoragePort

_PENDING: set[asyncio.Task[None]] = set()


def _report_cleanup(task: asyncio.Task[None], storage_ref: str) -> None:
    _PENDING.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("Could not remove unregistered output blob %s: %s", storage_ref, error)


async def discard_output_bytes(file_store: FileStoragePort, storage_ref: str) -> None:
    """Delete unregistered bytes without replacing an ordinary failure with a cleanup error."""
    # Retain the deletion until it finishes, even if its caller is cancelled.
    task = asyncio.create_task(file_store.delete(storage_ref))
    _PENDING.add(task)
    task.add_done_callback(lambda done: _report_cleanup(done, storage_ref))
    with contextlib.suppress(Exception):
        await asyncio.shield(task)
