"""Best-effort removal of unregistered output blobs."""

import asyncio
import contextlib

from gateway.ports.file_storage_port import FileStoragePort


async def discard_output_bytes(file_store: FileStoragePort, storage_ref: str) -> None:
    """Delete unregistered bytes without replacing an ordinary failure with a cleanup error."""
    # Shielding lets deletion continue if the caller is cancelled during the await.
    with contextlib.suppress(Exception):
        await asyncio.shield(file_store.delete(storage_ref))
