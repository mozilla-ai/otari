"""Pluggable blob storage for uploaded file bytes.

The ``/v1/files`` API stores file *metadata* in the database (see
``gateway.models.entities.FileObject``) and the raw *bytes* here, keyed by an
opaque ``storage_ref``. Keeping bytes out of the relational store lets large
uploads live on a filesystem / object store while the DB stays lean.

``put``/``get`` are the full-buffer path (kept for callers, like the content
normalizer, that need the whole blob in memory regardless). ``put_stream`` /
``get_stream`` let the upload and download routes move bytes chunk-by-chunk
instead of buffering an entire file, which is what actually bounds memory use
for concurrent large uploads (see issue #156).

Only a local-filesystem backend ships today; ``S3FileStore`` / ``GCSFileStore``
can implement the same :class:`FileStore` protocol without touching callers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

from gateway.core.config import GatewayConfig
from gateway.log_config import logger

_STREAM_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def _open_handle(path: Path, mode: str) -> AsyncIterator[IO[bytes]]:
    """Open ``path`` off the event loop, guaranteeing the handle is closed.

    The close itself is shielded: this ``finally`` can run while unwinding a
    cancellation (client disconnect), and an unshielded ``await`` there could
    be cut off by a repeated cancel() before the handle actually closes.
    """
    handle = await asyncio.to_thread(path.open, mode)
    try:
        yield handle
    finally:
        try:
            await asyncio.shield(asyncio.to_thread(handle.close))
        except Exception as close_exc:
            logger.warning("_open_handle: failed to close %s: %s", path, close_exc)


@runtime_checkable
class FileStore(Protocol):
    """Storage backend for raw uploaded file bytes."""

    async def put(self, file_id: str, data: bytes) -> str:
        """Persist ``data`` for ``file_id`` and return an opaque storage ref."""
        ...

    async def get(self, storage_ref: str) -> bytes:
        """Return the bytes previously stored under ``storage_ref``."""
        ...

    async def put_stream(self, file_id: str, chunks: AsyncIterator[bytes]) -> tuple[str, int]:
        """Persist ``chunks`` for ``file_id`` without buffering them fully in memory.

        Returns the opaque storage ref and the total byte count written. Callers
        that need to cap the upload size (e.g. the ``/v1/files`` route) must
        enforce that themselves while producing ``chunks``; this is a pure
        storage primitive and does not know about HTTP limits.
        """
        ...

    def get_stream(self, storage_ref: str) -> AsyncGenerator[bytes, None]:
        """Yield the bytes stored under ``storage_ref`` chunk-by-chunk.

        Not ``async def``: implementations are async generators, called
        synchronously and consumed with ``async for``, not awaited first.
        Typed as ``AsyncGenerator``, not the narrower ``AsyncIterator``,
        because callers rely on ``aclose()`` (e.g. the download route closes
        it early on client disconnect so the file handle doesn't linger).
        """
        ...

    async def delete(self, storage_ref: str) -> None:
        """Remove the bytes under ``storage_ref`` (no-op if already gone)."""
        ...


class LocalDirFileStore:
    """Filesystem-backed :class:`FileStore`.

    Files are sharded into 256 subdirectories by the first two hex characters of
    the file id to avoid pathologically large directories. The ``storage_ref``
    is the POSIX-relative path under the root, so it survives a root relocation.
    """

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def _shard(self, file_id: str) -> str:
        # file ids look like ``file-<hex>``; shard on the first two hex chars.
        token = file_id.split("-", 1)[-1] or file_id
        prefix = (token[:2] or "00").lower()
        return f"{prefix}/{file_id}"

    def _resolve(self, storage_ref: str) -> Path:
        """Resolve ``storage_ref`` under the root, rejecting any escape.

        ``storage_ref`` is server-generated today, but this is defence-in-depth:
        if a value ever came from elsewhere, a ``../`` traversal must not read or
        delete outside the store root.
        """
        root = self._root.resolve()
        path = (root / storage_ref).resolve()
        if not path.is_relative_to(root):
            msg = f"Invalid storage_ref escapes the file store root: {storage_ref!r}"
            raise ValueError(msg)
        return path

    async def put(self, file_id: str, data: bytes) -> str:
        ref = self._shard(file_id)
        path = self._resolve(ref)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(_write)
        return ref

    async def get(self, storage_ref: str) -> bytes:
        path = self._resolve(storage_ref)
        return await asyncio.to_thread(path.read_bytes)

    async def put_stream(self, file_id: str, chunks: AsyncIterator[bytes]) -> tuple[str, int]:
        ref = self._shard(file_id)
        path = self._resolve(ref)

        def _mkparent() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)

        def _unlink_partial() -> None:
            path.unlink(missing_ok=True)

        await asyncio.to_thread(_mkparent)
        total = 0
        try:
            async with _open_handle(path, "wb") as handle:
                async for chunk in chunks:
                    total += len(chunk)
                    await asyncio.to_thread(handle.write, chunk)
        except BaseException:
            # The chunk source (e.g. the route's size-cap check, or a client
            # disconnect) failed partway through; don't leave a truncated blob
            # with no storage_ref pointing at it, since the caller never gets a
            # ref back to clean it up. BaseException includes CancelledError,
            # so this cleanup itself runs inside an already-cancelling task;
            # shield it so a repeated cancel() can't cut it off before the
            # unlink completes, and don't let a cleanup failure mask the
            # original error.
            try:
                await asyncio.shield(asyncio.to_thread(_unlink_partial))
            except Exception as cleanup_exc:
                logger.warning("put_stream: failed to remove partial blob %s: %s", ref, cleanup_exc)
            raise
        return ref, total

    async def get_stream(self, storage_ref: str) -> AsyncGenerator[bytes, None]:
        path = self._resolve(storage_ref)
        async with _open_handle(path, "rb") as handle:
            while chunk := await asyncio.to_thread(handle.read, _STREAM_CHUNK_BYTES):
                yield chunk

    async def delete(self, storage_ref: str) -> None:
        path = self._resolve(storage_ref)

        def _unlink() -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                logger.debug("file_store delete: %s already absent", storage_ref)
                return
            # Best-effort: if this was the last file in its shard dir (e.g. a
            # zero-byte upload rejected right after being written), don't leave
            # an empty shard directory behind. rmdir fails harmlessly if the
            # shard still has other files in it, or on an unlikely concurrent
            # write race.
            try:
                path.parent.rmdir()
            except OSError:
                pass

        await asyncio.to_thread(_unlink)


def build_file_store(config: GatewayConfig) -> FileStore:
    """Construct the configured :class:`FileStore` backend."""
    backend = config.files_backend.strip().lower()
    if backend == "local":
        return LocalDirFileStore(config.files_local_dir)
    msg = f"Unsupported files_backend: {config.files_backend!r} (supported: 'local')"
    raise ValueError(msg)
