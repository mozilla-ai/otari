"""Pluggable blob storage for uploaded file bytes.

The ``/v1/files`` API stores file *metadata* in the database (see
``gateway.models.entities.FileObject``) and the raw *bytes* here, keyed by an
opaque ``storage_ref``. Keeping bytes out of the relational store lets large
uploads live on a filesystem / object store while the DB stays lean.

Only a local-filesystem backend ships today; ``S3FileStore`` / ``GCSFileStore``
can implement the same :class:`FileStore` protocol without touching callers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable

from gateway.core.config import GatewayConfig
from gateway.log_config import logger


@runtime_checkable
class FileStore(Protocol):
    """Storage backend for raw uploaded file bytes."""

    async def put(self, file_id: str, data: bytes) -> str:
        """Persist ``data`` for ``file_id`` and return an opaque storage ref."""
        ...

    async def get(self, storage_ref: str) -> bytes:
        """Return the bytes previously stored under ``storage_ref``."""
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

    async def delete(self, storage_ref: str) -> None:
        path = self._resolve(storage_ref)

        def _unlink() -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                logger.debug("file_store delete: %s already absent", storage_ref)

        await asyncio.to_thread(_unlink)


def build_file_store(config: GatewayConfig) -> FileStore:
    """Construct the configured :class:`FileStore` backend."""
    backend = config.files_backend.strip().lower()
    if backend == "local":
        return LocalDirFileStore(config.files_local_dir)
    msg = f"Unsupported files_backend: {config.files_backend!r} (supported: 'local')"
    raise ValueError(msg)
