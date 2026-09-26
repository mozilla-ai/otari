"""Where the raw bytes behind an uploaded file are kept.

The seam between an upload and whatever holds its blob.
A file's metadata stays in this deployment's database and only the bytes cross
here, keyed by an opaque ``storage_ref`` the store itself mints, so a large
upload can live on a filesystem or an object store while the relational store
stays lean.
The port owns no search: nothing above it asks the store to find a file, only
to hold one, hand it back and drop it.

The same bytes move in two shapes.
``put`` and ``get`` are the full-buffer pair, for a caller that needs the whole
blob in memory regardless.
``put_stream`` and ``get_stream`` move it chunk by chunk, which is what bounds
memory while several large uploads are in flight at once.
An adapter implements both pairs.

Storage failures cross as the ``OSError`` family, with ``FileNotFoundError``
for a blob that is not there, so an adapter over a remote store translates its
own client's exceptions before they reach a caller.
A malformed ``storage_ref`` is a ``ValueError`` instead, because a ref that
escapes the store's root is a caller error rather than a storage failure.

An adapter reaches its backend through ``asyncio.to_thread`` rather than an
AnyIO primitive. ``delete`` runs while a cancellation unwinds, and an AnyIO
checkpoint inside a cancelled scope raises before the backend is reached.

Stability: this interface is not frozen while Otari is pre-1.0.
Overlay authors should pin a released tag and expect the shape to move.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Protocol

__all__ = ["FileStoragePort"]


class FileStoragePort(Protocol):
    """What a build must answer to keep the bytes behind an uploaded file."""

    async def put(self, file_id: str, data: bytes) -> str:
        """Persist ``data`` for ``file_id`` and return an opaque storage ref."""
        ...

    async def get(self, storage_ref: str) -> bytes:
        """Return the bytes stored under ``storage_ref``."""
        ...

    async def put_stream(self, file_id: str, chunks: AsyncIterator[bytes]) -> tuple[str, int]:
        """Persist ``chunks`` for ``file_id``, and report the ref and the bytes written.

        No size ceiling of its own: the cap belongs to whoever produces
        ``chunks``, and this stores whatever it is given.
        An adapter writes to a temporary object and publishes it only after the
        source is fully consumed. If the source or write fails, or the task is
        cancelled before publication, the adapter removes the temporary object
        and leaves no object under the requested file id.
        """
        ...

    def get_stream(self, storage_ref: str) -> AsyncGenerator[bytes, None]:
        """Yield the bytes stored under ``storage_ref`` chunk by chunk.

        Declared without ``async`` because calling an async-generator function
        returns the generator rather than an awaitable; an implementation is an
        ordinary ``async def`` with ``yield``.
        ``AsyncGenerator`` rather than the narrower ``AsyncIterator``, because
        the stream can be closed early with ``aclose()``.
        """
        ...

    async def delete(self, storage_ref: str) -> None:
        """Remove the bytes under ``storage_ref``, and do nothing if they are gone."""
        ...
