"""Bounded request-scoped multipart spooling."""

import asyncio
import os
import secrets
import shutil
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from gateway.services.provider_files.contracts import FilesError

_ENVELOPE_BYTES = 65536


class UploadAdmission:
    """Cross-process ephemeral reservations, released by close or process termination."""

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity = capacity_bytes
        self.reserved = 0

    def _claim(self, size: int) -> tuple[int, Path]:
        if os.name != "posix":
            raise FilesError(503, "Provider file spooling requires a POSIX ephemeral filesystem")
        import fcntl

        directory = Path(tempfile.gettempdir()) / f"otari-file-admission-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if directory.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise FilesError(503, "Upload temporary storage is not private")
        coordinator = os.open(directory / "admission.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(coordinator, fcntl.LOCK_EX)
            reserved = 0
            for path in directory.glob("reservation-*"):
                try:
                    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                except FileNotFoundError:
                    continue
                try:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        reserved += int(os.read(descriptor, 32))
                    else:
                        path.unlink(missing_ok=True)
                finally:
                    os.close(descriptor)
            if reserved + size > min(self.capacity, shutil.disk_usage(directory).free):
                raise FilesError(429, "Upload temporary storage capacity exhausted")
            path = directory / f"reservation-{secrets.token_hex(16)}"
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                os.write(descriptor, str(size).encode())
            except BaseException:
                os.close(descriptor)
                path.unlink(missing_ok=True)
                raise
            return descriptor, path
        finally:
            os.close(coordinator)

    @staticmethod
    def _release(reservation: tuple[int, Path]) -> None:
        descriptor, path = reservation
        # Unlink while locked: another process can never reclaim a live reservation.
        path.unlink(missing_ok=True)
        os.close(descriptor)

    @asynccontextmanager
    async def reserve(self, size: int) -> AsyncIterator[None]:
        task = asyncio.create_task(asyncio.to_thread(self._claim, size))
        try:
            reservation = await asyncio.shield(task)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                reservation = await task
                await asyncio.to_thread(self._release, reservation)
            raise
        self.reserved += size
        try:
            yield
        finally:
            self.reserved -= size
            with anyio.CancelScope(shield=True):
                await asyncio.to_thread(self._release, reservation)


@asynccontextmanager
async def receive_upload(
    headers: Headers,
    stream: AsyncIterator[bytes],
    *,
    max_bytes: int,
    idle_seconds: float,
    allowed_fields: frozenset[str] = frozenset({"expires_in_seconds"}),
) -> AsyncIterator[tuple[UploadFile, dict[str, str]]]:
    async def bounded() -> AsyncGenerator[bytes, None]:
        total = 0
        iterator = aiter(stream)
        while True:
            try:
                async with asyncio.timeout(idle_seconds):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                break
            total += len(chunk)
            if total > max_bytes + _ENVELOPE_BYTES:
                raise FilesError(413, "File size limit exceeded")
            yield chunk

    parser = MultiPartParser(
        headers, bounded(), max_files=1, max_fields=len(allowed_fields), max_part_size=_ENVELOPE_BYTES
    )
    form: FormData | None = None
    try:
        form = await parser.parse()
        if set(form.keys()) - {"file", *allowed_fields}:
            raise FilesError(400, "Unsupported upload field")
        if len(form.getlist("file")) != 1 or any(len(form.getlist(name)) > 1 for name in allowed_fields):
            raise FilesError(400, "Expected one file and unique upload fields")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise FilesError(400, "Expected one file")
        if upload.size is None or upload.size > max_bytes:
            raise FilesError(413, "File size limit exceeded")
        fields: dict[str, str] = {}
        for name in allowed_fields:
            value = form.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise FilesError(400, "Invalid upload field")
                fields[name] = value
        yield upload, fields
    except MultiPartException:
        raise FilesError(400, "Invalid multipart upload") from None
    finally:
        if form is not None:
            with anyio.CancelScope(shield=True):
                await form.close()
        else:
            # Starlette closes on MultiPartException only; limits and cancellation also must close.
            for spool in parser._files_to_close_on_error:
                with anyio.CancelScope(shield=True):
                    await asyncio.to_thread(spool.close)
