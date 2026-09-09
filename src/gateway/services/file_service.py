"""Shared data-access helpers for uploaded files.

Used by the ``/v1/files`` route, the content normalizer (which resolves
``file_id`` references in chat messages back to bytes), and the code-execution
sandbox (which seeds uploads into a session and stores what a run produced).
Centralising the user-scoping, workspace-scoping, soft-delete and expiry rules
here keeps every call site consistent.
"""

from __future__ import annotations

import asyncio
import mimetypes
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import FileObject
from gateway.services.file_store import FileStore

# The purpose stamped on a file the code-execution sandbox produced, so a
# listing can tell a run's artifact from a user's upload.
CODE_EXECUTION_OUTPUT_PURPOSE = "code_execution_output"


def _is_expired(record: FileObject) -> bool:
    if record.expires_at is None:
        return False
    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at < datetime.now(UTC)


async def fetch_file(
    db: AsyncSession, file_id: str, user_id: str | None, *, workspace_id: uuid.UUID | None = None
) -> FileObject | None:
    """Return a live, non-deleted, unexpired file owned by ``user_id``.

    Returns ``None`` if the file does not exist, belongs to another user, is
    soft-deleted, or has expired: callers map that to a 404 so cross-user
    access is indistinguishable from a missing file.

    ``workspace_id`` narrows that to one workspace, so a key confined to one
    never reaches a file uploaded through a key in another, even when the same
    user holds both. Omitted for a master-key request, which is the operator
    acting deployment-wide and sees every workspace, matching ``GET /v1/keys``.
    """
    conditions = [
        FileObject.id == file_id,
        FileObject.user_id == user_id,
        FileObject.deleted_at.is_(None),
    ]
    if workspace_id is not None:
        conditions.append(FileObject.workspace_id == workspace_id)
    result = await db.execute(select(FileObject).where(*conditions))
    record = result.scalar_one_or_none()
    if record is None or _is_expired(record):
        return None
    return record


async def read_file_bytes(file_store: FileStore, record: FileObject) -> bytes:
    """Load the raw bytes for ``record`` from the blob backend."""
    return await file_store.get(record.storage_ref)


def guess_mime_type(filename: str | None, declared: str | None = None) -> str:
    """The media type for ``filename``: the declared one when it says something, else by extension."""
    if declared and declared != "application/octet-stream":
        return declared
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return declared or "application/octet-stream"


def expiry_for(config: GatewayConfig, now: datetime | None = None) -> datetime | None:
    """When a file stored now stops being served, or ``None`` when files are kept indefinitely."""
    if config.files_retention_hours is None:
        return None
    return (now or datetime.now(UTC)) + timedelta(hours=config.files_retention_hours)


@dataclass(frozen=True)
class StagedFile:
    """An uploaded file a request asked the code-execution sandbox to see.

    Recorded by the content normalizer while it walks the messages, and consumed
    by the sandbox backend when it opens the session. Carries the storage ref
    rather than the bytes so a large upload is read once, at staging time, and
    never held across the normalizer's whole pass.
    """

    file_id: str
    filename: str
    mime_type: str
    storage_ref: str


class SandboxFileBridge:
    """Moves files between the ``/v1/files`` store and one sandbox session.

    Built per request by the route, once the billed user and workspace are
    known, and handed to the sandbox backend. ``inputs`` are the uploads the
    request referenced for the sandbox; :meth:`store_output` persists a file a
    run produced as a new ``FileObject`` owned by the same user and workspace,
    so the caller can download it through ``GET /v1/files/{id}/content``.

    Standalone only: it needs the local database that hybrid mode does not have.
    """

    def __init__(
        self,
        *,
        file_store: FileStore,
        config: GatewayConfig,
        user_id: str,
        workspace_id: uuid.UUID,
        inputs: list[StagedFile],
    ) -> None:
        self._file_store = file_store
        self._config = config
        self._user_id = user_id
        self._workspace_id = workspace_id
        self.inputs = inputs
        # Everything stored through this bridge, so the route can report what a
        # request produced after the tool loop has finished.
        self.outputs: list[FileObject] = []

    @property
    def max_output_bytes(self) -> int:
        return self._config.files_max_bytes

    async def read_input(self, staged: StagedFile) -> bytes:
        return await self._file_store.get(staged.storage_ref)

    async def store_output(self, filename: str, data: bytes) -> str:
        """Persist ``data`` as a new file and return its ``file_id``.

        Opens a session of its own rather than borrowing the request's: this
        runs from inside the tool loop, while the request session is idle
        between the reservation and its settlement, and a commit there would
        interleave with that lifecycle.
        """
        file_id = f"file-{uuid.uuid4().hex}"
        storage_ref = await self._file_store.put(file_id, data)
        record = FileObject(
            id=file_id,
            user_id=self._user_id,
            workspace_id=self._workspace_id,
            filename=filename,
            mime_type=guess_mime_type(filename),
            bytes=len(data),
            purpose=CODE_EXECUTION_OUTPUT_PURPOSE,
            storage_ref=storage_ref,
            created_at=datetime.now(UTC),
            expires_at=expiry_for(self._config),
        )
        try:
            async with create_session() as db:
                db.add(record)
                await db.commit()
        except SQLAlchemyError:
            await self._file_store.delete(storage_ref)
            raise
        self.outputs.append(record)
        return file_id


async def sweep_files(db: AsyncSession, file_store: FileStore, *, batch_size: int) -> int:
    """Reclaim one batch of expired or soft-deleted files: their bytes, then their rows.

    Returns how many rows went, so a caller can loop until a batch comes back
    short. A blob that is already gone is not an error (the delete route removes
    the bytes best-effort before this ever sees the row); any other storage
    failure leaves the row in place for the next pass rather than orphaning
    bytes nothing references.
    """
    now = datetime.now(UTC)
    stmt = (
        select(FileObject)
        .where(or_(FileObject.deleted_at.is_not(None), FileObject.expires_at < now))
        .order_by(FileObject.created_at)
        .limit(batch_size)
    )
    records = (await db.execute(stmt)).scalars().all()
    reclaimed: list[str] = []
    for record in records:
        try:
            await file_store.delete(record.storage_ref)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("file sweep: could not remove blob %s for %s: %s", record.storage_ref, record.id, exc)
            continue
        reclaimed.append(record.id)
    if reclaimed:
        await db.execute(delete(FileObject).where(FileObject.id.in_(reclaimed)))
        await db.commit()
        logger.info("file sweep: reclaimed %d file(s)", len(reclaimed))
    return len(reclaimed)


_MAX_SWEEP_PASSES = 10


async def run_file_sweeper(interval: float, file_store: FileStore, *, batch_size: int = 200) -> None:
    """Reclaim expired and deleted files on a timer, forever. Cancelled at shutdown.

    Expiry alone only hides a file (``fetch_file`` answers 404); this is what
    gives its bytes back. Every error is swallowed and retried on the next tick,
    matching the other lifespan tasks: a storage or database blip must not kill
    the sweeper, because nothing would restart it.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                for _ in range(_MAX_SWEEP_PASSES):
                    if await sweep_files(db, file_store, batch_size=batch_size) < batch_size:
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("File sweep failed; retrying in %ss", interval, exc_info=True)
