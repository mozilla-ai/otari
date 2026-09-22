"""Shared data-access helpers for uploaded files.

Used by the ``/v1/files`` route, the content normalizer (which resolves
``file_id`` references in chat messages back to bytes), and the code-execution
sandbox (which seeds uploads into a session and stores what a run produced).
Centralising the user-scoping, workspace-scoping, soft-delete and expiry rules
here keeps every call site consistent.
"""

from __future__ import annotations

import mimetypes
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.tools import FileObject
from gateway.ports.file_storage_port import FileStoragePort

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


async def read_file_bytes(file_store: FileStoragePort, record: FileObject) -> bytes:
    """Load the raw bytes for ``record`` from the blob backend.

    Raises ``FileNotFoundError`` for a row with no stored bytes.
    """
    if record.storage_ref is None:
        raise FileNotFoundError(f"{record.id} has no stored bytes")
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
    # The name the file has inside the session's working directory, which is
    # also what the model is told; see ``sandbox_path_for``.
    filename: str
    mime_type: str
    storage_ref: str


def sandbox_path_for(filename: str, taken: Collection[str]) -> str:
    """The name a staged upload gets inside the session's working directory.

    The upload's own name reduced to its last path segment, so a name carrying
    separators neither nests nor escapes, and suffixed ``-2``, ``-3``, ... when
    an earlier attachment already took it, so two uploads named alike are both
    there rather than one overwriting the other. A name with no usable segment
    becomes ``file``.
    """
    base = PurePosixPath(filename.replace("\\", "/")).name
    if base in ("", ".", ".."):
        base = "file"
    if base not in taken:
        return base
    stem, dot, ext = base.rpartition(".")
    if not dot or not stem:
        stem, ext = base, ""
    else:
        ext = f".{ext}"
    n = 2
    while f"{stem}-{n}{ext}" in taken:
        n += 1
    return f"{stem}-{n}{ext}"
