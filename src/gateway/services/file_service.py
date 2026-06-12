"""Shared data-access helpers for uploaded files.

Used by both the ``/v1/files`` route and the content normalizer, which resolves
``file_id`` references in chat messages back to bytes. Centralising the
user-scoping + soft-delete + expiry rules here keeps the two call sites
consistent.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import FileObject
from gateway.services.file_store import FileStore


def _is_expired(record: FileObject) -> bool:
    if record.expires_at is None:
        return False
    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at < datetime.now(UTC)


async def fetch_file(db: AsyncSession, file_id: str, user_id: str | None) -> FileObject | None:
    """Return a live, non-deleted, unexpired file owned by ``user_id``.

    Returns ``None`` if the file does not exist, belongs to another user, is
    soft-deleted, or has expired — callers map that to a 404 so cross-user
    access is indistinguishable from a missing file.
    """
    result = await db.execute(
        select(FileObject).where(
            FileObject.id == file_id,
            FileObject.user_id == user_id,
            FileObject.deleted_at.is_(None),
        )
    )
    record = result.scalar_one_or_none()
    if record is None or _is_expired(record):
        return None
    return record


async def read_file_bytes(file_store: FileStore, record: FileObject) -> bytes:
    """Load the raw bytes for ``record`` from the blob backend."""
    return await file_store.get(record.storage_ref)
