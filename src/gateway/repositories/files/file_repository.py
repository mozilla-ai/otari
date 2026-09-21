"""Data access for the file rows the ``/v1/files`` API serves and reclaims.

The sweep's two statements and the code-execution output insert live here
rather than in the service that drives them, so the service orchestrates and
this module is the only place the queries are spelled.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import and_, delete, or_, select

from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.tools import FileObject


@dataclass(frozen=True)
class OutputFileRow:
    """One file a code-execution run produced, already resolved to its columns."""

    file_id: str
    user_id: str
    workspace_id: uuid.UUID
    filename: str
    mime_type: str
    bytes: int
    purpose: str
    storage_ref: str
    expires_at: datetime | None


async def record_output_file(uow: UnitOfWork, row: OutputFileRow) -> None:
    """Stage the row for a file a run wrote. Flushes; the caller's unit of work commits."""
    db = session_for(uow)
    db.add(
        FileObject(
            id=row.file_id,
            user_id=row.user_id,
            workspace_id=row.workspace_id,
            filename=row.filename,
            mime_type=row.mime_type,
            bytes=row.bytes,
            purpose=row.purpose,
            storage_ref=row.storage_ref,
            created_at=datetime.now(UTC),
            expires_at=row.expires_at,
        )
    )
    await db.flush()


async def reclaimable_files(
    uow: UnitOfWork,
    *,
    batch_size: int,
    after: tuple[datetime, str] | None = None,
    now: datetime | None = None,
) -> Sequence[FileObject]:
    """One batch of soft-deleted or expired rows, in ``(created_at, id)`` order.

    ``after`` is the previous batch's last key: paging by key rather than from
    the top is what keeps a row whose blob keeps failing to delete from parking
    at the head and hiding everything behind it.
    """
    stmt = select(FileObject).where(
        or_(FileObject.deleted_at.is_not(None), FileObject.expires_at < (now or datetime.now(UTC)))
    )
    if after is not None:
        created_at, file_id = after
        stmt = stmt.where(
            or_(
                FileObject.created_at > created_at,
                and_(FileObject.created_at == created_at, FileObject.id > file_id),
            )
        )
    stmt = stmt.order_by(FileObject.created_at, FileObject.id).limit(batch_size)
    return (await session_for(uow).execute(stmt)).scalars().all()


async def delete_file_rows(uow: UnitOfWork, file_ids: Collection[str]) -> None:
    """Remove the rows whose blobs are gone. The caller's unit of work commits."""
    if not file_ids:
        return
    await session_for(uow).execute(delete(FileObject).where(FileObject.id.in_(list(file_ids))))
