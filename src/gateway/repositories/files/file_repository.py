"""Data access for the file rows the Files API serves and reclaims."""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Never

from sqlalchemy import and_, delete, or_, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.files import FileObject
from gateway.repositories.base_repository import BaseRepository


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
    # Set when a provider's own sandbox produced the file, naming where it came from.
    provider: str | None = None
    provider_instance: str | None = None
    provider_container_id: str | None = None


class FileRepository(BaseRepository[FileObject, Never, Never]):
    """Query and stage file rows in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, FileObject)

    async def existing_ids(self, file_ids: Collection[str]) -> set[str]:
        """Which of ``file_ids`` already have a row, recorded or uploaded."""
        if not file_ids:
            return set()
        result = await self.db.execute(select(FileObject.id).where(FileObject.id.in_(list(file_ids))))
        return set(result.scalars())

    async def record_output(self, row: OutputFileRow) -> None:
        """Stage the row for a file a run wrote."""
        self.db.add(
            FileObject(
                id=row.file_id,
                user_id=row.user_id,
                workspace_id=row.workspace_id,
                filename=row.filename,
                mime_type=row.mime_type,
                bytes=row.bytes,
                purpose=row.purpose,
                storage_ref=row.storage_ref,
                provider=row.provider,
                provider_instance=row.provider_instance,
                provider_container_id=row.provider_container_id,
                created_at=datetime.now(UTC),
                expires_at=row.expires_at,
            )
        )
        await self.db.flush()

    async def reclaimable(
        self,
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
        return (await self.db.execute(stmt)).scalars().all()

    async def remove_all(self, file_ids: Collection[str]) -> None:
        """Stage the deletion of the rows whose blobs are gone."""
        if not file_ids:
            return
        await self.db.execute(delete(FileObject).where(FileObject.id.in_(list(file_ids))))
