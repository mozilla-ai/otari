"""Data access for the file rows the Files API serves and reclaims."""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Never

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.sql.elements import ColumnElement

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.files import FileObject
from gateway.repositories.base_repository import BaseRepository


def could_name_a_file(value: str) -> bool:
    """Whether ``value`` could be a file ID at all.

    Every file ID is printable ASCII, and PostgreSQL refuses a NUL in a text
    parameter, so a value that fails this must not reach a query.
    """
    return value.isascii() and value.isprintable()


def _ordering(*, ascending: bool) -> tuple[Any, Any]:
    """The ``(created_at, id)`` sort key a page is cut along."""
    if ascending:
        return FileObject.created_at.asc(), FileObject.id.asc()
    return FileObject.created_at.desc(), FileObject.id.desc()


def _past(key: tuple[datetime, str], *, ascending: bool) -> ColumnElement[bool]:
    """Everything strictly past ``key`` in the page's order.

    Spelled as two clauses rather than a row-value comparison, which SQLite only partly supports.
    """
    created_at, file_id = key
    if ascending:
        return or_(
            FileObject.created_at > created_at,
            and_(FileObject.created_at == created_at, FileObject.id > file_id),
        )
    return or_(
        FileObject.created_at < created_at,
        and_(FileObject.created_at == created_at, FileObject.id < file_id),
    )


@dataclass(frozen=True)
class OutputFileRow:
    """Metadata for a produced file whose bytes have already been stored."""

    file_id: str
    user_id: str
    workspace_id: uuid.UUID
    filename: str
    mime_type: str
    bytes: int
    purpose: str
    storage_ref: str
    expires_at: datetime | None
    provider: str | None = None
    provider_instance: str | None = None
    provider_container_id: str | None = None


@dataclass(frozen=True)
class FilePageQuery:
    """Which of one user's served files a page selects, and in what order."""

    user_id: str
    # None widens the page to every workspace, which is what a master-key listing asks for.
    workspace_id: uuid.UUID | None = None
    purpose: str | None = None
    file_ids: Sequence[str] | None = None
    # The previous page's last ``(created_at, id)``, which the page starts strictly past.
    after: tuple[datetime, str] | None = None
    ascending: bool = False
    limit: int = 100


def _expired(record: FileObject, now: datetime) -> bool:
    """Whether the file has stopped being served, reading a naive stored value as UTC."""
    if record.expires_at is None:
        return False
    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at < now


class FileRepository(BaseRepository[FileObject, Never, Never]):
    """Query and stage file rows in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, FileObject)

    async def add(self, record: FileObject) -> None:
        """Stage a new file row.

        The database generates nothing on this table, so the caller's instance
        already carries every value the row will have.
        """
        self.db.add(record)
        await self.db.flush()

    async def live(self, file_id: str, user_id: str, *, workspace_id: uuid.UUID | None = None) -> FileObject | None:
        """Return the file this caller is still served, or None.

        None covers four facts on purpose: no such row, another user's row, a
        soft-deleted row and an expired one. They are one answer so that reaching
        for another tenant's file cannot be told apart from naming nothing.
        ``workspace_id`` narrows that to one workspace, and is omitted for the
        master key, which acts deployment-wide.
        """
        record = await self.any_owned(file_id, user_id, workspace_id=workspace_id)
        if record is None or record.deleted_at is not None or _expired(record, datetime.now(UTC)):
            return None
        return record

    async def any_owned(
        self, file_id: str, user_id: str, *, workspace_id: uuid.UUID | None = None
    ) -> FileObject | None:
        """Return the row this caller owns under ``file_id``, served or not.

        A paging cursor names a position rather than a file, so a row deleted or
        expired between two pages still says where the next page starts. A value
        that could not name a file names no row rather than reaching the query.
        """
        if not could_name_a_file(file_id):
            return None
        conditions = [FileObject.id == file_id, FileObject.user_id == user_id]
        if workspace_id is not None:
            conditions.append(FileObject.workspace_id == workspace_id)
        return (await self.db.execute(select(FileObject).where(*conditions))).scalar_one_or_none()

    async def page(self, query: FilePageQuery) -> list[FileObject]:
        """Return one page of a user's served files, in ``(created_at, id)`` order.

        Expired rows are left out for the same reason :meth:`live` leaves them
        out: the sweep reclaims them on a timer, so between expiry and the next
        tick a page would otherwise offer files that every other verb refuses.
        The boundary matches :func:`_expired`, so one verb cannot drop a file
        another still serves.
        """
        stmt = select(FileObject).where(
            FileObject.user_id == query.user_id,
            FileObject.deleted_at.is_(None),
            or_(FileObject.expires_at.is_(None), FileObject.expires_at >= datetime.now(UTC)),
        )
        if query.workspace_id is not None:
            stmt = stmt.where(FileObject.workspace_id == query.workspace_id)
        if query.purpose is not None:
            stmt = stmt.where(FileObject.purpose == query.purpose)
        if query.file_ids is not None:
            stmt = stmt.where(FileObject.id.in_([f for f in query.file_ids if could_name_a_file(f)]))
        if query.after is not None:
            stmt = stmt.where(_past(query.after, ascending=query.ascending))
        stmt = stmt.order_by(*_ordering(ascending=query.ascending)).limit(query.limit)
        return list((await self.db.execute(stmt)).scalars().all())

    async def soft_delete(self, record: FileObject, at: datetime) -> None:
        """Stage the file's removal from every read, leaving the row for the sweep."""
        record.deleted_at = at
        await self.db.flush()

    async def existing_ids(self, file_ids: Collection[str]) -> set[str]:
        """Which of ``file_ids`` already have a row, recorded or uploaded.

        An ID that could not name a file has no row and is left out of the
        query, so one unusable ID in a batch costs only itself rather than
        failing the lookup the whole batch depends on.
        """
        usable = [file_id for file_id in file_ids if could_name_a_file(file_id)]
        if not usable:
            return set()
        result = await self.db.execute(select(FileObject.id).where(FileObject.id.in_(usable)))
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
            stmt = stmt.where(_past(after, ascending=True))
        stmt = stmt.order_by(*_ordering(ascending=True)).limit(batch_size)
        return (await self.db.execute(stmt)).scalars().all()

    async def remove_all(self, file_ids: Collection[str]) -> None:
        """Stage the deletion of the rows whose blobs are gone."""
        if not file_ids:
            return
        await self.db.execute(delete(FileObject).where(FileObject.id.in_(list(file_ids))))
