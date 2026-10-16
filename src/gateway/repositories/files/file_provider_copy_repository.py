"""Data access for the copies a provider holds of a stored file."""

from __future__ import annotations

import uuid
from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Never, cast

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import ProviderCopyNotRecordedError
from gateway.models.files import FileProviderCopy
from gateway.repositories.base_repository import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.engine import CursorResult


class FileProviderCopyRepository(BaseRepository[FileProviderCopy, Never, Never]):
    """Query and stage provider copies in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, FileProviderCopy)

    async def usable(
        self, file_ids: Collection[str], account_identity: str, *, expiring_after: datetime
    ) -> dict[str, FileProviderCopy]:
        """The confirmed copy with the most life left for each file, in one account.

        A file whose copies all expire by ``expiring_after`` is left out, and so
        is a pending copy, which has no provider ID to name yet.
        """
        if not file_ids:
            return {}
        result = await self.db.execute(
            select(FileProviderCopy)
            .where(
                FileProviderCopy.file_id.in_(list(file_ids)),
                FileProviderCopy.account_identity == account_identity,
                FileProviderCopy.pending_since.is_(None),
                FileProviderCopy.expires_at > expiring_after,
            )
            .order_by(FileProviderCopy.expires_at.desc())
        )
        copies: dict[str, FileProviderCopy] = {}
        for copy in result.scalars():
            copies.setdefault(copy.file_id, copy)
        return copies

    async def reserve(self, copy: FileProviderCopy) -> None:
        """Stage ``copy`` as pending, before the provider holds it.

        Raises:
            ProviderCopyNotRecordedError: the database refused the row.
        """
        self.db.add(copy)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise ProviderCopyNotRecordedError from exc

    async def confirm(self, copy_id: uuid.UUID, *, provider_file_id: str, expires_at: datetime) -> bool:
        """Stage a pending copy as held by the provider, and report whether it was still pending.

        False says the reservation is gone, so nothing names the copy the
        provider now holds.
        """
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(FileProviderCopy)
                .where(FileProviderCopy.id == copy_id, FileProviderCopy.pending_since.is_not(None))
                .values(provider_file_id=provider_file_id, expires_at=expires_at, pending_since=None)
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount == 1

    async def cancel(self, copy_id: uuid.UUID) -> None:
        """Stage the removal of a copy the provider will not hold."""
        await self.db.execute(delete(FileProviderCopy).where(FileProviderCopy.id == copy_id))

    async def remove_expired(self, *, expired_before: datetime, limit: int) -> int:
        """Stage the removal of up to ``limit`` confirmed copies that expired before ``expired_before``.

        The provider has already let such a copy go, so its row names nothing
        a request could use or a deletion could reach.
        """
        expired = (
            select(FileProviderCopy.id)
            .where(FileProviderCopy.pending_since.is_(None), FileProviderCopy.expires_at < expired_before)
            .order_by(FileProviderCopy.expires_at)
            .limit(limit)
        )
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(delete(FileProviderCopy).where(FileProviderCopy.id.in_(expired))),
        )
        return result.rowcount

    async def remove_stale_pending(self, *, pending_before: datetime, limit: int) -> int:
        """Stage the removal of up to ``limit`` copies pending since before ``pending_before``.

        Such a row never had a provider ID recorded, so it cannot name a copy
        the provider holds.
        """
        stale = (
            select(FileProviderCopy.id)
            .where(FileProviderCopy.pending_since < pending_before)
            .order_by(FileProviderCopy.pending_since)
            .limit(limit)
        )
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(delete(FileProviderCopy).where(FileProviderCopy.id.in_(stale))),
        )
        return result.rowcount
