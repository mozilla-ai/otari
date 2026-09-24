"""Data access for the copies a provider holds of a stored file."""

from __future__ import annotations

import uuid
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.files import FileProviderCopy
from gateway.repositories.base_repository import BaseRepository


class FileProviderCopyRepository(BaseRepository[FileProviderCopy, Never, Never]):
    """Query and stage provider copies in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, FileProviderCopy)

    async def in_account(
        self, file_id: str, *, provider: str, provider_instance: str, credential_workspace_id: uuid.UUID
    ) -> FileProviderCopy | None:
        """The copy this file has in one provider account, or None.

        Every part of the key is required, because a copy found under a
        different one names a file that account does not hold.
        """
        result = await self.db.execute(
            select(FileProviderCopy).where(
                FileProviderCopy.file_id == file_id,
                FileProviderCopy.provider == provider,
                FileProviderCopy.provider_instance == provider_instance,
                FileProviderCopy.credential_workspace_id == credential_workspace_id,
            )
        )
        return result.scalar_one_or_none()

    async def record(self, copy: FileProviderCopy) -> None:
        """Stage ``copy`` as the one copy its file has in that provider account.

        A copy the provider has since expired is replaced rather than kept
        beside the new one, which is what the primary key already says.
        """
        await self.db.merge(copy)
        await self.db.flush()
