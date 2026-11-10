"""Data access for ``memory_organization_settings``. Flushes, never commits.

Not fenced: the table holds one switch per organization, never content.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryOrganizationSettings
from gateway.repositories.base_repository import BaseRepository


class OrganizationSettingsRepository(BaseRepository[MemoryOrganizationSettings, Never, Never]):
    """Read and change an organization's memory switch."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryOrganizationSettings)

    async def enabled(self, organization_id: uuid.UUID) -> bool:
        """Whether the organization has memory on. No row is off."""
        row = await self.db.get(MemoryOrganizationSettings, organization_id)
        return row is not None and row.enabled

    async def enabled_among(self, organization_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """The organizations among ``organization_ids`` that have memory on."""
        if not organization_ids:
            return set()
        result = await self.db.execute(
            select(MemoryOrganizationSettings.organization_id).where(
                MemoryOrganizationSettings.organization_id.in_(list(organization_ids)),
                MemoryOrganizationSettings.enabled.is_(True),
            )
        )
        return set(result.scalars().all())

    async def put(self, organization_id: uuid.UUID, *, enabled: bool) -> MemoryOrganizationSettings:
        """Stage the organization's switch, creating its row the first time."""
        row = await self.db.get(MemoryOrganizationSettings, organization_id)
        if row is None:
            row = MemoryOrganizationSettings(organization_id=organization_id)
            self.db.add(row)
        row.enabled = enabled
        await self.db.flush()
        return row
