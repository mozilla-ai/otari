"""Data access for organizations."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import Organization, OrganizationCreate, OrganizationUpdate
from gateway.repositories.base_repository import BaseRepository


class OrganizationRepository(BaseRepository[Organization, OrganizationCreate, OrganizationUpdate]):
    """Repository for organization rows."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Organization)

    async def get_by_slug(self, slug: str) -> Organization | None:
        """Return the organization with this slug, or None."""
        result = await self.db.execute(select(Organization).where(col(Organization.slug) == slug))
        return result.scalars().first()

    async def create_organization(
        self,
        *,
        name: str,
        slug: str,
        created_by_user_id: uuid.UUID | None,
    ) -> Organization:
        """Stage a new organization."""
        organization = Organization(name=name, slug=slug, created_by_user_id=created_by_user_id)
        self.db.add(organization)
        await self.db.flush()
        await self.db.refresh(organization)
        return organization

    async def update_organization(
        self,
        organization: Organization,
        update_data: dict[str, Any],
    ) -> Organization:
        """Stage an update from a plain mapping of columns."""
        organization.sqlmodel_update(update_data)
        self.db.add(organization)
        await self.db.flush()
        await self.db.refresh(organization)
        return organization


__all__ = ["OrganizationRepository"]
