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

    async def lock(self, organization_id: uuid.UUID) -> None:
        """Take a row lock on the organization, serializing its invariant checks.

        Two of this slice's rules are "the organization keeps at least one of
        these": one active owner, and one workspace. Both are read-then-write,
        and neither has a unique index to lose to, so the ``IntegrityError``
        guards elsewhere in this slice cannot catch them: two transactions each
        counting two owners, then demoting a different one, both commit and
        leave none. Serializing on the parent row is what makes the count the
        writer acts on still true when it writes.

        ``FOR UPDATE`` is a no-op on SQLite, which has no row locks. That engine
        admits one writer at a time for the whole database, so the same pair of
        transactions serializes there anyway; PostgreSQL is where the lock is
        load-bearing.
        """
        await self.db.execute(
            select(col(Organization.id)).where(col(Organization.id) == organization_id).with_for_update()
        )

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
