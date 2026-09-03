"""Data access for organization email-domain claims."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import OrganizationDomain, OrganizationDomainCreate, OrganizationDomainUpdate
from gateway.repositories.base_repository import BaseRepository


class OrganizationDomainRepository(
    BaseRepository[OrganizationDomain, OrganizationDomainCreate, OrganizationDomainUpdate]
):
    """Repository for organization domain rows."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, OrganizationDomain)

    async def get_by_domain(self, domain: str) -> OrganizationDomain | None:
        """Return the claim on ``domain``, or None.

        Deployment-wide rather than scoped to an organization, which is what the
        UNIQUE constraint makes meaningful: this is the sign-in lookup, and at
        that point there is no organization yet to scope by.
        """
        result = await self.db.execute(select(OrganizationDomain).where(col(OrganizationDomain.domain) == domain))
        return result.scalars().first()

    async def get_by_id_and_organization(
        self,
        organization_domain_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> OrganizationDomain | None:
        """Return one claim, only if it belongs to this organization.

        Both halves in the query rather than a fetch and an ownership check
        after it, so another organization's claim is indistinguishable from one
        that does not exist.
        """
        result = await self.db.execute(
            select(OrganizationDomain).where(
                col(OrganizationDomain.id) == organization_domain_id,
                col(OrganizationDomain.organization_id) == organization_id,
            )
        )
        return result.scalars().first()

    async def list_by_organization(self, organization_id: uuid.UUID) -> list[OrganizationDomain]:
        """Return an organization's claims, oldest first."""
        result = await self.db.execute(
            select(OrganizationDomain)
            .where(col(OrganizationDomain.organization_id) == organization_id)
            .order_by(col(OrganizationDomain.created_at), col(OrganizationDomain.id))
        )
        return list(result.scalars().all())

    async def create_domain(
        self,
        *,
        organization_id: uuid.UUID,
        domain: str,
        default_role: str,
        enabled: bool,
        verification_token: str,
    ) -> OrganizationDomain:
        """Stage a new, unverified claim."""
        row = OrganizationDomain(
            organization_id=organization_id,
            domain=domain,
            default_role=default_role,
            enabled=enabled,
            verification_token=verification_token,
        )
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def update_domain(self, row: OrganizationDomain, update_data: dict[str, Any]) -> OrganizationDomain:
        """Stage a change from a plain mapping of columns."""
        row.sqlmodel_update(update_data)
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def mark_verified(self, row: OrganizationDomain, *, verified_at: datetime) -> OrganizationDomain:
        """Stamp a claim as proven."""
        row.verified_at = verified_at
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def delete_domain(self, row: OrganizationDomain) -> None:
        """Stage the removal of a claim."""
        await self.db.delete(row)
        await self.db.flush()


__all__ = ["OrganizationDomainRepository"]
