"""Data access for organization email-domain claims."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
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

    async def get_verified_by_domain(self, domain: str) -> OrganizationDomain | None:
        """Return the one *proven* claim on ``domain``, or None.

        Deployment-wide rather than scoped to an organization: this is the
        sign-in lookup, and at that point there is no organization to scope by.
        At most one row can match, which is what the partial unique index
        guarantees; unproven claims are excluded here because they grant
        nothing and any number of organizations may hold one.
        """
        result = await self.db.execute(
            select(OrganizationDomain).where(
                col(OrganizationDomain.domain) == domain,
                col(OrganizationDomain.verified_at).is_not(None),
            )
        )
        return result.scalars().first()

    async def get_by_domain_and_organization(
        self,
        domain: str,
        organization_id: uuid.UUID,
    ) -> OrganizationDomain | None:
        """Return this organization's own claim on ``domain``, or None."""
        result = await self.db.execute(
            select(OrganizationDomain).where(
                col(OrganizationDomain.domain) == domain,
                col(OrganizationDomain.organization_id) == organization_id,
            )
        )
        return result.scalars().first()

    async def list_rival_unverified(self, domain: str, *, winner_id: uuid.UUID) -> list[OrganizationDomain]:
        """Return the unproven claims on ``domain`` that a proof has just beaten."""
        result = await self.db.execute(
            select(OrganizationDomain).where(
                col(OrganizationDomain.domain) == domain,
                col(OrganizationDomain.id) != winner_id,
                col(OrganizationDomain.verified_at).is_(None),
            )
        )
        return list(result.scalars().all())

    async def count_for_organization(self, organization_id: uuid.UUID) -> int:
        """How many domains this organization currently claims."""
        result = await self.db.execute(
            select(func.count())
            .select_from(OrganizationDomain)
            .where(col(OrganizationDomain.organization_id) == organization_id)
        )
        return int(result.scalar_one())

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
