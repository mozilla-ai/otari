"""Data access for organization memberships."""

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import (
    Organization,
    OrganizationMember,
    OrganizationMemberCreate,
    OrganizationMemberUpdate,
    User,
)
from gateway.repositories.base_repository import BaseRepository
from gateway.repositories.tenancy.user_repository import user_alphabetical_order

# Statuses a membership can hold and still belong on the roster. Removal
# suspends rather than deletes, so "suspended" is the one that drops off.
LISTABLE_STATUSES = ("active", "invited")


class OrganizationMemberRepository(
    BaseRepository[OrganizationMember, OrganizationMemberCreate, OrganizationMemberUpdate]
):
    """Repository for organization membership rows."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, OrganizationMember)

    async def get_by_organization_and_user(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return the membership joining a user to an organization, whatever its status."""
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.user_id) == user_id,
            )
        )
        return result.scalars().first()

    async def get_active_by_organization_and_user(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return the active membership joining a user to an organization."""
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.user_id) == user_id,
                col(OrganizationMember.status) == "active",
            )
        )
        return result.scalars().first()

    async def get_by_id_and_organization(
        self,
        organization_member_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return a membership by id, scoped to one organization.

        The scoping is the cross-tenant boundary: a caller passing another
        organization's member id resolves to nothing rather than reading or
        mutating that organization's row.
        """
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.id) == organization_member_id,
                col(OrganizationMember.organization_id) == organization_id,
            )
        )
        return result.scalars().first()

    async def get_by_user(self, user_id: uuid.UUID, *, active_only: bool = False) -> list[OrganizationMember]:
        """Return a user's memberships, oldest first."""
        statement = select(OrganizationMember).where(col(OrganizationMember.user_id) == user_id)
        if active_only:
            statement = statement.where(col(OrganizationMember.status) == "active")
        result = await self.db.execute(statement.order_by(col(OrganizationMember.created_at)))
        return list(result.scalars().all())

    async def get_first_active_for_user(self, user_id: uuid.UUID) -> OrganizationMember | None:
        """Return a user's oldest active membership, or None."""
        result = await self.db.execute(
            select(OrganizationMember)
            .where(
                col(OrganizationMember.user_id) == user_id,
                col(OrganizationMember.status) == "active",
            )
            .order_by(col(OrganizationMember.created_at))
        )
        return result.scalars().first()

    async def create_membership(
        self,
        *,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str,
        status: str = "active",
    ) -> OrganizationMember:
        """Stage a new membership."""
        member = OrganizationMember(
            organization_id=organization_id,
            user_id=user_id,
            role=role,
            status=status,
        )
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        return member

    async def update_membership(
        self,
        member: OrganizationMember,
        update_data: dict[str, Any],
    ) -> OrganizationMember:
        """Stage an update from a plain mapping of columns."""
        member.sqlmodel_update(update_data)
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        return member

    async def count_active_owners(self, organization_id: uuid.UUID) -> int:
        """Count an organization's active owners.

        The last-owner guard reads this: an organization with no owner left has
        nobody who can delete it or manage its members.
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(OrganizationMember)
            .where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.role) == "owner",
                col(OrganizationMember.status) == "active",
            )
        )
        return result.scalar_one()

    async def get_active_owner(self, organization_id: uuid.UUID) -> OrganizationMember | None:
        """Return an organization's earliest active owner, if any.

        Ordered by ``created_at`` then ``id`` so the answer is stable across
        calls even when two owner rows share a timestamp.
        """
        result = await self.db.execute(
            select(OrganizationMember)
            .where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.role) == "owner",
                col(OrganizationMember.status) == "active",
            )
            .order_by(col(OrganizationMember.created_at), col(OrganizationMember.id))
        )
        return result.scalars().first()

    async def get_by_organization_with_users(
        self,
        organization_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[tuple[OrganizationMember, User]], int]:
        """Return a page of the roster as ``(membership, identity)`` pairs, plus the total.

        One join rather than a lookup per row, so a roster of N members costs
        one query instead of N+1.
        """
        count_result = await self.db.execute(
            select(func.count())
            .select_from(OrganizationMember)
            .where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.status).in_(LISTABLE_STATUSES),
            )
        )
        count = count_result.scalar_one()

        result = await self.db.execute(
            select(OrganizationMember, User)
            .join(User, col(OrganizationMember.user_id) == col(User.id))
            .where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.status).in_(LISTABLE_STATUSES),
            )
            .order_by(user_alphabetical_order(), col(OrganizationMember.id))
            .offset(skip)
            .limit(limit)
        )
        return [(member, user) for member, user in result.all()], count

    async def get_by_user_with_organizations(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
    ) -> list[tuple[OrganizationMember, Organization]]:
        """Return a user's memberships joined to their organizations, oldest first."""
        statement = (
            select(OrganizationMember, Organization)
            .join(Organization, col(OrganizationMember.organization_id) == col(Organization.id))
            .where(col(OrganizationMember.user_id) == user_id)
            .order_by(col(OrganizationMember.created_at))
        )
        if status is not None:
            statement = statement.where(col(OrganizationMember.status) == status)
        result = await self.db.execute(statement)
        return [(member, organization) for member, organization in result.all()]


__all__ = ["LISTABLE_STATUSES", "OrganizationMemberRepository"]
