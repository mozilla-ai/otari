"""Data access for organization invitations."""

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import (
    Invitation,
    InvitationCreate,
    InvitationUpdate,
    Organization,
    OrganizationMember,
)
from gateway.repositories.base_repository import BaseRepository


class InvitationRepository(BaseRepository[Invitation, InvitationCreate, InvitationUpdate]):
    """Repository for invitation rows."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Invitation)

    async def get_by_token_hash(self, token_hash: str) -> Invitation | None:
        """Return the invitation this token hashes to, or None."""
        result = await self.db.execute(select(Invitation).where(col(Invitation.token_hash) == token_hash))
        return result.scalars().first()

    async def get_pending_by_organization_members(
        self,
        organization_member_ids: Iterable[uuid.UUID],
    ) -> list[Invitation]:
        """Return the still-pending invitations among a set of memberships.

        Batched for the roster: one query for a page of N members rather than
        one lookup per invited row. Filtered to ``pending`` rather than every
        invitation ever issued for these memberships, so a resolved
        (accepted/cancelled/expired) one never resurfaces as if still actionable.
        """
        ids = list(organization_member_ids)
        if not ids:
            return []
        result = await self.db.execute(
            select(Invitation).where(
                col(Invitation.organization_member_id).in_(ids),
                col(Invitation.status) == "pending",
            )
        )
        return list(result.scalars().all())

    async def get_live_pending_for_user_with_context(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[tuple[Invitation, OrganizationMember, Organization]], int]:
        """Return a page of the invitations still awaiting this identity, plus the total.

        The invitee-side counterpart to ``get_pending_by_organization_members``,
        which answers the roster's question ("which of these members have an
        invitation out") rather than the inbox's ("which organizations are
        waiting on me"). Joined in one query, so a row arrives with the
        organization's name and the role on offer instead of a lookup per row.

        Filtered on ``expires_at`` rather than on ``status`` alone, because
        expiry is lazy: ``_resolve_pending_invitation`` only flips a row to
        ``expired`` when someone presents its token, so an unopened link sits
        ``pending`` with its deadline already past. Filtering here rather than
        in the service is what keeps ``count`` and the page boundaries honest,
        which dropping lapsed rows after the fact would not.

        Both sides of the pair are checked, not just the invitation: a
        membership that has since been suspended (revoked, or removed outright)
        keeps its cancelled invitation queryable, and only a membership still
        ``invited`` has an accept that would mean anything.

        ``id`` breaks the ``created_at`` tie for the reason
        ``get_by_user_with_organizations`` gives: two rows written in one
        transaction share a timestamp, and a page whose order is not total can
        return one row twice and another never.
        """
        conditions = [
            col(OrganizationMember.user_id) == user_id,
            col(OrganizationMember.status) == "invited",
            col(Invitation.status) == "pending",
            col(Invitation.expires_at) >= now,
        ]
        joined = (
            select(Invitation, OrganizationMember, Organization)
            .join(OrganizationMember, col(Invitation.organization_member_id) == col(OrganizationMember.id))
            .join(Organization, col(Invitation.organization_id) == col(Organization.id))
        )

        count_result = await self.db.execute(
            select(func.count())
            .select_from(Invitation)
            .join(OrganizationMember, col(Invitation.organization_member_id) == col(OrganizationMember.id))
            .where(*conditions)
        )
        count = count_result.scalar_one()

        result = await self.db.execute(
            joined.where(*conditions)
            .order_by(col(Invitation.created_at), col(Invitation.id))
            .offset(skip)
            .limit(limit)
        )
        return [(invitation, membership, organization) for invitation, membership, organization in result.all()], count

    async def create_invitation(
        self,
        *,
        organization_id: uuid.UUID,
        organization_member_id: uuid.UUID,
        email: str,
        invited_by_user_id: uuid.UUID | None,
        token_hash: str,
        workspace_assignments: list[dict[str, str]],
        expires_at: datetime,
    ) -> Invitation:
        """Stage a new invitation."""
        invitation = Invitation(
            organization_id=organization_id,
            organization_member_id=organization_member_id,
            email=email,
            invited_by_user_id=invited_by_user_id,
            token_hash=token_hash,
            workspace_assignments=workspace_assignments,
            expires_at=expires_at,
        )
        self.db.add(invitation)
        await self.db.flush()
        await self.db.refresh(invitation)
        return invitation

    async def update_status(self, invitation: Invitation, update_data: dict[str, Any]) -> Invitation:
        """Stage a status change from a plain mapping of columns."""
        invitation.sqlmodel_update(update_data)
        self.db.add(invitation)
        await self.db.flush()
        await self.db.refresh(invitation)
        return invitation


__all__ = ["InvitationRepository"]
