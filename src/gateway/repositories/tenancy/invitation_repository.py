"""Data access for organization invitations."""

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import Invitation, InvitationCreate, InvitationUpdate
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
