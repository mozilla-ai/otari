"""Data access for the audit record of who invited each member."""

import uuid
from datetime import datetime
from typing import Never

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.organizations import OrganizationMembershipOrigin
from gateway.repositories.base_repository import BaseRepository


class MembershipOriginRepository(BaseRepository[OrganizationMembershipOrigin, Never, Never]):
    """Saves the audit record of who invited a member, when they were invited, and when they joined."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, OrganizationMembershipOrigin)

    async def record(
        self,
        *,
        organization_member_id: uuid.UUID,
        invited_by_user_id: uuid.UUID | None,
        invited_at: datetime,
        joined_at: datetime,
    ) -> None:
        """Save one member's audit record in the current transaction."""
        self.db.add(
            OrganizationMembershipOrigin(
                organization_member_id=organization_member_id,
                invited_by_user_id=invited_by_user_id,
                invited_at=invited_at,
                joined_at=joined_at,
            )
        )
        await self.db.flush()
