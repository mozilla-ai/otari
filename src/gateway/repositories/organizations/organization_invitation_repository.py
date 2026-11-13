"""Data access for organization invitations."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import ColumnElement, CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.addresses import normalized_address
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.organizations import OrganizationInvitation, WorkspaceAssignment
from gateway.models.tenancy import OrganizationMemberRole
from gateway.repositories.base_repository import BaseRepository


@dataclass(frozen=True, kw_only=True)
class NewInvitation:
    """The values of an invitation to save."""

    email: str
    expires_at: datetime
    invited_by_user_id: uuid.UUID | None
    organization_id: uuid.UUID
    role: OrganizationMemberRole
    token_hash: str
    workspace_assignments: Sequence[WorkspaceAssignment]


class OrganizationInvitationRepository(BaseRepository[OrganizationInvitation, Never, Never]):
    """Reads and writes organization invitations.

    Every address is stored and matched in its normalized form.
    """

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, OrganizationInvitation)

    async def add(self, new: NewInvitation) -> OrganizationInvitation:
        """Save a new invitation in the current transaction and return it.

        Raise ``ValueError`` when ``new.email`` is not an email address.
        """
        email = normalized_address(new.email)
        if email is None:
            msg = "An invitation needs an email address."
            raise ValueError(msg)
        invitation = OrganizationInvitation(
            email=email,
            expires_at=new.expires_at,
            invited_by_user_id=new.invited_by_user_id,
            organization_id=new.organization_id,
            role=new.role,
            token_hash=new.token_hash,
            workspace_assignments=list(new.workspace_assignments),
        )
        self.db.add(invitation)
        await self.db.flush()
        return invitation

    async def delete_expired_before(self, cutoff: datetime) -> int:
        """Delete the invitations that expired before ``cutoff``, and return how many there were."""
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                delete(OrganizationInvitation)
                .where(OrganizationInvitation.expires_at < cutoff)
                .execution_options(synchronize_session="fetch")
            ),
        )
        return result.rowcount

    async def delete_in_organization(self, invitation_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
        """Delete the invitation if it belongs to this organization, and return True if it was deleted.

        When two transactions delete the same invitation, only one of them returns True.
        """
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                delete(OrganizationInvitation)
                .where(
                    OrganizationInvitation.id == invitation_id,
                    OrganizationInvitation.organization_id == organization_id,
                )
                .execution_options(synchronize_session="fetch")
            ),
        )
        return result.rowcount == 1

    async def get_by_organization_and_email(
        self, organization_id: uuid.UUID, email: str
    ) -> OrganizationInvitation | None:
        """Return the organization's invitation for this address, or None if it has none."""
        address = normalized_address(email)
        if address is None:
            return None
        result = await self.db.execute(
            select(OrganizationInvitation).where(
                OrganizationInvitation.organization_id == organization_id,
                OrganizationInvitation.email == address,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_token_hash(self, token_hash: str) -> OrganizationInvitation | None:
        """Return the invitation with this token hash, or None if there is none."""
        result = await self.db.execute(
            select(OrganizationInvitation).where(OrganizationInvitation.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def get_in_organization(
        self, invitation_id: uuid.UUID, organization_id: uuid.UUID
    ) -> OrganizationInvitation | None:
        """Return the invitation if it belongs to this organization, or None if it does not."""
        result = await self.db.execute(
            select(OrganizationInvitation).where(
                OrganizationInvitation.id == invitation_id,
                OrganizationInvitation.organization_id == organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_organization(
        self, organization_id: uuid.UUID, *, skip: int, limit: int
    ) -> tuple[list[OrganizationInvitation], int]:
        """Return one page of the organization's invitations, oldest first, and how many it has in total."""
        scope: ColumnElement[bool] = OrganizationInvitation.organization_id == organization_id
        rows = await self.db.execute(
            select(OrganizationInvitation)
            .where(scope)
            .order_by(OrganizationInvitation.created_at, OrganizationInvitation.id)
            .offset(skip)
            .limit(limit)
        )
        total = await self.db.execute(select(func.count()).select_from(OrganizationInvitation).where(scope))
        return list(rows.scalars().all()), total.scalar_one()
