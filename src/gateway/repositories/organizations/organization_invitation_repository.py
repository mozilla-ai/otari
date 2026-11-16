"""Data access for organization invitations."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import ColumnElement, CursorResult, case, delete, func, literal, null, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.addresses import stored_address
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.organizations import OrganizationInvitation, WorkspaceAssignment
from gateway.models.tenancy import OrganizationMemberRole
from gateway.repositories.base_repository import BaseRepository


@dataclass(frozen=True, kw_only=True)
class DueInvitation:
    """The values of an invitation whose send was claimed.

    ``attempt`` is 1 for the first send of the invitation.
    """

    attempt: int
    declined_at: datetime | None
    email: str
    expires_at: datetime
    id: uuid.UUID
    invited_by_user_id: uuid.UUID | None
    organization_id: uuid.UUID
    role: OrganizationMemberRole
    workspace_assignments: Sequence[WorkspaceAssignment]


@dataclass(frozen=True, kw_only=True)
class NewInvitation:
    """The values of an invitation to save.

    ``next_send_at`` is when the invitation email is due, or None when no send is due.
    """

    email: str
    expires_at: datetime
    invited_by_user_id: uuid.UUID | None
    next_send_at: datetime | None
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

        Raise ``ValueError`` when ``new.email`` is not an email address that can be stored.
        """
        email = stored_address(new.email)
        if email is None:
            msg = "An invitation needs an email address that can be stored."
            raise ValueError(msg)
        invitation = OrganizationInvitation(
            email=email,
            expires_at=new.expires_at,
            invited_by_user_id=new.invited_by_user_id,
            next_send_at=new.next_send_at,
            organization_id=new.organization_id,
            role=new.role,
            token_hash=new.token_hash,
            workspace_assignments=list(new.workspace_assignments),
        )
        self.db.add(invitation)
        await self.db.flush()
        return invitation

    async def claim_due_send(
        self, *, now: datetime, held_until: datetime, max_attempts: int, token_hash: str
    ) -> DueInvitation | None:
        """Claim the invitation whose email has been due the longest, and return its values.

        Return None when no email is due.
        An email is due when the invitation's due time is ``now`` or earlier.
        The claim counts one attempt and replaces the invitation's token hash with ``token_hash``.
        The email is due again at ``held_until`` unless a result is recorded before then.
        The claim of attempt ``max_attempts`` or later is the last one, and it leaves no email due.
        When two transactions claim the same invitation, one of them gets it and the other gets None.
        Raise ``ValueError`` when ``held_until`` is not later than ``now``.
        """
        if held_until <= now:
            msg = "A claimed invitation must be held until a time later than now."
            raise ValueError(msg)
        is_due: ColumnElement[bool] = OrganizationInvitation.next_send_at <= now
        attempt = OrganizationInvitation.delivery_attempts + 1
        due_longest = (
            select(OrganizationInvitation.id)
            .where(is_due)
            .order_by(OrganizationInvitation.next_send_at, OrganizationInvitation.id)
            .limit(1)
            .scalar_subquery()
        )
        result = await self.db.execute(
            update(OrganizationInvitation)
            # NOTE: The due test must stay here as well as in the subquery.
            # Without it, a claim that waits for a concurrent claim takes the same invitation.
            .where(OrganizationInvitation.id == due_longest, is_due)
            .values(
                delivery_attempts=attempt,
                next_send_at=case(
                    (attempt >= max_attempts, null()),
                    else_=literal(held_until, OrganizationInvitation.next_send_at.type),
                ),
                token_hash=token_hash,
            )
            .returning(OrganizationInvitation)
            .execution_options(populate_existing=True, synchronize_session=False)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return DueInvitation(
            attempt=row.delivery_attempts,
            declined_at=row.declined_at,
            email=row.email,
            expires_at=row.expires_at,
            id=row.id,
            invited_by_user_id=row.invited_by_user_id,
            organization_id=row.organization_id,
            role=row.role,
            workspace_assignments=row.workspace_assignments,
        )

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

    async def finish_send(self, invitation_id: uuid.UUID, *, token_hash: str) -> bool:
        """Record that the send of a claim is finished, and return True if it was recorded.

        No email is due for the invitation after this.
        ``token_hash`` is the hash that the claim stored.
        Return False when the invitation no longer holds that hash or no longer exists.
        """
        return await self._set_next_send_at(invitation_id, token_hash=token_hash, next_send_at=None)

    async def get_by_organization_and_email(
        self, organization_id: uuid.UUID, email: str
    ) -> OrganizationInvitation | None:
        """Return the organization's invitation for this address, or None if it has none."""
        address = stored_address(email)
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

    async def retry_send(self, invitation_id: uuid.UUID, *, token_hash: str, next_send_at: datetime) -> bool:
        """Record that the send of a claim failed, and return True if it was recorded.

        The email is due again at ``next_send_at``.
        ``token_hash`` is the hash that the claim stored.
        Return False when the invitation no longer holds that hash or no longer exists.
        """
        return await self._set_next_send_at(invitation_id, token_hash=token_hash, next_send_at=next_send_at)

    async def _set_next_send_at(
        self, invitation_id: uuid.UUID, *, token_hash: str, next_send_at: datetime | None
    ) -> bool:
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                update(OrganizationInvitation)
                .where(OrganizationInvitation.id == invitation_id, OrganizationInvitation.token_hash == token_hash)
                .values(next_send_at=next_send_at)
                .execution_options(synchronize_session="fetch")
            ),
        )
        return result.rowcount == 1
