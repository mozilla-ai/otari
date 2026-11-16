"""Data access for organization invitations."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import ColumnElement, CursorResult, case, delete, func, literal, null, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.addresses import stored_address
from gateway.core.sql import insert_ignoring_duplicates
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.organizations import OrganizationInvitation, WorkspaceAssignment
from gateway.models.tenancy import OrganizationMemberRole
from gateway.repositories.base_repository import BaseRepository

# NOTE: Two writers that each hold a row the other wants fail fast instead of waiting.
_ADD_LOCK_TIMEOUT_MS = 2000


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

    created_at: datetime
    email: str
    expires_at: datetime
    invited_by_user_id: uuid.UUID | None
    next_send_at: datetime | None
    organization_id: uuid.UUID
    role: OrganizationMemberRole
    token_hash: str
    workspace_assignments: Sequence[WorkspaceAssignment]


@dataclass(frozen=True, kw_only=True)
class SendRequest:
    """The values a send, first or repeated, replaces on an invitation."""

    expires_at: datetime
    invited_by_user_id: uuid.UUID | None
    next_send_at: datetime | None
    requested_at: datetime
    token_hash: str


@dataclass(frozen=True, kw_only=True)
class SendResult:
    """A send request's result, and the invitation as it now stands.

    ``invitation`` is None only when the invitation no longer exists.
    """

    invitation: OrganizationInvitation | None
    sent: bool


def _in_organization(invitation_id: uuid.UUID, organization_id: uuid.UUID) -> tuple[ColumnElement[bool], ...]:
    """The conditions that match the invitation only when it belongs to the organization."""
    return (OrganizationInvitation.id == invitation_id, OrganizationInvitation.organization_id == organization_id)


class OrganizationInvitationRepository(BaseRepository[OrganizationInvitation, Never, Never]):
    """Reads and writes organization invitations.

    Every address is stored and matched in its normalized form.
    """

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, OrganizationInvitation)

    async def add(self, new: NewInvitation) -> OrganizationInvitation | None:
        """Save a new invitation in the current transaction and return it.

        Creating an invitation counts as its first send request.
        Returns None when the organization already has an invitation for this address.
        The transaction stays usable.
        Raise ``ValueError`` when ``new.email`` is not an email address that can be stored.
        """
        email = stored_address(new.email)
        if email is None:
            msg = "An invitation needs an email address that can be stored."
            raise ValueError(msg)
        invitation = OrganizationInvitation(
            created_at=new.created_at,
            email=email,
            expires_at=new.expires_at,
            invited_by_user_id=new.invited_by_user_id,
            last_send_requested_at=new.created_at,
            next_send_at=new.next_send_at,
            organization_id=new.organization_id,
            role=new.role,
            send_count=1,
            token_hash=new.token_hash,
            workspace_assignments=list(new.workspace_assignments),
        )
        rows = await insert_ignoring_duplicates(
            self.db,
            [invitation],
            conflict=[OrganizationInvitation.organization_id, OrganizationInvitation.email],
            returning=OrganizationInvitation,
            lock_timeout_ms=_ADD_LOCK_TIMEOUT_MS,
        )
        return rows[0] if rows else None

    async def add_many(self, news: Sequence[NewInvitation]) -> list[OrganizationInvitation]:
        """Save every new invitation in one statement, skipping any address its organization already has.

        Creating an invitation counts as its first send request.
        An address missing from the result already had an invitation in its organization.
        The transaction stays usable.
        Raise ``ValueError`` when one of ``news`` has an email that is not an email address that can be stored.

        NOTE: Rows are inserted in address order,
        so two calls sharing addresses only ever wait on each other here, never deadlock.
        That is why this sets no lock timeout to fail fast on.
        """
        if not news:
            return []
        validated: list[tuple[str, NewInvitation]] = []
        for new in news:
            email = stored_address(new.email)
            if email is None:
                msg = "An invitation needs an email address that can be stored."
                raise ValueError(msg)
            validated.append((email, new))
        validated.sort(key=lambda item: item[0])
        invitations = [
            OrganizationInvitation(
                created_at=new.created_at,
                email=email,
                expires_at=new.expires_at,
                invited_by_user_id=new.invited_by_user_id,
                last_send_requested_at=new.created_at,
                next_send_at=new.next_send_at,
                organization_id=new.organization_id,
                role=new.role,
                send_count=1,
                token_hash=new.token_hash,
                workspace_assignments=list(new.workspace_assignments),
            )
            for email, new in validated
        ]
        return await insert_ignoring_duplicates(
            self.db,
            invitations,
            conflict=[OrganizationInvitation.organization_id, OrganizationInvitation.email],
            returning=OrganizationInvitation,
        )

    async def claim_due_send(
        self, *, now: datetime, held_until: datetime, max_attempts: int, token_hash: str
    ) -> DueInvitation | None:
        """Claim the invitation whose email has been due the longest, and return its values.

        Leaves ``send_count`` and ``last_send_requested_at`` as they were.
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

    async def delete_expired_before(self, cutoff: datetime, *, limit: int) -> int:
        """Delete up to ``limit`` of the oldest invitations that expired before ``cutoff``.

        Returns how many there were, which may be fewer than ``limit``.
        A row another writer already holds is left for a later sweep rather than waited for.

        NOTE: A caller must not look up one of these by id again on the same transaction;
        the lookup can still return it.
        """
        candidates = await self.db.execute(
            select(OrganizationInvitation.id)
            .where(OrganizationInvitation.expires_at < cutoff)
            .order_by(OrganizationInvitation.expires_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        invitation_ids = list(candidates.scalars().all())
        if not invitation_ids:
            return 0
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                delete(OrganizationInvitation)
                # The predicate again, not the ids alone: SQLite has no FOR UPDATE,
                # so a row renewed between the select and here is no longer one this may delete.
                .where(OrganizationInvitation.id.in_(invitation_ids), OrganizationInvitation.expires_at < cutoff)
                .execution_options(synchronize_session=False)
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
                .where(*_in_organization(invitation_id, organization_id))
                .execution_options(synchronize_session="fetch")
            ),
        )
        return result.rowcount == 1

    async def delete_expired_in_organization(
        self, invitation_ids: Collection[uuid.UUID], organization_id: uuid.UUID, *, expired_by: datetime
    ) -> int:
        """Delete these invitations if they belong to this organization and are still expired as of ``expired_by``.

        Returns how many were deleted, which may be fewer than given.
        A renewal that committed first leaves that row in place for its own invitation to keep.

        NOTE: A caller must not look up one of these by id again on the same transaction;
        the lookup can still return it.
        """
        if not invitation_ids:
            return 0
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                delete(OrganizationInvitation)
                .where(
                    OrganizationInvitation.id.in_(invitation_ids),
                    OrganizationInvitation.organization_id == organization_id,
                    OrganizationInvitation.expires_at <= expired_by,
                )
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount

    async def finish_send(self, invitation_id: uuid.UUID, *, token_hash: str) -> bool:
        """Record that the send of a claim is finished, and return True if it was recorded.

        No email is due for the invitation after this.
        ``token_hash`` is the hash that the claim stored.
        Return False when the invitation no longer holds that hash or no longer exists.
        """
        return await self._set_next_send_at(invitation_id, token_hash=token_hash, next_send_at=None)

    async def get_by_organization_and_emails(
        self, organization_id: uuid.UUID, emails: Collection[str]
    ) -> list[OrganizationInvitation]:
        """Return the organization's invitations for these addresses, in one query.

        Text that is not an address that can be stored matches nothing.
        """
        addresses = {address for email in emails if (address := stored_address(email)) is not None}
        if not addresses:
            return []
        result = await self.db.execute(
            select(OrganizationInvitation).where(
                OrganizationInvitation.organization_id == organization_id,
                OrganizationInvitation.email.in_(addresses),
            )
        )
        return list(result.scalars().all())

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
            select(OrganizationInvitation).where(*_in_organization(invitation_id, organization_id))
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

    async def request_send(
        self,
        invitation_id: uuid.UUID,
        organization_id: uuid.UUID,
        request: SendRequest,
        *,
        expected_send_count: int,
    ) -> SendResult:
        """Record a send requested against the invitation's current send count.

        The invitation must belong to ``organization_id``.
        Resets the delivery attempts to zero and advances the send count by one.
        The write matches nothing, and changes nothing, unless the invitation still exists,
        belongs to that organization, and holds ``expected_send_count``.

        Replaces the token hash, so a send already in flight records nothing.

        NOTE: Callers must take the invitation from the result rather than reading it again.
        """
        result = await self.db.execute(
            update(OrganizationInvitation)
            .where(
                OrganizationInvitation.id == invitation_id,
                OrganizationInvitation.organization_id == organization_id,
                OrganizationInvitation.send_count == expected_send_count,
            )
            .values(
                delivery_attempts=0,
                expires_at=request.expires_at,
                invited_by_user_id=request.invited_by_user_id,
                last_send_requested_at=request.requested_at,
                next_send_at=request.next_send_at,
                send_count=expected_send_count + 1,
                token_hash=request.token_hash,
            )
            .returning(OrganizationInvitation)
            .execution_options(populate_existing=True, synchronize_session=False)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            return SendResult(invitation=row, sent=True)
        current = await self.db.execute(
            select(OrganizationInvitation)
            .where(
                OrganizationInvitation.id == invitation_id,
                OrganizationInvitation.organization_id == organization_id,
            )
            .execution_options(populate_existing=True)
        )
        return SendResult(invitation=current.scalar_one_or_none(), sent=False)

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
