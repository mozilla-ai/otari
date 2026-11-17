"""Answering an invitation link: what it offers, and declining it."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from gateway.core.addresses import same_address
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import (
    InvitationAddressMismatchError,
    InvitationNeedsSsoSignInError,
    InvitationNoLongerValidError,
)
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import OrganizationMemberRole, WorkspaceAssignmentRequest
from gateway.repositories.organizations import OrganizationRepositories
from gateway.services.organizations._admission import MembershipAdmission
from gateway.services.organizations._invitation_facts import InvitationFacts, InvitationFactsReader
from gateway.services.organizations._invitation_link import InvitationLink
from gateway.services.organizations._sso_enforcement import SsoEnforcement
from gateway.services.organizations._workspace_assignments import workspace_id_of


@dataclass(frozen=True, kw_only=True)
class AcceptingPerson:
    """The account accepting an invitation: who is signed in, and the address they hold."""

    email: str
    user_id: uuid.UUID


def _offered_workspaces(invitation: OrganizationInvitation) -> list[WorkspaceAssignmentRequest]:
    """The workspace grants an invitation offers, skipping any whose stored id is not one.

    The same reading of the column the preview makes, so the page and the acceptance offer the
    same workspaces.
    """
    return [
        WorkspaceAssignmentRequest(workspace_id=workspace_id, role=assignment["role"])
        for assignment in invitation.workspace_assignments
        if (workspace_id := workspace_id_of(assignment)) is not None
    ]


def _require_the_invited_address(invited: str, by: AcceptingPerson) -> None:
    """Refuse unless the person accepting holds the invited address."""
    if not same_address(invited, by.email):
        raise InvitationAddressMismatchError(invited)


@dataclass(frozen=True, kw_only=True)
class Acceptance:
    """What an invitee joined, for the page that confirmed it."""

    organization_id: uuid.UUID
    organization_name: str
    role: OrganizationMemberRole


@dataclass(frozen=True, kw_only=True)
class InvitationPreview:
    """What the page behind an invitation link shows before its holder decides."""

    facts: InvitationFacts
    invited_email: str


class InvitationAcceptance:
    """Answers an invitation link: the facts it offers, and the invitee's decline.

    NOTE: No answer and no refusal may depend on whether the invited address holds an account anywhere.
    Add no read to this service that could answer that question.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: OrganizationRepositories,
        *,
        admission: MembershipAdmission,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        facts: InvitationFactsReader,
        link: InvitationLink,
        sso: SsoEnforcement,
    ) -> None:
        self._uow = uow
        self._invitations = repositories.invitations
        self._organizations = repositories.organizations
        self._origins = repositories.membership_origins
        self._admission = admission
        self._clock = clock
        self._facts = facts
        self._link = link
        self._sso = sso

    async def preview(self, token: str) -> InvitationPreview:
        """What this invitation offers, raising the one refusal every unusable link gets."""
        async with self._uow:
            invitation = await self._valid_or_refused(token)
            facts = await self._facts.read(invitation, now=self._clock())
            if facts is None:
                raise InvitationNoLongerValidError()
            return InvitationPreview(facts=facts, invited_email=invitation.email)

    async def accept(self, token: str, *, by: AcceptingPerson) -> Acceptance:
        """Join the organization this link invites ``by`` to, on the terms it offers.

        The link works once: the invitation is removed, and whichever of two acceptances removes it
        is the one that joins. A workspace deleted since the offer was made is dropped.

        Raises:
            InvitationAddressMismatchError: the account accepting does not hold the invited address.
            InvitationNoLongerValidError: the link was never one, or it expired, was declined,
                cancelled, blocked, or already used.

        NOTE: This method owns its transaction. Its refusals are raised inside the block, which
        rolls back whatever that block holds even where a caller catches them, and its lock order
        is stated against this transaction alone. Callers must not call it inside one of their own.
        """
        now = self._clock()
        async with self._uow:
            invitation = await self._valid_or_refused(token)
            _require_the_invited_address(invitation.email, by)
            # Taken before the enforcement is read, because turning it on takes the same lock.
            await self._organizations.lock_settings(invitation.organization_id)
            organization = await self._organizations.get(invitation.organization_id)
            if organization is None:
                raise InvitationNoLongerValidError()
            if await self._sso.blocks_joining(
                organization_id=invitation.organization_id, role=invitation.role, user_id=by.user_id
            ):
                raise InvitationNeedsSsoSignInError(invitation.email)
            # Removing the row is what makes the link work once, and settles two acceptances at once.
            accepted = await self._invitations.delete_by_token_hash(invitation.token_hash)
            if accepted is None:
                raise InvitationNoLongerValidError()
            admitted = await self._admission.admit(
                organization_id=accepted.organization_id,
                role=accepted.role,
                user_id=by.user_id,
                workspaces=_offered_workspaces(accepted),
            )
            await self._origins.record(
                organization_member_id=admitted.member_id,
                invited_by_user_id=accepted.invited_by_user_id,
                invited_at=accepted.created_at,
                joined_at=now,
            )
            return Acceptance(organization_id=organization.id, organization_name=organization.name, role=accepted.role)

    async def decline(self, token: str) -> None:
        """Record that the invitee declined, which ends the link.

        The organization's admins see no change: the invitation stays pending until it expires.
        """
        async with self._uow:
            invitation = await self._valid_or_refused(token)
            if await self._invitations.decline_by_token_hash(invitation.token_hash, at=self._clock()) is None:
                raise InvitationNoLongerValidError()

    async def _valid_or_refused(self, token: str) -> OrganizationInvitation:
        """The invitation this token opens, refusing every unusable case with one answer.

        NOTE: Callers must call this inside an open Unit of Work block.
        """
        invitation = await self._link.open(token)
        if invitation is None:
            raise InvitationNoLongerValidError()
        return invitation
