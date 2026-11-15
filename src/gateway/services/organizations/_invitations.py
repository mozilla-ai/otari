"""Creating and listing an organization's invitations."""

import uuid
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from gateway.core.addresses import stored_address
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyError
from gateway.exceptions.identity_exceptions import InvalidEmailError
from gateway.exceptions.organizations_exceptions import (
    AddressAlreadyInvitedError,
    DuplicateInvitationAddressError,
    InvitationLimitCeiling,
    InvitationLimitError,
    InvitationRequestTooLargeError,
    InvitationsNeedSsoError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberAlreadyExistsError,
    WorkspaceNotFoundError,
)
from gateway.models.organizations import OrganizationInvitation, WorkspaceAssignment
from gateway.models.tenancy import (
    MANAGEMENT_ROLES,
    MAX_BULK_INVITATIONS,
    MAX_WORKSPACE_ASSIGNMENTS,
    OWNER_ROLE,
    OrganizationMemberRole,
    WorkspaceMemberRole,
)
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.services.mail import Mailer
from gateway.services.organizations._claimed_domains import ClaimedDomain
from gateway.services.organizations._invitation_limits import Admission, InvitationLimits
from gateway.services.organizations._send_token import Mint, mint_send
from gateway.services.organizations._workspace_assignments import workspace_id_of

# A page asked for more rows than this is still answered, just not with that many.
_MAX_PAGE_SIZE = 200


@dataclass(frozen=True, kw_only=True)
class ActingMember:
    """The caller an invitation is made or listed on behalf of.

    NOTE: Callers must set ``organization_id`` to the organization the caller acts in.
    """

    organization_id: uuid.UUID
    user_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class InvitationPolicy:
    """How long an invitation lives, and how often this organization may ask for one.

    Every field must be positive.
    """

    expires_after: timedelta
    per_address_per_day: int
    per_day: int
    resend_limit: int
    resend_wait: timedelta

    def __post_init__(self) -> None:
        for name, duration in (("expires_after", self.expires_after), ("resend_wait", self.resend_wait)):
            if duration <= timedelta(0):
                msg = f"{name} must be positive"
                raise ValueError(msg)
        for name, count in (
            ("per_address_per_day", self.per_address_per_day),
            ("per_day", self.per_day),
            ("resend_limit", self.resend_limit),
        ):
            if count < 1:
                msg = f"{name} must be positive"
                raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class OfferedWorkspace:
    """One workspace an invitation offers membership in, and the role it grants there."""

    role: WorkspaceMemberRole
    workspace_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class InvitationOffer:
    """The role an invitation grants in the organization, and the workspaces it offers alongside it."""

    role: OrganizationMemberRole
    workspaces: Sequence[OfferedWorkspace]


class InvitationState(StrEnum):
    """The state an invitation is shown in, to the organization's own admin.

    NOTE: This is not whether the invitation can be accepted.
    A declined invitation is still shown as pending.
    """

    EXPIRED = "expired"
    PENDING = "pending"


@dataclass(frozen=True, kw_only=True)
class InvitationEntry:
    """An invitation as the inviting organization's own admin sees it."""

    created_at: datetime
    email: str
    expires_at: datetime
    id: uuid.UUID
    resend_available_at: datetime | None
    role: OrganizationMemberRole
    state: InvitationState
    workspaces: tuple[OfferedWorkspace, ...]


@dataclass(frozen=True, kw_only=True)
class IssuedInvitation:
    """One invitation just created.

    ``email`` is the address exactly as the caller typed it; ``invitation.email`` is the
    normalized form that was stored. A caller lines the two up by address this way, the
    same way a refusal does.

    NOTE: ``accept_link`` is set only when this deployment cannot email the invitee.
    Whoever invited them must hand it on.
    """

    accept_link: str | None
    email: str
    invitation: InvitationEntry


@dataclass(frozen=True, kw_only=True)
class RefusedAddress:
    """One address a bulk invitation did not issue, and why."""

    email: str
    refusal: TenancyError


@dataclass(frozen=True, kw_only=True)
class BulkInvitations:
    """The result of inviting several addresses at once, each one issued or refused.

    Both lists keep the order the addresses were given in.
    """

    issued: list[IssuedInvitation]
    refused: list[RefusedAddress]


class OwnClaimedDomains(Protocol):
    """Answers which email domains an organization has claimed.

    NOTE: Deliberately narrow: it has no way to ask which organization proved a domain,
    so a caller cannot answer that question even by mistake.
    """

    async def for_organization(self, organization_id: uuid.UUID) -> list[ClaimedDomain]:
        """Return the organization's claims, in the order the claim surface lists them."""
        ...


@dataclass
class _Candidate:
    """One address through a bulk invitation call, in the order the caller gave it.

    Each field is set once its answer is known, and ``outcome`` holds the issued invitation or the refusal.
    ``replaces`` holds the id of an expired invitation for the same address, to delete before writing this one.
    """

    typed_email: str
    address: str | None = None
    admission: Admission | None = None
    mint: Mint | None = None
    replaces: uuid.UUID | None = None
    outcome: IssuedInvitation | TenancyError | None = None


class Invitations:
    """Creates and lists an organization's invitations.

    NOTE: A refusal must never name another organization or say whether an address holds an account anywhere.
    Add no read to this service that could answer either question.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: OrganizationRepositories,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        domains: OwnClaimedDomains,
        mailer: Mailer,
        policy: InvitationPolicy,
        store: RateLimitStorePort,
    ) -> None:
        self._uow = uow
        self._invitations = repositories.invitations
        self._members = repositories.members
        self._organizations = repositories.organizations
        self._workspaces = repositories.workspaces
        self._clock = clock
        self._domains = domains
        self._mailer = mailer
        self._policy = policy
        self._limits = InvitationLimits(store, per_day=policy.per_day, per_address_per_day=policy.per_address_per_day)

    async def invite(self, by: ActingMember, email: str, offer: InvitationOffer) -> IssuedInvitation:
        """Invite one address, raising the refusal that stopped it.

        NOTE: Callers must not call this from inside an already-open Unit of Work block.
        """
        result = await self._invite_many(by, [email], offer)
        if result.refused:
            raise result.refused[0].refusal
        return result.issued[0]

    async def invite_many(self, by: ActingMember, emails: Sequence[str], offer: InvitationOffer) -> BulkInvitations:
        """Invite every address, reporting each refusal and issuing the rest.

        A refusal on the whole request's own shape, the caller's authority,
        the organization's SSO requirement, or an offered workspace raises instead of naming an address.
        Every other refusal, the invitation limit included, is reported against the one address it stopped.

        NOTE: Callers must not call this from inside an already-open Unit of Work block.
        """
        return await self._invite_many(by, emails, offer)

    async def page(self, by: ActingMember, *, skip: int, limit: int) -> tuple[list[InvitationEntry], int]:
        """Return one page of the organization's invitations, oldest first.

        ``limit`` is bounded to ``_MAX_PAGE_SIZE``.

        NOTE: Raises a bare ``ValueError`` for a negative ``skip`` rather than one of this domain's own refusals.
        A caller must validate ``skip`` itself, or that error reaches a client unhandled.
        """
        if skip < 0:
            msg = "skip must not be negative"
            raise ValueError(msg)
        limit = max(1, min(limit, _MAX_PAGE_SIZE))
        now = self._clock()
        async with self._uow:
            await self._require_management(by)
            rows, total = await self._invitations.list_for_organization(by.organization_id, skip=skip, limit=limit)
        return [self._entry(row, now=now) for row in rows], total

    async def _invite_many(self, by: ActingMember, emails: Sequence[str], offer: InvitationOffer) -> BulkInvitations:
        """Invite every address, owning its own Unit of Work transaction from here to its return."""
        if len(emails) > MAX_BULK_INVITATIONS:
            raise InvitationRequestTooLargeError.too_many_addresses(MAX_BULK_INVITATIONS)
        self._check_offer_shape(offer)

        candidates = [_Candidate(typed_email=email) for email in emails]
        seen: set[str] = set()
        for candidate in candidates:
            address = stored_address(candidate.typed_email)
            if address is None:
                candidate.outcome = InvalidEmailError(candidate.typed_email)
            elif address in seen:
                candidate.outcome = DuplicateInvitationAddressError(candidate.typed_email)
            else:
                seen.add(address)
                candidate.address = address

        now = self._clock()
        # Read-only, and before any counting: every refusal a database read alone can decide,
        # so an address this organization's own state already refuses is never paid for below.
        async with self._uow:
            role = await self._require_management(by)
            if offer.role == OWNER_ROLE and role != OWNER_ROLE:
                raise MembershipUpdateError("Only organization owners can grant the owner role")
            if await self._organizations.is_sso_enforced(by.organization_id):
                raise InvitationsNeedSsoError()
            await self._check_workspaces(by.organization_id, offer.workspaces)
            await self._refuse_existing_members_and_invitations(candidates, by=by, now=now)

        async with AsyncExitStack() as stack:
            for candidate in candidates:
                if candidate.address is None or candidate.outcome is not None:
                    continue
                admission = await self._limits.admit(
                    account_id=by.user_id, email=candidate.address, organization_id=by.organization_id
                )
                if isinstance(admission, InvitationLimitCeiling):
                    candidate.outcome = InvitationLimitError(admission)
                    continue
                candidate.admission = await stack.enter_async_context(admission)

            async with self._uow:
                # Taken first, this lock makes the authority, owner-only and SSO checks below atomic with the write.
                # A role change takes `organizations.lock`, `FOR UPDATE`,
                # which conflicts with this call's `FOR NO KEY UPDATE` on the same row.
                await self._organizations.lock_settings(by.organization_id)
                role = await self._require_management(by)
                if offer.role == OWNER_ROLE and role != OWNER_ROLE:
                    raise MembershipUpdateError("Only organization owners can grant the owner role")
                if await self._organizations.is_sso_enforced(by.organization_id):
                    raise InvitationsNeedSsoError()
                await self._issue_admitted(candidates, by=by, offer=offer, now=now)

            # Only after the write above has returned (committed):
            # a failure between the first insert and the commit rolls every row back,
            # and must roll every count back with it rather than keep one for an invitation that no longer exists.
            for candidate in candidates:
                if isinstance(candidate.outcome, IssuedInvitation) and candidate.admission is not None:
                    candidate.admission.keep()

        return BulkInvitations(
            issued=[candidate.outcome for candidate in candidates if isinstance(candidate.outcome, IssuedInvitation)],
            refused=[
                RefusedAddress(email=candidate.typed_email, refusal=candidate.outcome)
                for candidate in candidates
                if isinstance(candidate.outcome, TenancyError)
            ],
        )

    async def _refuse_existing_members_and_invitations(
        self, candidates: list[_Candidate], *, by: ActingMember, now: datetime
    ) -> None:
        """Set the outcome of every candidate an active membership or an open invitation already refuses.

        Leaves ``replaces`` on a candidate whose organization holds an expired invitation for its address,
        so the write that follows knows to delete it before writing a fresh one.

        NOTE: Callers must call this inside an open Unit of Work block.
        """
        open_candidates = [c for c in candidates if c.address is not None and c.outcome is None]
        if not open_candidates:
            return
        addresses = [c.address for c in open_candidates if c.address is not None]
        active = await self._members.active_member_addresses(by.organization_id, addresses)
        for candidate in open_candidates:
            if candidate.address in active:
                candidate.outcome = OrganizationMemberAlreadyExistsError(candidate.typed_email)

        still_open = [c for c in open_candidates if c.outcome is None]
        existing = await self._invitations.get_by_organization_and_emails(
            by.organization_id, [c.address for c in still_open if c.address is not None]
        )
        existing_by_address = {row.email: row for row in existing}
        for candidate in still_open:
            if candidate.address is None:
                continue
            row = existing_by_address.get(candidate.address)
            if row is None:
                continue
            if row.expires_at > now:
                candidate.outcome = AddressAlreadyInvitedError(candidate.typed_email)
            else:
                candidate.replaces = row.id

    async def _issue_admitted(
        self, candidates: list[_Candidate], *, by: ActingMember, offer: InvitationOffer, now: datetime
    ) -> None:
        """Write every candidate still standing after the counting above, in one statement."""
        to_write = [(c.address, c) for c in candidates if c.address is not None and c.outcome is None]
        if not to_write:
            return
        replaced_ids = [candidate.replaces for _, candidate in to_write if candidate.replaces is not None]
        if replaced_ids:
            # NOTE: The result is unchecked.
            # A renewed row makes the insert below conflict, and a row already gone, from a cleanup or a cancel,
            # simply gets its new invitation.
            await self._invitations.delete_expired_in_organization(replaced_ids, by.organization_id, expired_by=now)

        news: list[NewInvitation] = []
        for address, candidate in to_write:
            mint = mint_send(self._mailer, now)
            candidate.mint = mint
            news.append(
                NewInvitation(
                    created_at=now,
                    email=address,
                    expires_at=now + self._policy.expires_after,
                    invited_by_user_id=by.user_id,
                    next_send_at=mint.next_send_at,
                    organization_id=by.organization_id,
                    role=offer.role,
                    token_hash=mint.token_hash,
                    workspace_assignments=[
                        WorkspaceAssignment(workspace_id=str(workspace.workspace_id), role=workspace.role)
                        for workspace in offer.workspaces
                    ],
                )
            )
        rows = await self._invitations.add_many(news)
        issued_by_address = {row.email: row for row in rows}
        for address, candidate in to_write:
            row = issued_by_address.get(address)
            if row is None or candidate.mint is None:
                candidate.outcome = AddressAlreadyInvitedError(candidate.typed_email)
                continue
            candidate.outcome = IssuedInvitation(
                accept_link=candidate.mint.accept_link,
                email=candidate.typed_email,
                invitation=self._entry(row, now=now),
            )

    async def _require_management(self, by: ActingMember) -> OrganizationMemberRole:
        """Return the caller's role, refusing unless it is an active owner or admin of the organization."""
        role = await self._members.active_role(by.organization_id, by.user_id)
        if role is None or role not in MANAGEMENT_ROLES:
            raise NotAuthorizedError()
        return role

    def _check_offer_shape(self, offer: InvitationOffer) -> None:
        if len(offer.workspaces) > MAX_WORKSPACE_ASSIGNMENTS:
            raise InvitationRequestTooLargeError.too_many_workspaces(MAX_WORKSPACE_ASSIGNMENTS)
        seen: set[uuid.UUID] = set()
        for workspace in offer.workspaces:
            if workspace.workspace_id in seen:
                raise InvitationRequestTooLargeError.duplicate_workspace(workspace.workspace_id)
            seen.add(workspace.workspace_id)

    async def _check_workspaces(self, organization_id: uuid.UUID, workspaces: Sequence[OfferedWorkspace]) -> None:
        if not workspaces:
            return
        rows = await self._workspaces.get_by_organization_and_ids(
            organization_id, [workspace.workspace_id for workspace in workspaces]
        )
        found = {row.id for row in rows}
        for workspace in workspaces:
            if workspace.workspace_id not in found:
                raise WorkspaceNotFoundError(workspace.workspace_id)

    def _entry(self, row: OrganizationInvitation, *, now: datetime) -> InvitationEntry:
        # NOTE: state ignores row.declined_at on purpose.
        # A decline is recorded quietly,
        # so a declined invitation reads as pending, then expired, exactly like one nobody has answered.
        state = InvitationState.EXPIRED if row.expires_at <= now else InvitationState.PENDING
        resend_available_at = (
            None
            if row.send_count > self._policy.resend_limit
            else row.last_send_requested_at + self._policy.resend_wait
        )
        workspaces = tuple(
            OfferedWorkspace(role=assignment["role"], workspace_id=workspace_id)
            for assignment in row.workspace_assignments
            if (workspace_id := workspace_id_of(assignment)) is not None
        )
        return InvitationEntry(
            created_at=row.created_at,
            email=row.email,
            expires_at=row.expires_at,
            id=row.id,
            resend_available_at=resend_available_at,
            role=row.role,
            state=state,
            workspaces=workspaces,
        )
