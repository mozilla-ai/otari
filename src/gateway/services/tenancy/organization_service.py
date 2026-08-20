"""Organizations: active-organization resolution, CRUD, and membership.

Rehomed from the platform's ``OrganizationService`` plus the membership half of
``OrganizationMembershipService``, converted to async. The authorization rules,
the membership constraints, and the response shapes are the platform's; what is
gone is the depth that has no home in the OSS base yet: mixpanel tracking,
managed provider-key and default-gateway provisioning, email-domain auto-join,
teams, and the org's budget and pricing surfaces. Those arrive with their own
slices, tracked under mozilla-ai/otari-ai#1452, and this service is where they
attach. Emailed invitations shipped here in mozilla-ai/otari#641 (see
``invite_active_organization_member_for_user``/``accept_invitation`` below);
``create_active_organization_member_for_user`` is the older, still-supported
immediate path this replaced no part of.

One rule runs through every method: a caller only ever acts inside the
organization their identity is currently pointed at, and no method takes an
organization id from the request, so a request cannot name another tenant's
organization at all.

A standalone deployment has one organization, provisioned at first boot, so
creating, switching between and deleting them are not part of this surface. They
are what make a deployment multi-tenant, and a self-hosted gateway is one tenant
with several people in it. The scoping below is written as if there could be
many, because the hosted edition has many and the schema is edition-invariant.
"""

import asyncio
import hashlib
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.tenancy import (
    MANAGEMENT_ROLES,
    AcceptInvitationResultPublic,
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberCreateResultPublic,
    ActiveOrganizationMemberPublic,
    ActiveOrganizationMembersPublic,
    ActiveOrganizationMemberUpdateRequest,
    CallerWorkspaceMembershipPublic,
    Invitation,
    InvitationPreviewPublic,
    InviteOrganizationMemberRequest,
    InviteOrganizationMemberResultPublic,
    Organization,
    OrganizationMember,
    OrganizationMembershipContextPublic,
    OrganizationPublic,
    User,
    WorkspaceAssignmentRequest,
    WorkspaceMemberUpdate,
)
from gateway.repositories.tenancy import (
    InvitationRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.users_repository import (
    get_or_create_attribution_user,
    live_attribution_user_ids,
)
from gateway.services.mail_service import send_mail
from gateway.services.tenancy.errors import (
    InvalidEmailError,
    InvitationAlreadyPendingError,
    InvitationAlreadyUsedError,
    InvitationExpiredError,
    InvitationNotFoundError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberAlreadyExistsError,
    OrganizationMemberNotFoundError,
    OrganizationNameRequiredError,
    OrganizationNotFoundError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.invitation_email import render_invitation_email

# A shape check, not an authority on deliverability: one @, something either
# side, a dot in the domain, and no whitespace. See InvalidEmailError.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validated_organization_name(name: str | None) -> str:
    """Reject a name that is blank once trimmed, rather than substituting one.

    ``min_length=1`` on the request admits a single space, and the fallback this
    replaced then stored the literal "Organization", renaming the organization to
    something the caller never asked for. ``workspace_service`` already refuses
    the same input, so the two agree now.
    """
    trimmed = (name or "").strip()
    if not trimmed:
        raise OrganizationNameRequiredError
    return trimmed


def _validated_email(email: str) -> str:
    """Normalize an address to lower case, refusing one that cannot be a handle."""
    candidate = email.strip().lower()
    if not _EMAIL_PATTERN.match(candidate):
        raise InvalidEmailError(email)
    return candidate


def _hash_invitation_token(token: str) -> str:
    """SHA-256 hex of an invitation token; only the hash is ever stored.

    Same reasoning as ``dashboard_session_service.hash_session_token``: a
    bearer-style secret sitting in a queryable column is the same risk class
    as a password, so it is hashed at rest and compared by hash.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def _invitation_accept_link(config: GatewayConfig, token: str) -> str:
    """Build the accept-invitation URL, absolute when the deployment knows its own address.

    Relative otherwise (``config.public_base_url`` unset): still a valid link
    an operator can share and a browser already on this dashboard can follow,
    just not one that means anything outside a browser, which is why sending
    it by email is gated on ``invitation_mail_ready`` rather than on this.
    """
    path = f"/#/accept-invitation?token={token}"
    if config.public_base_url:
        return f"{config.public_base_url.rstrip('/')}{path}"
    return path


# The most workspaces a switcher seed carries. Above the repository's paging
# default so the common deployment is never truncated, and bounded so one
# unusually large organization cannot make every context read unbounded.
CALLER_WORKSPACE_LIMIT = 1000


class OrganizationService:
    """Business logic for the organization surface."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationRepository(db)
        self.members = OrganizationMemberRepository(db)
        self.users = UserRepository(db)
        self.workspaces = WorkspaceMemberRepository(db)
        self.workspace_rows = WorkspaceRepository(db)
        self.invitations = InvitationRepository(db)

    # ------------------------------------------------------------------
    # Context resolution and authorization
    # ------------------------------------------------------------------

    async def get_active_organization_for_user(self, user: User) -> Organization:
        """Return the organization the caller is pointed at, if their membership is live.

        Deliberately does not bootstrap or auto-switch: a page that reads the
        active organization must not create one as a side effect.
        """
        organization = await self.organizations.get(user.active_organization_id)
        if organization is None:
            raise OrganizationNotFoundError(user.active_organization_id)
        await self._require_active_membership(user, organization)
        return organization

    async def _require_active_membership(self, user: User, organization: Organization) -> OrganizationMember:
        membership = await self.members.get_active_by_organization_and_user(organization.id, user.id)
        if membership is None:
            raise NotAuthorizedError(f"Identity {user.id} has no active membership in this organization")
        return membership

    @staticmethod
    def _enforce_management_role(membership: OrganizationMember, user: User) -> None:
        if membership.role not in MANAGEMENT_ROLES and not user.is_superuser:
            raise NotAuthorizedError

    async def require_active_organization_management_access(
        self,
        *,
        user: User,
        organization: Organization,
    ) -> OrganizationMember:
        """Return the caller's membership, refusing unless it may manage the organization."""
        membership = await self._require_active_membership(user, organization)
        self._enforce_management_role(membership, user)
        return membership

    async def user_has_active_membership(self, *, organization_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Whether an identity is an active member of an organization."""
        return await self.members.get_active_by_organization_and_user(organization_id, user_id) is not None

    @staticmethod
    def _to_context(
        *,
        membership: OrganizationMember,
        organization: Organization,
        workspace_memberships: list[CallerWorkspaceMembershipPublic],
    ) -> OrganizationMembershipContextPublic:
        return OrganizationMembershipContextPublic(
            organization_member_id=membership.id,
            role=membership.role,
            status=membership.status,
            organization=OrganizationPublic.model_validate(organization),
            workspace_memberships=workspace_memberships,
            # The platform answers "does this org have a self-hosted gateway
            # attached". A standalone deployment reading this *is* that gateway,
            # so its own provider credentials are always available to it.
            byo_provider_keys_allowed=True,
        )

    async def _caller_workspace_memberships(
        self,
        *,
        user: User,
        organization: Organization,
    ) -> list[CallerWorkspaceMembershipPublic]:
        """The workspaces the caller belongs to, with the name and their role.

        Two queries rather than a join helper, reusing the repository methods the
        workspace surface already has. Only the caller's own memberships, so a
        shell can pick a default workspace without being handed a directory of
        the organization's workspaces: listing those is a separate, authorized
        read.
        """
        # Explicit limit, because the repository's default is 100 and this is a
        # switcher seed rather than a page: silently dropping the caller's 101st
        # workspace would hide it from every context the shell can select.
        memberships, _ = await self.workspaces.get_workspaces_for_user(
            user_id=user.id,
            organization_id=organization.id,
            limit=CALLER_WORKSPACE_LIMIT,
        )
        if not memberships:
            return []
        names = {
            workspace.id: workspace.name
            for workspace in await self.workspace_rows.get_by_ids([m.workspace_id for m in memberships])
        }
        return [
            CallerWorkspaceMembershipPublic(
                workspace_id=membership.workspace_id,
                name=names[membership.workspace_id],
                role=membership.role,
            )
            for membership in memberships
            if membership.workspace_id in names
        ]

    async def _resolve_active_organization(self, user: User) -> Organization:
        """Resolve the caller's organization, repairing a stale pointer if it can.

        The reachable stale case is a membership that was suspended after the
        pointer was set: falling back to the caller's oldest live membership
        keeps the dashboard usable instead of refusing every page. A pointer at a
        *deleted* organization is not reachable at all in this edition, since
        nothing deletes one and the foreign key would refuse; the check survives
        for a database that arrived by another route.

        Unlike the platform this never provisions an organization as a fallback:
        first boot owns that (see `provisioning_service`), and every later
        identity is created inside the one that already exists.
        """
        organization = await self.organizations.get(user.active_organization_id)
        if organization is not None:
            membership = await self.members.get_active_by_organization_and_user(organization.id, user.id)
            if membership is not None:
                return organization

        fallback = await self.members.get_first_active_for_user(user.id)
        if fallback is None:
            raise OrganizationNotFoundError(user.active_organization_id)
        recovered = await self.organizations.get(fallback.organization_id)
        if recovered is None:
            raise OrganizationNotFoundError(fallback.organization_id)

        await self.users.set_active_organization(user, recovered.id)
        await self.db.commit()
        return recovered

    async def get_active_membership_context_for_user(self, user: User) -> OrganizationMembershipContextPublic:
        """Return the caller's organization and their standing in it."""
        organization = await self._resolve_active_organization(user)
        membership = await self._require_active_membership(user, organization)
        return self._to_context(
            membership=membership,
            organization=organization,
            workspace_memberships=await self._caller_workspace_memberships(user=user, organization=organization),
        )

    # ------------------------------------------------------------------
    # The organization itself
    # ------------------------------------------------------------------

    async def update_active_organization_for_user(
        self,
        *,
        user: User,
        organization_name: str,
    ) -> OrganizationMembershipContextPublic:
        """Rename the caller's organization."""
        organization = await self.get_active_organization_for_user(user)
        membership = await self.require_active_organization_management_access(user=user, organization=organization)

        updated = await self.organizations.update_organization(
            organization,
            {"name": _validated_organization_name(organization_name)},
        )
        await self.db.commit()

        return self._to_context(
            membership=membership,
            organization=updated,
            workspace_memberships=await self._caller_workspace_memberships(user=user, organization=updated),
        )

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    async def list_active_organization_members_for_user(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> ActiveOrganizationMembersPublic:
        """List the organization's roster. Any active member may read it."""
        organization = await self.get_active_organization_for_user(user)

        rows, count = await self.members.get_by_organization_with_users(organization.id, skip=skip, limit=limit)
        # One query for the whole page rather than a lookup per row: the roster is
        # the picker the dashboard builds its key-owner list from, so every row
        # needs to say whether it can own a key.
        live = await live_attribution_user_ids(self.db, [str(member_user.id) for _, member_user in rows])
        # Likewise for which invited rows have a pending invitation to revoke:
        # one query for the page's invited memberships rather than one per row.
        pending = await self.invitations.get_pending_by_organization_members(
            membership.id for membership, _ in rows if membership.status == "invited"
        )
        invitation_by_member = {invitation.organization_member_id: invitation.id for invitation in pending}
        return ActiveOrganizationMembersPublic(
            data=[
                self._to_member_public(
                    membership,
                    member_user,
                    live=live,
                    invitation_id=invitation_by_member.get(membership.id),
                )
                for membership, member_user in rows
            ],
            count=count,
        )

    async def create_active_organization_member_for_user(
        self,
        *,
        user: User,
        request: ActiveOrganizationMemberCreateRequest,
    ) -> ActiveOrganizationMemberCreateResultPublic:
        """Add someone to the caller's organization, by address.

        The platform's two branches both end at a membership that has to be
        accepted: a known address gets an ``invited`` membership, an unknown one
        gets an emailed invitation. Neither half exists here, and a membership
        nobody can accept is a dead state, so both branches land ``active``
        instead and an unknown address creates a local identity carrying it (M4's
        claimable identity). That identity cannot authenticate until the sign-in
        flow lands; until then it is a roster and attribution entry, which is
        what the gateway's own ``users`` are today.

        Any workspace assignments are applied in the same transaction rather than
        parked, since there is no acceptance to park them until.
        """
        organization = await self.get_active_organization_for_user(user)
        actor_membership = await self.require_active_organization_management_access(
            user=user,
            organization=organization,
        )

        email = _validated_email(request.email)
        assignments = request.workspace_assignments or []
        await self._require_workspaces_in_organization(organization, assignments)

        target = await self.users.get_by_email(email)
        try:
            if target is None:
                target = await self.users.create_local_identity(
                    full_name=None,
                    email=email,
                    active_organization_id=organization.id,
                )

            membership = await self.members.get_by_organization_and_user(organization.id, target.id)
            if membership is not None and membership.status == "active":
                raise OrganizationMemberAlreadyExistsError(email)

            if membership is None:
                # The same rule the revive branch gets from
                # `_validate_membership_update`: an admin may not mint an owner.
                # There is no membership to validate against yet, so the one
                # applicable clause is applied directly.
                if actor_membership.role != "owner" and request.role == "owner":
                    raise MembershipUpdateError("Only organization owners can grant the owner role")
                membership = await self.members.create_membership(
                    organization_id=organization.id,
                    user_id=target.id,
                    role=request.role,
                )
            else:
                # Re-adding someone who was removed: removal suspends the
                # membership rather than deleting it, so this revives that row
                # and keeps their history attached to it.
                #
                # Through the same guard PATCH and DELETE use, because this
                # branch also writes a role. Without it, adding an address whose
                # membership is suspended lets an admin rewrite an owner's role,
                # which PATCH refuses; the write is the same, so the rule is.
                await self._validate_membership_update(
                    actor_membership=actor_membership,
                    target_membership=membership,
                    update_data={"role": request.role, "status": "active"},
                    organization_id=organization.id,
                )
                membership = await self.members.update_membership(
                    membership,
                    {"role": request.role, "status": "active"},
                )

            # After both branches, not inside either: keyed on the identity's
            # UUID, so the create path mints and the revive path finds the row it
            # minted the first time rather than a second one.
            attribution = await get_or_create_attribution_user(
                self.db,
                user_id=str(target.id),
                alias=email,
            )

            await self._apply_workspace_assignments(user_id=target.id, assignments=assignments)
            await self.db.commit()
        except IntegrityError:
            # Two admins adding the same address at once: the unique index on
            # email, or on (organization, user), decides, and the loser reports
            # the conflict rather than a 500.
            await self.db.rollback()
            raise OrganizationMemberAlreadyExistsError(email) from None

        return ActiveOrganizationMemberCreateResultPublic(
            status="active",
            organization_member_id=membership.id,
            user_id=target.id,
            attribution_user_id=attribution.user_id,
            email=email,
            full_name=target.full_name,
            role=membership.role,
            created_at=membership.created_at,
            updated_at=membership.updated_at,
        )

    async def _require_workspaces_in_organization(
        self,
        organization: Organization,
        assignments: list[WorkspaceAssignmentRequest],
    ) -> None:
        """Refuse the whole request if an assignment names a workspace elsewhere.

        Checked before anything is written, so a foreign or unknown workspace id
        fails the add rather than silently dropping that one grant. A workspace
        in another organization is reported as not found, like everywhere else.
        """
        if not assignments:
            return
        requested = {assignment.workspace_id for assignment in assignments}
        found = {
            workspace.id
            for workspace in await WorkspaceRepository(self.db).get_by_ids(requested)
            if workspace.organization_id == organization.id
        }
        missing = requested - found
        if missing:
            raise WorkspaceNotFoundError(next(iter(sorted(missing, key=str))))

    async def _drop_vanished_workspace_assignments(
        self,
        organization: Organization,
        assignments: list[WorkspaceAssignmentRequest],
    ) -> list[WorkspaceAssignmentRequest]:
        """Keep only the assignments whose workspace still exists in this organization.

        The accept counterpart to `_require_workspaces_in_organization`, and
        deliberately not the same behavior: an assignment parked on an
        invitation can be up to `invitation_expiry_hours` stale (a week by
        default) by the time the recipient follows the link, on a public
        endpoint with no operator present to correct anything. Raising there
        would leave the invitation permanently un-acceptable (nothing retries
        with a smaller set) and would name the missing workspace's id in the
        4xx body `_tenancy_error_handler` passes through verbatim, to a caller
        who has only ever held a token. Dropping the vanished assignment and
        applying the rest instead lands the invitee as a member with one
        grant missing, which an operator can restore from the workspace
        roster once they notice. Invite-time keeps the strict check: that
        caller is present in the same request to fix a bad workspace id.
        """
        if not assignments:
            return assignments
        found = {
            workspace.id
            for workspace in await WorkspaceRepository(self.db).get_by_ids(
                {assignment.workspace_id for assignment in assignments}
            )
            if workspace.organization_id == organization.id
        }
        return [assignment for assignment in assignments if assignment.workspace_id in found]

    async def _apply_workspace_assignments(
        self,
        *,
        user_id: uuid.UUID,
        assignments: list[WorkspaceAssignmentRequest],
    ) -> None:
        """Grant each assigned workspace, reviving a suspended membership.

        Deduplicated by workspace first, keeping the first role named for one, and
        every existing membership is resolved in one query before the loop, so a
        body naming N workspaces costs one lookup rather than N.
        `_require_workspaces_in_organization` already resolves the same set
        rather than the list.

        An existing membership is updated rather than skipped, as the platform's
        own assignment path does: a suspended row that is left alone would leave
        the member listed in a workspace they were just granted while still
        being refused everything in it.
        """
        members = WorkspaceMemberRepository(self.db)
        wanted: dict[uuid.UUID, str] = {}
        for assignment in assignments:
            wanted.setdefault(assignment.workspace_id, assignment.role)

        # One IN query rather than one lookup per assignment: with the ceiling at
        # MAX_WORKSPACE_ASSIGNMENTS that is the difference between one round trip
        # and fifty inside a single request.
        existing_by_workspace = {
            member.workspace_id: member for member in await members.get_by_workspaces_and_user(wanted, user_id)
        }
        for workspace_id, role in wanted.items():
            existing = existing_by_workspace.get(workspace_id)
            if existing is not None:
                await members.update(existing, WorkspaceMemberUpdate(role=role, status="active"))
                continue
            await members.create(workspace_id=workspace_id, user_id=user_id, role=role)

    async def invite_active_organization_member_for_user(
        self,
        *,
        user: User,
        request: InviteOrganizationMemberRequest,
        config: GatewayConfig,
    ) -> InviteOrganizationMemberResultPublic:
        """Invite an address to the caller's organization by email.

        Unlike ``create_active_organization_member_for_user``, the membership
        lands ``invited`` rather than ``active``: an ``Invitation`` row is
        created alongside it, and only ``accept_invitation`` (below) flips the
        pair to ``active``. Workspace assignments are parked on the invitation
        rather than applied now, for the same reason there is nothing yet to
        grant them to.

        Refuses an address that already holds an active membership, or one
        with a still-unexpired invitation pending: resending on purpose is
        revoke (which cancels the pending invitation and suspends the
        membership) followed by a fresh invite. An address whose invitation
        has expired unaccepted is not refused, though: this supersedes it
        directly through the same "revive a suspended membership" branch
        below that re-adding a removed address already goes through, since
        without that a link nobody ever opened would dead-end every future
        invite to the same address with no way through but revoke-then-invite.
        """
        organization = await self.get_active_organization_for_user(user)
        actor_membership = await self.require_active_organization_management_access(
            user=user,
            organization=organization,
        )

        email = _validated_email(request.email)
        assignments = request.workspace_assignments or []
        await self._require_workspaces_in_organization(organization, assignments)

        if actor_membership.role != "owner" and request.role == "owner":
            raise MembershipUpdateError("Only organization owners can grant the owner role")

        target = await self.users.get_by_email(email)
        try:
            if target is None:
                target = await self.users.create_local_identity(
                    full_name=None,
                    email=email,
                    active_organization_id=organization.id,
                )

            # Locked before the status check that decides create/revive/refuse,
            # not just inside the revive branch below: two concurrent invites to
            # the same suspended membership can otherwise both read "suspended"
            # (nothing has committed yet to see), both fall through to revive
            # it, and both mint their own live pending invitation for the one
            # membership, since organization_member_id carries no uniqueness to
            # catch that as an IntegrityError instead. The second caller through
            # this lock re-reads the membership fresh, so it sees the first
            # caller's write.
            await self.organizations.lock(organization.id)
            membership = await self.members.get_by_organization_and_user(organization.id, target.id)
            if membership is not None and membership.status == "active":
                raise OrganizationMemberAlreadyExistsError(email)
            if membership is not None and membership.status == "invited":
                # Expiry is lazy: `_resolve_pending_invitation` only flips a
                # `pending` row to `expired` when someone presents its token,
                # so a link nobody ever opened can sit `pending` in the
                # database indefinitely with its `expires_at` already in the
                # past. Re-checking the timestamp here, not the stored status,
                # is what keeps re-inviting from dead-ending on an
                # unaccepted, unopened, long-expired link forever.
                pending = await self.invitations.get_pending_by_organization_members([membership.id])
                now = datetime.now(UTC)
                if any(invitation.expires_at >= now for invitation in pending):
                    raise InvitationAlreadyPendingError(email)
                # Every row here is stale; expire them explicitly so this
                # fresh invite is the only `pending` one for the membership,
                # rather than leaving one whose own timestamp has already
                # passed to fight the new one over which invitation_id the
                # roster shows.
                for stale in pending:
                    await self.invitations.update_status(stale, {"status": "expired"})

            if membership is None:
                membership = await self.members.create_membership(
                    organization_id=organization.id,
                    user_id=target.id,
                    role=request.role,
                    status="invited",
                )
            else:
                # Reviving a suspended membership: the same guard the plain
                # add-member revive branch uses, since this also writes a role.
                await self._validate_membership_update(
                    actor_membership=actor_membership,
                    target_membership=membership,
                    update_data={"role": request.role, "status": "invited"},
                    organization_id=organization.id,
                )
                membership = await self.members.update_membership(
                    membership,
                    {"role": request.role, "status": "invited"},
                )

            token = secrets.token_urlsafe(32)
            expires_at = datetime.now(UTC) + timedelta(hours=config.invitation_expiry_hours)
            invitation = await self.invitations.create_invitation(
                organization_id=organization.id,
                organization_member_id=membership.id,
                email=email,
                invited_by_user_id=user.id,
                token_hash=_hash_invitation_token(token),
                workspace_assignments=[assignment.model_dump(mode="json") for assignment in assignments],
                expires_at=expires_at,
            )
            await self.db.commit()
        except IntegrityError:
            # Two admins inviting the same address at once: the unique index on
            # (organization, user) decides which racer's insert wins, and the
            # loser reports the conflict rather than a 500. The row lock taken
            # above is what actually decides the invited-vs-suspended-vs-active
            # question this branch answers; `Invitation.organization_member_id`
            # itself carries no uniqueness (see its own comment on the model),
            # since a membership can be invited, revoked, and re-invited more
            # than once over its life.
            await self.db.rollback()
            raise OrganizationMemberAlreadyExistsError(email) from None

        accept_link = _invitation_accept_link(config, token)
        mail_sent = False
        if config.invitation_mail_ready:
            subject, html, text = render_invitation_email(
                organization_name=organization.name,
                inviter_name=user.full_name or "An organization admin",
                role=membership.role,
                accept_link=accept_link,
                expiry_hours=config.invitation_expiry_hours,
            )
            # send_mail is synchronous socket I/O (smtplib), and each of
            # connect/starttls/login/sendmail carries its own 10s timeout, so
            # calling it directly here could block this request's whole event
            # loop thread for tens of seconds against a slow or unreachable
            # SMTP host, stalling every other request the same worker is
            # serving. Off-loaded to a thread so only this coroutine waits.
            mail_sent = await asyncio.to_thread(send_mail, config, to=email, subject=subject, html=html, text=text)

        return InviteOrganizationMemberResultPublic(
            invitation_id=invitation.id,
            organization_member_id=membership.id,
            email=email,
            role=membership.role,
            mail_sent=mail_sent,
            accept_link=accept_link,
            expires_at=invitation.expires_at,
            created_at=invitation.created_at,
        )

    async def _resolve_pending_invitation(self, token: str) -> tuple[Invitation, OrganizationMember, Organization]:
        """Look up a still-acceptable invitation by token, or raise why not.

        Shared by the preview and accept paths so the two answer identically:
        an unknown or foreign token collapses into one ``InvitationNotFoundError``
        (see that error's docstring for why), and an invitation past its status
        or its expiry raises the specific reason once here rather than being
        checked twice by two callers.
        """
        invitation = await self.invitations.get_by_token_hash(_hash_invitation_token(token))
        if invitation is None:
            raise InvitationNotFoundError

        if invitation.status != "pending":
            raise InvitationAlreadyUsedError

        if invitation.expires_at < datetime.now(UTC):
            await self.invitations.update_status(invitation, {"status": "expired"})
            await self.db.commit()
            raise InvitationExpiredError

        membership = await self.members.get(invitation.organization_member_id)
        organization = await self.organizations.get(invitation.organization_id)
        if membership is None or organization is None:
            raise InvitationNotFoundError
        return invitation, membership, organization

    async def get_invitation_preview(self, token: str) -> InvitationPreviewPublic:
        """Look up a pending invitation by token, for the accept page. No auth: the token is the proof."""
        invitation, membership, organization = await self._resolve_pending_invitation(token)
        return InvitationPreviewPublic(
            email=invitation.email,
            organization_name=organization.name,
            role=membership.role,
            expires_at=invitation.expires_at,
        )

    async def accept_invitation(self, token: str) -> AcceptInvitationResultPublic:
        """Resolve a pending invitation to an active membership.

        No session is minted (see ``AcceptInvitationResultPublic``): this only
        flips the paired membership to ``active`` and applies the parked
        workspace assignments, the same way immediate ones are applied on
        ``POST /me/members``.
        """
        _, _, organization = await self._resolve_pending_invitation(token)
        # Locked, then re-resolved, before any write: two concurrent accepts of
        # the same token could otherwise both pass the pending check above
        # (nothing has committed yet to see), both flip the membership, and
        # both reach _apply_workspace_assignments, whose existing-then-create
        # shape lets the second racer's insert violate
        # uq_workspace_member_workspace_user as an uncaught IntegrityError on
        # this public, unauthenticated endpoint. Same pattern as the
        # organization lock in invite_active_organization_member_for_user; the
        # second caller through the lock re-resolves and finds the invitation
        # no longer pending, so it raises InvitationAlreadyUsedError instead.
        await self.organizations.lock(organization.id)
        invitation, membership, organization = await self._resolve_pending_invitation(token)

        membership = await self.members.update_membership(membership, {"status": "active"})
        assignments = [
            WorkspaceAssignmentRequest.model_validate(assignment) for assignment in invitation.workspace_assignments
        ]
        # Re-checked rather than trusted: these ids were validated against the
        # organization at invite time, but acceptance can arrive up to
        # invitation_expiry_hours later (seven days by default), and a
        # workspace named in them may have been deleted since. Dropped rather
        # than refused: see _drop_vanished_workspace_assignments for why a
        # 404 here would be both wrong (it would leak a workspace id to an
        # unauthenticated caller) and worse than useless (it would leave the
        # invitation permanently stuck pending).
        assignments = await self._drop_vanished_workspace_assignments(organization, assignments)
        await self._apply_workspace_assignments(user_id=membership.user_id, assignments=assignments)
        # Same as the immediate-add path: keyed on the identity's UUID, so an
        # address invited and later re-invited (revoke, then re-add) finds the
        # row it minted the first time rather than a second one. Without this,
        # an accepted invitee's roster row would carry no attribution_user_id
        # and could never be offered as a key owner, unlike a member added
        # directly through POST /me/members.
        await get_or_create_attribution_user(self.db, user_id=str(membership.user_id), alias=invitation.email)
        await self.invitations.update_status(invitation, {"status": "accepted"})
        await self.db.commit()

        return AcceptInvitationResultPublic(organization_name=organization.name, role=membership.role)

    async def revoke_organization_member_invitation_for_user(
        self,
        *,
        user: User,
        invitation_id: uuid.UUID,
    ) -> None:
        """Revoke an unaccepted invitation. Organization owners and admins only.

        Cancels the invitation and suspends its paired membership, mirroring
        ``remove_active_organization_member_for_user``'s suspend-not-delete
        reasoning: re-inviting the same address later revives it rather than
        starting over. Goes through the same ``_validate_membership_update``
        guard that path uses, too: without it, an admin (who already passes
        the organization-management check above) could suspend a pending
        *owner*-role invitation, which every other membership-status write
        refuses ("only an owner outranks an owner").
        """
        organization = await self.get_active_organization_for_user(user)
        actor_membership = await self.require_active_organization_management_access(
            user=user,
            organization=organization,
        )

        # Locked before the invitation is read, not only inside
        # _validate_membership_update below: accept_invitation holds this same
        # lock across its own read-then-write, so taking it first here means
        # this call runs entirely before that accept commits or entirely
        # after, never in between. Without this, the read just below could see
        # a still-"pending" invitation, block on the lock while a concurrent
        # accept commits, and then unconditionally overwrite the now-accepted
        # membership back to "suspended" and the invitation back to
        # "cancelled". Re-acquiring the same row lock a few lines later, inside
        # _validate_membership_update, is a no-op within one transaction.
        await self.organizations.lock(organization.id)

        invitation = await self.invitations.get(invitation_id)
        if invitation is None or invitation.organization_id != organization.id:
            raise InvitationNotFoundError(invitation_id)
        if invitation.status != "pending":
            raise InvitationAlreadyUsedError

        membership = await self.members.get_by_id_and_organization(invitation.organization_member_id, organization.id)
        if membership is not None:
            await self._validate_membership_update(
                actor_membership=actor_membership,
                target_membership=membership,
                update_data={"status": "suspended"},
                organization_id=organization.id,
            )
            await self.members.update_membership(membership, {"status": "suspended"})
        await self.invitations.update_status(invitation, {"status": "cancelled"})
        await self.db.commit()

    async def _cancel_pending_invitation_for_membership(self, organization_member_id: uuid.UUID) -> None:
        """Cancel a membership's pending invitation, if it has one.

        Called wherever a membership is suspended by a path other than
        ``revoke_organization_member_invitation_for_user`` (namely, removing an
        `invited` member the same way an `active` one is removed). Without
        this, the membership goes to `suspended` while its invitation stays
        `pending`, and the emailed link still works: accepting it would flip
        the membership back to `active`, silently undoing the removal.
        """
        pending = await self.invitations.get_pending_by_organization_members([organization_member_id])
        for invitation in pending:
            await self.invitations.update_status(invitation, {"status": "cancelled"})

    async def update_active_organization_member_for_user(
        self,
        *,
        user: User,
        organization_member_id: uuid.UUID,
        request: ActiveOrganizationMemberUpdateRequest,
    ) -> ActiveOrganizationMemberPublic:
        """Change a member's role or status."""
        organization = await self.get_active_organization_for_user(user)
        actor_membership = await self.require_active_organization_management_access(
            user=user,
            organization=organization,
        )

        target = await self.members.get_by_id_and_organization(organization_member_id, organization.id)
        if target is None:
            raise OrganizationMemberNotFoundError(organization_member_id)

        # ``exclude_unset`` keeps an explicit ``null`` (the generated client types
        # both fields as nullable, so a form that clears one sends it), and both
        # columns are NOT NULL, so passing it through reached the database as an
        # integrity error rather than as "leave this field alone".
        update_data = {key: value for key, value in request.model_dump(exclude_unset=True).items() if value is not None}
        await self._validate_membership_update(
            actor_membership=actor_membership,
            target_membership=target,
            update_data=update_data,
            organization_id=organization.id,
        )

        # Snapshotted before the write: `update_membership` mutates `target` in
        # place (SQLModel `sqlmodel_update` + `refresh`), so `target.status`
        # itself would already read the new value afterwards.
        was_invited = target.status == "invited"
        updated = await self.members.update_membership(target, update_data)
        # Any transition away from `invited` through this generic path, not
        # only to `suspended`: `OrganizationMemberSettableStatus` also lets a
        # caller PATCH straight to `active`, bypassing accept_invitation
        # entirely. Left uncancelled, the invitation stays `pending` and its
        # token still resolves; if the membership is later removed by any
        # path, accepting it would silently reactivate the membership and
        # re-apply the parked workspace grants nobody re-confirmed.
        if was_invited and updated.status != "invited":
            await self._cancel_pending_invitation_for_membership(updated.id)
        target_user = await self.users.get(updated.user_id)
        if target_user is None:
            raise OrganizationMemberNotFoundError(organization_member_id)
        await self.db.commit()

        live = await live_attribution_user_ids(self.db, [str(target_user.id)])
        invitation_id = None
        if updated.status == "invited":
            pending = await self.invitations.get_pending_by_organization_members([updated.id])
            invitation_id = pending[0].id if pending else None
        return self._to_member_public(updated, target_user, live=live, invitation_id=invitation_id)

    async def remove_active_organization_member_for_user(
        self,
        *,
        user: User,
        organization_member_id: uuid.UUID,
    ) -> None:
        """Remove a member by suspending their membership.

        Suspension rather than deletion, as on the platform: the row is what
        every historical attribution resolves through, so it outlives the access
        it granted. If the membership was `invited`, its pending invitation is
        cancelled in the same transaction (see
        ``_cancel_pending_invitation_for_membership``): the dashboard routes an
        invited row to Revoke instead, which does the same thing, but this
        stays correct for a caller that removes one directly.
        """
        organization = await self.get_active_organization_for_user(user)
        actor_membership = await self.require_active_organization_management_access(
            user=user,
            organization=organization,
        )

        target = await self.members.get_by_id_and_organization(organization_member_id, organization.id)
        if target is None:
            raise OrganizationMemberNotFoundError(organization_member_id)

        await self._validate_membership_update(
            actor_membership=actor_membership,
            target_membership=target,
            update_data={"status": "suspended"},
            organization_id=organization.id,
        )

        was_invited = target.status == "invited"
        await self.members.update_membership(target, {"status": "suspended"})
        if was_invited:
            await self._cancel_pending_invitation_for_membership(target.id)
        await self.db.commit()

    async def _validate_membership_update(
        self,
        *,
        actor_membership: OrganizationMember,
        target_membership: OrganizationMember,
        update_data: dict[str, str],
        organization_id: uuid.UUID,
    ) -> None:
        """Refuse the membership changes that would break the organization.

        An admin cannot act on an owner (only an owner outranks an owner), an
        admin cannot *make* an owner either, and the last active owner cannot be
        demoted or deactivated, which would leave the organization with nobody
        able to manage or delete it.
        """
        # Serialized on the parent row before anything is read: the last-owner
        # rule below is read-then-write with no unique index behind it, so
        # without this two concurrent demotions of two different owners both
        # count two and both commit. See ``OrganizationRepository.lock``.
        await self.organizations.lock(organization_id)

        new_role = update_data.get("role", target_membership.role)
        new_status = update_data.get("status", target_membership.status)

        if actor_membership.role != "owner" and target_membership.role == "owner":
            raise MembershipUpdateError("Only organization owners can modify owner memberships")

        # Deliberately narrower than the platform, whose guard reads the target's
        # *current* role alone: there, an admin may promote anyone, themselves
        # included, to owner, and an owner may not be removed by an admin
        # afterwards. That is privilege escalation with a lock on the door behind
        # it. Unreachable while one bootstrap operator is the only identity that
        # can authenticate, and reachable the day per-identity sign-in lands
        # (otari-ai#1716), which makes this the cheap moment to close it. The
        # narrowing is widenable later; the escalation would not be.
        if actor_membership.role != "owner" and new_role == "owner":
            raise MembershipUpdateError("Only organization owners can grant the owner role")

        target_is_active_owner = target_membership.role == "owner" and target_membership.status == "active"
        if target_is_active_owner and (new_role != "owner" or new_status != "active"):
            if await self.members.count_active_owners(organization_id) <= 1:
                raise MembershipUpdateError("An organization must keep at least one active owner")

    @staticmethod
    def _to_member_public(
        membership: OrganizationMember,
        user: User,
        *,
        live: set[str],
        invitation_id: uuid.UUID | None = None,
    ) -> ActiveOrganizationMemberPublic:
        attribution_user_id = str(user.id)
        return ActiveOrganizationMemberPublic(
            organization_member_id=membership.id,
            user_id=user.id,
            attribution_user_id=attribution_user_id if attribution_user_id in live else None,
            invitation_id=invitation_id,
            email=user.email,
            full_name=user.full_name,
            role=membership.role,
            status=membership.status,
            created_at=membership.created_at,
            updated_at=membership.updated_at,
        )


__all__ = ["OrganizationService"]
