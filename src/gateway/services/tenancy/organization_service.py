"""Organizations: active-organization resolution, CRUD, and membership.

Rehomed from the platform's ``OrganizationService`` plus the membership half of
``OrganizationMembershipService``, converted to async. The authorization rules,
the membership constraints, and the response shapes are the platform's; what is
gone is the depth that has no home in the OSS base yet: mixpanel tracking,
managed provider-key and default-gateway provisioning, email-domain auto-join,
emailed invitations, teams, and the org's budget and pricing surfaces. Those
arrive with their own slices, tracked under mozilla-ai/otari-ai#1452, and this
service is where they attach.

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

import re
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import (
    MANAGEMENT_ROLES,
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberCreateResultPublic,
    ActiveOrganizationMemberPublic,
    ActiveOrganizationMembersPublic,
    ActiveOrganizationMemberUpdateRequest,
    Organization,
    OrganizationMember,
    OrganizationMembershipContextPublic,
    OrganizationPublic,
    User,
    WorkspaceAssignmentRequest,
    WorkspaceMemberUpdate,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.errors import (
    InvalidEmailError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberAlreadyExistsError,
    OrganizationMemberNotFoundError,
    OrganizationNotFoundError,
    WorkspaceNotFoundError,
)

# A shape check, not an authority on deliverability: one @, something either
# side, a dot in the domain, and no whitespace. See InvalidEmailError.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validated_email(email: str) -> str:
    """Normalize an address to lower case, refusing one that cannot be a handle."""
    candidate = email.strip().lower()
    if not _EMAIL_PATTERN.match(candidate):
        raise InvalidEmailError(email)
    return candidate


class OrganizationService:
    """Business logic for the organization surface."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationRepository(db)
        self.members = OrganizationMemberRepository(db)
        self.users = UserRepository(db)

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
    ) -> OrganizationMembershipContextPublic:
        return OrganizationMembershipContextPublic(
            organization_member_id=membership.id,
            role=membership.role,
            status=membership.status,
            organization=OrganizationPublic.model_validate(organization),
            # The platform answers "does this org have a self-hosted gateway
            # attached". A standalone deployment reading this *is* that gateway,
            # so its own provider credentials are always available to it.
            byo_provider_keys_allowed=True,
        )

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
        return self._to_context(membership=membership, organization=organization)

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
            {"name": organization_name.strip() or "Organization"},
        )
        await self.db.commit()

        return self._to_context(membership=membership, organization=updated)

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
        return ActiveOrganizationMembersPublic(
            data=[self._to_member_public(membership, member_user) for membership, member_user in rows],
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

    async def _apply_workspace_assignments(
        self,
        *,
        user_id: uuid.UUID,
        assignments: list[WorkspaceAssignmentRequest],
    ) -> None:
        """Grant each assigned workspace, reviving a suspended membership.

        Deduplicated by workspace first, keeping the first role named for one, so
        a body that repeats an id costs one round trip rather than one per
        repetition. `_require_workspaces_in_organization` already resolves the
        same set rather than the list.

        An existing membership is updated rather than skipped, as the platform's
        own assignment path does: a suspended row that is left alone would leave
        the member listed in a workspace they were just granted while still
        being refused everything in it.
        """
        members = WorkspaceMemberRepository(self.db)
        seen: set[uuid.UUID] = set()
        for assignment in assignments:
            if assignment.workspace_id in seen:
                continue
            seen.add(assignment.workspace_id)
            existing = await members.get_by_workspace_and_user(assignment.workspace_id, user_id)
            if existing is not None:
                await members.update(
                    existing,
                    WorkspaceMemberUpdate(role=assignment.role, status="active"),
                )
                continue
            await members.create(
                workspace_id=assignment.workspace_id,
                user_id=user_id,
                role=assignment.role,
            )

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

        updated = await self.members.update_membership(target, update_data)
        target_user = await self.users.get(updated.user_id)
        if target_user is None:
            raise OrganizationMemberNotFoundError(organization_member_id)
        await self.db.commit()

        return self._to_member_public(updated, target_user)

    async def remove_active_organization_member_for_user(
        self,
        *,
        user: User,
        organization_member_id: uuid.UUID,
    ) -> None:
        """Remove a member by suspending their membership.

        Suspension rather than deletion, as on the platform: the row is what
        every historical attribution resolves through, so it outlives the access
        it granted.
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

        await self.members.update_membership(target, {"status": "suspended"})
        await self.db.commit()

    async def _validate_membership_update(
        self,
        *,
        actor_membership: OrganizationMember,
        target_membership: OrganizationMember,
        update_data: dict[str, str],
        organization_id: uuid.UUID,
    ) -> None:
        """Refuse the two membership changes that would break the organization.

        An admin cannot act on an owner (only an owner outranks an owner), and
        the last active owner cannot be demoted or deactivated, which would leave
        the organization with nobody able to manage or delete it.
        """
        new_role = update_data.get("role", target_membership.role)
        new_status = update_data.get("status", target_membership.status)

        if actor_membership.role != "owner" and target_membership.role == "owner":
            raise MembershipUpdateError("Only organization owners can modify owner memberships")

        target_is_active_owner = target_membership.role == "owner" and target_membership.status == "active"
        if target_is_active_owner and (new_role != "owner" or new_status != "active"):
            if await self.members.count_active_owners(organization_id) <= 1:
                raise MembershipUpdateError("An organization must keep at least one active owner")

    @staticmethod
    def _to_member_public(membership: OrganizationMember, user: User) -> ActiveOrganizationMemberPublic:
        return ActiveOrganizationMemberPublic(
            organization_member_id=membership.id,
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=membership.role,
            status=membership.status,
            created_at=membership.created_at,
            updated_at=membership.updated_at,
        )


__all__ = ["OrganizationService"]
