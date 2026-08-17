"""Organizations: active-organization resolution, CRUD, and membership.

Rehomed from the platform's ``OrganizationService`` plus the membership half of
``OrganizationMembershipService``, converted to async. The authorization rules,
the membership constraints, and the response shapes are the platform's; what is
gone is the depth that has no home in the OSS base yet: mixpanel tracking,
managed provider-key and default-gateway provisioning, email-domain auto-join,
invitations, teams, and the org's budget and pricing surfaces. Those arrive with
their own slices (see the PR description) and this service is where they attach.

One rule runs through every method: a caller only ever acts inside the
organization their identity is currently pointed at. Nothing takes an
organization id from the request except the switch endpoint, which checks
membership before it moves the pointer, so a request cannot name another
tenant's organization at all.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import (
    MANAGEMENT_ROLES,
    ActiveOrganizationMemberPublic,
    ActiveOrganizationMembersPublic,
    ActiveOrganizationMemberUpdateRequest,
    Organization,
    OrganizationMember,
    OrganizationMembershipContextPublic,
    OrganizationMembershipContextsPublic,
    OrganizationPublic,
    User,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.errors import (
    LastOrganizationError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberNotFoundError,
    OrganizationNotFoundError,
)
from gateway.services.tenancy.provisioning_service import DEFAULT_WORKSPACE_NAME
from gateway.services.tenancy.slug import slugify


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
        *deleted* organization is not reachable while foreign keys are enforced,
        which is why `delete_active_organization` repoints first; the check
        survives anyway, for a database whose keys are not.

        Unlike the platform this never provisions a personal organization as a
        fallback: in the OSS base first boot owns that (see
        `provisioning_service`), and every later identity is created inside an
        organization that already exists.
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

    async def list_membership_contexts_for_user(self, user: User) -> OrganizationMembershipContextsPublic:
        """List every organization the caller is an active member of, by name."""
        await self._resolve_active_organization(user)

        memberships = await self.members.get_by_user(user.id, active_only=True)
        organizations = await self.organizations.get_by_ids([m.organization_id for m in memberships])
        by_id = {organization.id: organization for organization in organizations}

        contexts = [
            self._to_context(membership=membership, organization=organization)
            for membership in memberships
            if (organization := by_id.get(membership.organization_id)) is not None
        ]
        contexts.sort(
            key=lambda context: (
                context.organization.name.casefold(),
                context.organization.name,
                str(context.organization.id),
            )
        )
        return OrganizationMembershipContextsPublic(data=contexts, count=len(contexts))

    # ------------------------------------------------------------------
    # Organization lifecycle
    # ------------------------------------------------------------------

    async def switch_active_organization_for_user(
        self,
        *,
        user: User,
        organization_id: uuid.UUID,
    ) -> OrganizationMembershipContextPublic:
        """Point the caller at another of their organizations.

        The membership check is the tenant boundary for this endpoint, the only
        one that accepts an organization id from the request: an id the caller is
        not an active member of is refused rather than switched to.
        """
        organization = await self.organizations.get(organization_id)
        if organization is None:
            raise OrganizationNotFoundError(organization_id)
        membership = await self._require_active_membership(user, organization)

        await self.users.set_active_organization(user, organization.id)
        await self.db.commit()

        return self._to_context(membership=membership, organization=organization)

    async def create_organization_for_user(
        self,
        *,
        user: User,
        organization_name: str,
    ) -> OrganizationMembershipContextPublic:
        """Create an organization owned by the caller, and switch them into it."""
        name = organization_name.strip() or "Organization"
        organization = await self.organizations.create_organization(
            name=name,
            slug=await self._unique_slug_for(name),
            created_by_user_id=user.id,
        )
        membership = await self.members.create_membership(
            organization_id=organization.id,
            user_id=user.id,
            role="owner",
        )
        await self._provision_default_workspace(organization=organization, user=user)
        await self.users.set_active_organization(user, organization.id)
        await self.db.commit()

        return self._to_context(membership=membership, organization=organization)

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

    async def delete_active_organization(self, *, current_user: User) -> None:
        """Delete the caller's organization. Owners only.

        Members and workspaces ride the database cascade, but ``user`` does not:
        ``active_organization_id`` is NOT NULL with no delete rule, so every
        identity pointed here is moved to another of its organizations first. An
        identity with nowhere to go blocks the delete rather than being orphaned;
        the OSS base has no path that would re-home it (the platform's fallback
        was to mint a personal organization from the user's email, and a local
        identity has none).
        """
        organization = await self.get_active_organization_for_user(current_user)
        membership = await self._require_active_membership(current_user, organization)
        if membership.role != "owner" and not current_user.is_superuser:
            raise NotAuthorizedError("Only an organization owner can delete it")

        for affected in await self.users.get_by_active_organization(organization.id):
            destination = next(
                (
                    other.organization_id
                    for other in await self.members.get_by_user(affected.id, active_only=True)
                    if other.organization_id != organization.id
                ),
                None,
            )
            if destination is None:
                raise LastOrganizationError(affected.id)
            await self.users.set_active_organization(affected, destination)

        await self.organizations.delete(organization)
        await self.db.commit()

    async def _unique_slug_for(self, name: str) -> str:
        """Return ``name``'s slug, suffixed until it is free."""
        base = slugify(name)
        candidate = base
        suffix = 2
        while await self.organizations.get_by_slug(candidate) is not None:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    async def _provision_default_workspace(self, *, organization: Organization, user: User) -> None:
        """Give a new organization one workspace, owned by its creator.

        An organization with no workspace has no usable surface, so every
        creation path provisions one, exactly as the platform's
        ``ensure_default_workspace`` does.
        """
        workspace = await WorkspaceRepository(self.db).create_workspace(
            name=DEFAULT_WORKSPACE_NAME,
            organization_id=organization.id,
            created_by_user_id=user.id,
        )
        await WorkspaceMemberRepository(self.db).create(
            workspace_id=workspace.id,
            user_id=user.id,
            role="owner",
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
        return ActiveOrganizationMembersPublic(
            data=[self._to_member_public(membership, member_user) for membership, member_user in rows],
            count=count,
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

        update_data = request.model_dump(exclude_unset=True)
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
