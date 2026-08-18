"""Workspaces and their members.

Rehomed from the platform's ``WorkspaceService``, converted to async, with the
same two-level authorization model:

- creating and deleting a workspace is an organization owner/admin action,
- managing a workspace's members and settings is open to an organization
  owner/admin *or* to an owner/admin of that workspace,
- reading is open to any member of the workspace, plus organization
  owners/admins, who see every workspace in the organization.

Dropped on arrival, each with its own slice to arrive in: mixpanel tracking,
per-member budget-policy materialization, the Playground token anchor, and
organization guardrail provisioning. Their absence is why creation is a plain
two-row insert here.
"""

import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import (
    MANAGEMENT_ROLES,
    WORKSPACE_MEMBER_ROLES,
    Organization,
    User,
    Workspace,
    WorkspaceCreate,
    WorkspaceMemberPublic,
    WorkspaceMembersPublic,
    WorkspaceMemberUpdate,
    WorkspacePublic,
    WorkspacesPublic,
    WorkspaceUpdate,
)
from gateway.repositories.tenancy import WorkspaceMemberRepository, WorkspaceRepository
from gateway.services.tenancy.errors import (
    InvalidRoleError,
    LastWorkspaceError,
    NotAnOrganizationMemberError,
    NotAuthorizedError,
    WorkspaceAlreadyExistsError,
    WorkspaceMemberAlreadyExistsError,
    WorkspaceMemberNotFoundError,
    WorkspaceNameRequiredError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.organization_service import OrganizationService


class WorkspaceService:
    """Business logic for the workspace surface."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.workspaces = WorkspaceRepository(db)
        self.members = WorkspaceMemberRepository(db)
        self.organizations = OrganizationService(db)

    # ------------------------------------------------------------------
    # Scoping and authorization
    # ------------------------------------------------------------------

    async def _active_organization(self, user: User) -> Organization:
        return await self.organizations.get_active_organization_for_user(user)

    async def _workspace_in_active_organization(self, *, user: User, workspace_id: uuid.UUID) -> Workspace:
        """Resolve a workspace the caller may see, or raise not-found.

        Every "may not see" case answers 404 rather than 403 on purpose: another
        organization's workspace, and a workspace in this organization that the
        caller is not a member of, must be indistinguishable from one that does
        not exist.
        """
        organization = await self._active_organization(user)
        workspace = await self.workspaces.get(workspace_id)
        if workspace is None or workspace.organization_id != organization.id:
            raise WorkspaceNotFoundError(workspace_id)

        if user.is_superuser:
            return workspace

        organization_membership = await self.organizations.members.get_active_by_organization_and_user(
            organization.id,
            user.id,
        )
        if organization_membership is not None and organization_membership.role in MANAGEMENT_ROLES:
            return workspace

        if await self.members.get_by_workspace_and_user(workspace.id, user.id) is None:
            raise WorkspaceNotFoundError(workspace_id)
        return workspace

    @staticmethod
    def _validated_role(role: str) -> str:
        """Reject an unknown role before it is stored.

        The role arrives as a query parameter and lands on a table model, and
        SQLModel skips validation when constructing a table instance, so nothing
        else between the request and the row checks it.
        """
        if role not in WORKSPACE_MEMBER_ROLES:
            raise InvalidRoleError(role, WORKSPACE_MEMBER_ROLES)
        return role

    @staticmethod
    def _validated_name(name: str | None) -> str:
        """Reject a null or blank workspace name for the same reason as a role.

        ``name`` also lands on a table model, and the column is NOT NULL with no
        minimum length, so an explicit ``null`` reached the database as an
        integrity error and an empty string stored a nameless workspace.
        """
        trimmed = (name or "").strip()
        if not trimmed:
            raise WorkspaceNameRequiredError
        return trimmed

    async def _require_workspace_management_access(self, *, user: User, workspace: Workspace) -> None:
        """Allow an organization owner/admin, or an owner/admin of this workspace."""
        organization_membership = await self.organizations.members.get_active_by_organization_and_user(
            workspace.organization_id,
            user.id,
        )
        if organization_membership is not None and organization_membership.role in MANAGEMENT_ROLES:
            return

        membership = await self.members.get_by_workspace_and_user(workspace.id, user.id)
        if membership is not None and membership.status == "active" and membership.role in MANAGEMENT_ROLES:
            return

        raise NotAuthorizedError

    # ------------------------------------------------------------------
    # Workspace lifecycle
    # ------------------------------------------------------------------

    async def create_workspace(self, *, user: User, workspace_create: WorkspaceCreate) -> WorkspacePublic:
        """Create a workspace in the caller's organization, with them as its owner."""
        organization = await self._active_organization(user)
        await self.organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )

        name = self._validated_name(workspace_create.name)
        if await self.workspaces.get_by_organization_and_name(organization.id, name) is not None:
            raise WorkspaceAlreadyExistsError(name)

        # The pre-check above races the insert, so the unique constraint is what
        # actually decides. Without this the loser of that race answers 500
        # instead of the 409 the pre-check would have given it.
        try:
            workspace = await self.workspaces.create_workspace(
                name=name,
                description=workspace_create.description,
                organization_id=organization.id,
                created_by_user_id=user.id,
            )
            await self.members.create(workspace_id=workspace.id, user_id=user.id, role="owner")
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise WorkspaceAlreadyExistsError(name) from None

        return WorkspacePublic.model_validate(workspace)

    async def get_workspace(self, *, user: User, workspace_id: uuid.UUID) -> WorkspacePublic:
        """Return one workspace the caller may see."""
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)
        return WorkspacePublic.model_validate(workspace)

    async def list_workspaces(self, *, user: User, skip: int = 0, limit: int = 100) -> WorkspacesPublic:
        """List the workspaces the caller may see in their organization.

        Organization owners, admins and superusers see all of them; everyone else
        sees the ones they are a member of.
        """
        organization = await self._active_organization(user)

        membership = await self.organizations.members.get_active_by_organization_and_user(organization.id, user.id)
        sees_every_workspace = user.is_superuser or (membership is not None and membership.role in MANAGEMENT_ROLES)

        if sees_every_workspace:
            workspaces, count = await self.workspaces.get_by_organization(organization.id, skip=skip, limit=limit)
        else:
            member_rows, count = await self.members.get_workspaces_for_user(
                user_id=user.id,
                organization_id=organization.id,
                skip=skip,
                limit=limit,
            )
            workspaces = await self.workspaces.get_by_ids([row.workspace_id for row in member_rows])

        return WorkspacesPublic(
            data=[WorkspacePublic.model_validate(workspace) for workspace in workspaces],
            count=count,
        )

    async def update_workspace(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        workspace_update: WorkspaceUpdate,
    ) -> WorkspacePublic:
        """Rename a workspace or change its description."""
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)
        await self._require_workspace_management_access(user=user, workspace=workspace)

        update_data = workspace_update.model_dump(exclude_unset=True)
        if "name" in update_data:
            new_name = self._validated_name(update_data["name"])
            update_data["name"] = new_name
            if new_name != workspace.name:
                clash = await self.workspaces.get_by_organization_and_name(workspace.organization_id, new_name)
                if clash is not None:
                    raise WorkspaceAlreadyExistsError(new_name)

        try:
            updated = await self.workspaces.update_workspace(workspace, update_data)
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise WorkspaceAlreadyExistsError(str(update_data.get("name", workspace.name))) from None
        return WorkspacePublic.model_validate(updated)

    async def delete_workspace(self, *, user: User, workspace_id: uuid.UUID) -> None:
        """Delete a workspace. Members ride the database cascade.

        The last one cannot go: every creation path provisions a workspace
        because an organization without one has no usable surface, and nothing
        would provision a replacement for an organization that already exists.
        """
        organization = await self._active_organization(user)
        await self.organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)

        _, remaining = await self.workspaces.get_by_organization(organization.id, limit=1)
        if remaining <= 1:
            raise LastWorkspaceError

        await self.workspaces.delete_workspace(workspace)
        await self.db.commit()

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    async def list_members(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> WorkspaceMembersPublic:
        """List a workspace's members. Any member of the workspace may read it."""
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)
        members, count = await self.members.get_by_workspace(workspace.id, skip=skip, limit=limit)
        return WorkspaceMembersPublic(
            data=[WorkspaceMemberPublic.model_validate(member) for member in members],
            count=count,
        )

    async def add_member(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str = "member",
    ) -> WorkspaceMemberPublic:
        """Add an existing organization member to a workspace.

        Membership of the organization is a precondition, not something this
        grants: a workspace cannot be a back door into the tenant.
        """
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)
        await self._require_workspace_management_access(user=user, workspace=workspace)
        role = self._validated_role(role)

        if not await self.organizations.user_has_active_membership(
            organization_id=workspace.organization_id,
            user_id=user_id,
        ):
            raise NotAnOrganizationMemberError(user_id)

        if await self.members.get_by_workspace_and_user(workspace.id, user_id) is not None:
            raise WorkspaceMemberAlreadyExistsError(user_id)

        # As in create_workspace: the pre-check races the insert, and the unique
        # constraint is what actually decides.
        try:
            member = await self.members.create(workspace_id=workspace.id, user_id=user_id, role=role)
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise WorkspaceMemberAlreadyExistsError(user_id) from None
        return WorkspaceMemberPublic.model_validate(member)

    async def update_member_role(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str,
    ) -> WorkspaceMemberPublic:
        """Change a workspace member's role."""
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)
        await self._require_workspace_management_access(user=user, workspace=workspace)
        role = self._validated_role(role)

        member = await self.members.get_by_workspace_and_user(workspace.id, user_id)
        if member is None:
            raise WorkspaceMemberNotFoundError(workspace_id, user_id)

        updated = await self.members.update(member, WorkspaceMemberUpdate(role=role))
        await self.db.commit()
        return WorkspaceMemberPublic.model_validate(updated)

    async def remove_member(self, *, user: User, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Remove a member from a workspace. Idempotent."""
        workspace = await self._workspace_in_active_organization(user=user, workspace_id=workspace_id)
        await self._require_workspace_management_access(user=user, workspace=workspace)

        member = await self.members.get_by_workspace_and_user(workspace.id, user_id)
        if member is None:
            return

        await self.members.delete(member)
        await self.db.commit()


__all__ = ["WorkspaceService"]
