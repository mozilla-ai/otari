"""Shared workspace-visibility and workspace-management checks.

Extracted from :class:`WorkspaceService`, whose private methods now delegate
here, so :class:`WorkspaceBudgetDefaultService` enforces the same two rules
rather than carrying a second, driftable copy of them. A leaf module on
purpose: it reaches :class:`OrganizationService` (to resolve the caller's
active organization and organization-level role) but nothing here reaches
back, which is what lets ``workspace_budget_default_service`` sit between
``workspace_service`` and ``organization_service`` in the import graph without
closing a cycle.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import MANAGEMENT_ROLES, Organization, User, Workspace
from gateway.repositories.tenancy import WorkspaceMemberRepository, WorkspaceRepository
from gateway.services.tenancy.errors import NotAuthorizedError, WorkspaceNotFoundError
from gateway.services.tenancy.organization_service import OrganizationService


async def resolve_visible_workspace(
    db: AsyncSession,
    *,
    user: User,
    workspace_id: uuid.UUID,
    organizations: OrganizationService,
) -> Workspace:
    """Resolve a workspace the caller may see, in their active organization, or raise not-found.

    Visible to a superuser, an organization owner/admin (who see every
    workspace in it), or an active member of the workspace itself. Every "may
    not see" case answers 404 rather than 403: another organization's
    workspace, and a workspace in this organization the caller is not a member
    of, must be indistinguishable from one that does not exist.
    """
    organization = await organizations.get_active_organization_for_user(user)
    return await resolve_workspace_in_organization(
        db,
        user=user,
        workspace_id=workspace_id,
        organization=organization,
        organizations=organizations,
    )


async def resolve_workspace_in_organization(
    db: AsyncSession,
    *,
    user: User,
    workspace_id: uuid.UUID,
    organization: Organization,
    organizations: OrganizationService,
) -> Workspace:
    """The body of :func:`resolve_visible_workspace`, for an already-resolved organization.

    Split out because a caller that already paid for the organization lookup
    (``WorkspaceService.delete_workspace``, which needs it for its own checks
    too) should not pay for it twice.
    """
    workspace = await WorkspaceRepository(db).get(workspace_id)
    if workspace is None or workspace.organization_id != organization.id:
        raise WorkspaceNotFoundError(workspace_id)

    if user.is_superuser:
        return workspace

    organization_membership = await organizations.members.get_active_by_organization_and_user(
        organization.id,
        user.id,
    )
    if organization_membership is not None and organization_membership.role in MANAGEMENT_ROLES:
        return workspace

    # Active-only: a suspended membership grants nothing.
    if await WorkspaceMemberRepository(db).get_active_by_workspace_and_user(workspace.id, user.id) is None:
        raise WorkspaceNotFoundError(workspace_id)
    return workspace


async def require_workspace_management_access(
    db: AsyncSession,
    *,
    user: User,
    workspace: Workspace,
    organizations: OrganizationService,
) -> None:
    """Allow a superuser, an organization owner/admin, or an owner/admin of this workspace.

    The superuser arm is what makes this agree with the two checks either side
    of it: ``resolve_visible_workspace`` grants a superuser read on every
    workspace, and organization-level management access grants them
    organization management, so without it a superuser could delete a
    workspace but not rename it.
    """
    if user.is_superuser:
        return

    organization_membership = await organizations.members.get_active_by_organization_and_user(
        workspace.organization_id,
        user.id,
    )
    if organization_membership is not None and organization_membership.role in MANAGEMENT_ROLES:
        return

    membership = await WorkspaceMemberRepository(db).get_by_workspace_and_user(workspace.id, user.id)
    if membership is not None and membership.status == "active" and membership.role in MANAGEMENT_ROLES:
        return

    raise NotAuthorizedError


__all__ = [
    "require_workspace_management_access",
    "resolve_visible_workspace",
    "resolve_workspace_in_organization",
]
