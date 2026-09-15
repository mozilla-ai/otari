"""Shared workspace-visibility and workspace-management checks.

Extracted from :class:`WorkspaceService`, whose private methods now delegate
here, so :class:`WorkspaceBudgetDefaultService` enforces the same two rules
rather than carrying a second, driftable copy of them. A leaf module on
purpose: it reaches :class:`OrganizationService` (to resolve the caller's
active organization and organization-level role) but nothing here reaches
back, which is what lets ``workspace_budget_default_service`` sit between
``workspace_service`` and ``organization_service`` in the import graph without
closing a cycle.

Every check below reads only active organization and workspace membership.
``User.is_superuser`` is deployment-operator status, not an organization or
workspace role, and is deliberately never consulted here: an operator's
tenant-scoped access is decided by their own membership exactly as anyone
else's is. Deployment-wide surfaces are gated separately, by
``require_deployment_operator`` at the router level. See mozilla-ai/otari#1011.
"""

import uuid
from dataclasses import dataclass

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

    Visible to an organization owner/admin (who see every workspace in it), or
    an active member of the workspace itself. Every "may not see" case answers
    404 rather than 403: another organization's workspace, and a workspace in
    this organization the caller is not a member of, must be indistinguishable
    from one that does not exist.
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


async def has_workspace_management_access(
    db: AsyncSession,
    *,
    user: User,
    workspace: Workspace,
    organizations: OrganizationService,
) -> bool:
    """Whether the caller is an organization owner/admin, or an owner/admin of this workspace.

    The predicate form exists for the one caller that has to *report* the answer
    rather than act on it: the first-request setup guide tells the dashboard
    whether to offer itself, so a member who may see the workspace without
    managing it is told "not for you" instead of being refused. Everything else
    wants :func:`require_workspace_management_access`.
    """
    organization_membership = await organizations.members.get_active_by_organization_and_user(
        workspace.organization_id,
        user.id,
    )
    if organization_membership is not None and organization_membership.role in MANAGEMENT_ROLES:
        return True

    membership = await WorkspaceMemberRepository(db).get_active_by_workspace_and_user(workspace.id, user.id)
    return membership is not None and membership.role in MANAGEMENT_ROLES


async def require_workspace_management_access(
    db: AsyncSession,
    *,
    user: User,
    workspace: Workspace,
    organizations: OrganizationService,
) -> None:
    """Allow an organization owner/admin, or an owner/admin of this workspace."""
    if not await has_workspace_management_access(db, user=user, workspace=workspace, organizations=organizations):
        raise NotAuthorizedError


@dataclass(frozen=True)
class VisibleWorkspaceScope:
    """How much of one organization the caller may be shown.

    The set form of :func:`resolve_visible_workspace`, and deliberately the same
    rule: an owner or an admin sees every workspace in the organization, and
    everyone else sees the ones they actively belong to. Superuser status is
    not part of that rule (see the module docstring). It is meant to be the one
    place this breadth decision is stated, so a tenant-scoped read cannot
    invent a fifth answer to "how much of this organization is yours"; the
    workspace resolver and the organization's usage and routing reads go
    through it. ``WorkspaceService.list_workspaces`` still re-derives the same
    boolean inline rather than calling this, for pagination reasons noted on
    that method; keep the two in agreement until that is unified.

    ``workspace_ids`` is ``None`` for the whole organization rather than a list of
    every workspace in it, so the caller can express that as a predicate over
    ``organization_id`` and not an ``IN`` list that grows with the tenant. An
    *empty* list is a real answer and not an error: a member who belongs to no
    workspace yet may see nothing, which is an empty page rather than a refusal.

    Carries no SQL. Turning this into a WHERE clause is the reading route's job,
    because the tables a scope applies to are not this module's business.
    """

    organization: Organization
    workspace_ids: list[uuid.UUID] | None

    @property
    def sees_every_workspace(self) -> bool:
        return self.workspace_ids is None


async def resolve_visible_workspace_scope(
    db: AsyncSession,
    *,
    user: User,
    organizations: OrganizationService,
) -> VisibleWorkspaceScope:
    """Resolve how much of their active organization this caller may be shown.

    The organization is the caller's own ``active_organization_id``, put through
    ``get_active_organization_for_user`` so a pointer with no live membership
    behind it raises rather than resolving. It is never taken from the request:
    a caller moves between organizations with ``POST /v1/organizations/me/switch``,
    which refuses an organization they hold no active membership in, so there is
    no parameter here for a cross-tenant read to travel on.
    """
    organization = await organizations.get_active_organization_for_user(user)
    membership = await organizations.members.get_active_by_organization_and_user(organization.id, user.id)
    # Not None: ``get_active_organization_for_user`` refuses without one.
    if membership is not None and membership.role in MANAGEMENT_ROLES:
        return VisibleWorkspaceScope(organization=organization, workspace_ids=None)

    workspace_ids = await WorkspaceMemberRepository(db).get_workspace_ids_for_user(
        user_id=user.id,
        organization_id=organization.id,
    )
    return VisibleWorkspaceScope(organization=organization, workspace_ids=workspace_ids)


__all__ = [
    "VisibleWorkspaceScope",
    "has_workspace_management_access",
    "require_workspace_management_access",
    "resolve_visible_workspace",
    "resolve_visible_workspace_scope",
    "resolve_workspace_in_organization",
]
