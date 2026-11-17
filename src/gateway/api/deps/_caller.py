"""The organization a deployment-wide request acts in, and the tenant rows it may reach."""

import uuid
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps.identity import CurrentIdentity
from gateway.core.database import get_db
from gateway.repositories.tenancy import WorkspaceRepository, ids_in_organization
from gateway.services.tenancy import OrganizationService
from gateway.services.usage_and_telemetry import UsageScope


async def _caller_organization_id(
    db: Annotated[AsyncSession, Depends(get_db)],
    identity: CurrentIdentity,
) -> uuid.UUID:
    """The organization this request acts in.

    A key is minted, listed and revoked inside one organization, and so is a
    spend identity read, so every deployment-wide route that touches a tenant's
    rows resolves the caller's organization before it does. A dashboard session
    names the identity behind it and resolves that identity's active
    organization, which is what ``POST /api/v1/organizations/me/switch`` moves; a
    header master key names nobody, resolves the bootstrap operator, and
    therefore acts in the default organization. That is the same rule
    ``services/workspace_scope`` already documents for a deployment-wide write,
    so an operator running several organizations behind one gateway works in the
    one they are currently in rather than across all of them (otari#817).
    """
    return (await OrganizationService(db, membership_listener=None).get_active_organization_for_user(identity)).id


CallerOrganization = Annotated[uuid.UUID, Depends(_caller_organization_id)]


async def _caller_workspace_ids(
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: CallerOrganization,
) -> frozenset[uuid.UUID]:
    """Every workspace in the organization this request acts in.

    What a deployment-wide read of workspace-owned rows is confined to, so an
    operator sees their active organization's rows rather than every tenant's.
    """
    return frozenset(await WorkspaceRepository(db).get_ids_by_organization(organization_id))


CallerWorkspaces = Annotated[frozenset[uuid.UUID], Depends(_caller_workspace_ids)]


def _caller_workspace_query(organization_id: CallerOrganization) -> Select[uuid.UUID]:
    """The same workspaces as a subquery, so a read of a growing table filters in the database."""
    return ids_in_organization(organization_id)


CallerWorkspaceQuery = Annotated[Select[uuid.UUID], Depends(_caller_workspace_query)]


def _caller_usage_scope(workspaces: CallerWorkspaceQuery) -> UsageScope:
    """The usage rows an operator's usage reads cover: those of the caller's organization."""
    return UsageScope(workspaces=workspaces)


CallerUsageScope = Annotated[UsageScope, Depends(_caller_usage_scope)]
