"""Per-workspace web-search configuration (standalone mode only).

The deployment-wide search backend (its URL, its engines, its purpose hint)
stays on ``/v1/tool-settings``, and the tools ``POST /v1/search`` dispatches to
stay on ``/v1/search-tools``; this surface says which workspaces on that
deployment may search and how far their searches may reach. Thin composition
over `gateway.services.tenancy.workspace_web_search_service`, following
`routes/workspace_code_execution_policy.py`'s shape (master key on the router,
plus the caller's tenancy identity for the per-workspace role checks).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.services.tenancy.workspace_web_search_service import (
    WorkspaceWebSearchConfigPublic,
    WorkspaceWebSearchConfigUpdate,
    WorkspaceWebSearchService,
)

# Auth is declared on the router, matching the sibling per-workspace routers:
# every handler here needs the master key, and a future one that forgot the
# decorator would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/web-search",
    tags=["workspace-web-search"],
    dependencies=[Depends(verify_master_key)],
)


def get_workspace_web_search_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceWebSearchService:
    """Build the service on the request's session.

    The backend-presence check is the same one ``prepare_gateway_tools`` makes
    when it decides whether a request may declare ``otari_web_search``, so the
    page and the request path agree about whether this deployment can search at
    all.
    """
    return WorkspaceWebSearchService(
        db,
        web_search_configured=bool(config.web_search_url or otari_env("WEB_SEARCH_URL")),
    )


WorkspaceWebSearchServiceDep = Annotated[WorkspaceWebSearchService, Depends(get_workspace_web_search_service)]


@router.get("")
async def get_workspace_web_search_config(
    service: WorkspaceWebSearchServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceWebSearchConfigPublic:
    """Read a workspace's web-search configuration.

    Takes the same role as setting it (an organization owner/admin, or an
    owner/admin of this workspace), because the row describes the workspace's
    posture rather than one member's allowance. A workspace with no row answers
    with the unconfigured shape (``configured: false``), which is the
    deployment's own behavior described in the same shape rather than a 404.
    """
    return await service.get_config(user=current_identity, workspace_id=workspace_id)


@router.put("")
async def set_workspace_web_search_config(
    service: WorkspaceWebSearchServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceWebSearchConfigUpdate,
) -> WorkspaceWebSearchConfigPublic:
    """Set a workspace's web-search configuration, replacing any existing one.

    An organization owner/admin, or an owner/admin of this workspace, may write
    it. The configuration can only narrow what the deployment permits: turning
    web search off for the workspace, lowering the result ceiling, and adding to
    the domains a search may not reach. It never turns on a backend the
    deployment has not configured, and it carries no credential.
    """
    return await service.set_config(user=current_identity, workspace_id=workspace_id, request=body)


@router.delete("")
async def clear_workspace_web_search_config(
    service: WorkspaceWebSearchServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceWebSearchConfigPublic:
    """Drop a workspace's configuration, returning it to the deployment's behavior.

    Idempotent: a workspace that has no configuration is already in the state
    this asks for, so it answers with the unconfigured shape rather than a 404.
    """
    return await service.clear_config(user=current_identity, workspace_id=workspace_id)
