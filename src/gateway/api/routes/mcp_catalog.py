"""The MCP server catalog: remote servers a workspace can add without hunting for a URL.

Listing a server grants nothing. A workspace's stored servers stay its allow
list, and a catalog server becomes one through the same create a pasted URL does.

Every route admits any signed-in caller. Asking for a sync is the one write, and
the service decides who may: an operator, or an admin of the workspace named.
Listed in ``CORE_FEATURES`` as ``FEATURE`` below.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status

from gateway.api.deps import (
    CurrentIdentity,
    McpCatalogServiceDep,
    OperatesDeployment,
    build_mcp_catalog_service,
    verify_master_key,
)
from gateway.api.routes._logo_response import LOGO_RESPONSES, logo_response
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.schemas.mcp_catalog import (
    McpCatalogServersPublic,
    McpCatalogSyncRequest,
    McpCatalogSyncStatus,
)
from gateway.services.mcp_catalog import McpCatalogService

_WorkspaceId = Annotated[
    uuid.UUID | None,
    Query(description="A workspace the caller can see. The servers it already has are left out."),
]

router = APIRouter(prefix="/mcp-catalog", tags=["mcp-catalog"], dependencies=[Depends(verify_master_key)])


@router.get("/servers")
async def search_mcp_catalog(
    service: McpCatalogServiceDep,
    current_identity: CurrentIdentity,
    q: Annotated[str, Query(max_length=200, description="Text to find in a server's title, name or description")] = "",
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 25,
    workspace_id: _WorkspaceId = None,
) -> McpCatalogServersPublic:
    """Search the catalog of remote MCP servers, or page through all of it with no `q`.

    A title that starts with `q` comes first, then a title that contains it, then a
    match in the name or description alone; within each, a server whose
    `publisher_kind` is `vendor` comes first. The catalog is
    empty until a sync has run, and stays empty on a deployment whose
    `mcp_catalog_sync` is `off`. Listing a server here permits nothing: a
    workspace reaches a server only once it is added to that workspace.
    """
    return await service.search(query=q, skip=skip, limit=limit, user=current_identity, workspace_id=workspace_id)


@router.get("/popular")
async def list_popular_mcp_servers(
    service: McpCatalogServiceDep, current_identity: CurrentIdentity, workspace_id: _WorkspaceId = None
) -> McpCatalogServersPublic:
    """Well-known remote MCP servers, from a list bundled with Otari, that the catalog holds now.

    Each is matched to the catalog by its URL, so the list is empty until a sync has run.
    """
    return await service.popular(user=current_identity, workspace_id=workspace_id)


@router.get("/logos/{name:path}", include_in_schema=False, responses=LOGO_RESPONSES)
async def get_mcp_catalog_logo(request: Request, name: str, service: McpCatalogServiceDep) -> Response:
    """A catalog server's logo, served from this gateway; `logo_url` on a catalog entry points here.

    The first request for a logo nobody has looked for yet makes the gateway look for it, within a
    few seconds and a per-process limit on looks. A name the catalog does not list is a 404 and
    reaches nothing outside the gateway.
    """
    return logo_response(request, await service.logo(name))


@router.get("/sync")
async def get_mcp_catalog_sync(service: McpCatalogServiceDep) -> McpCatalogSyncStatus:
    """Whether the catalog syncs from the MCP Registry, and what the sync in progress or the last one did."""
    return await service.sync_status()


@router.post(
    "/sync",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_403_FORBIDDEN: {
            "description": "The caller does not operate the deployment and may not add MCP servers to the workspace."
        },
        status.HTTP_409_CONFLICT: {"description": "The sync is switched off on this deployment; nothing was started."},
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "A sync ended too recently for a workspace admin to ask for another; see next_request_at."
        },
    },
)
async def request_mcp_catalog_sync(
    service: McpCatalogServiceDep,
    current_identity: CurrentIdentity,
    operates_deployment: OperatesDeployment,
    body: McpCatalogSyncRequest | None = None,
) -> McpCatalogSyncStatus:
    """Start a sync within a minute. Refused with a 409 while `mcp_catalog_sync` is `off`.

    An operator may always ask. Anyone else names a `workspace_id` they may add MCP
    servers to, and waits out `next_request_at` after the last sync: an hour after
    a complete one, five minutes after any other. A sync already running or asked
    for is left to finish, and this request returns its status.
    """
    return await service.request_sync(
        user=current_identity,
        workspace_id=body.workspace_id if body is not None else None,
        operates_deployment=operates_deployment,
    )


async def _poll(config: GatewayConfig) -> None:
    await McpCatalogService.poll(lambda uow: build_mcp_catalog_service(uow, config), config)


FEATURE = CoreFeature(
    name="mcp_catalog",
    surface=Surface("mcp_catalog"),
    enabled=lambda config: config.mcp_catalog_enabled,
    routers=lambda config: (router,),
    worker=_poll,
)
