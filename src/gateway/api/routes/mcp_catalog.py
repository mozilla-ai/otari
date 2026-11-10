"""The MCP server catalog: remote servers a workspace can add without hunting for a URL.

Listing a server grants nothing. A workspace's stored servers stay its allow
list, and a catalog server becomes one through the same create a pasted URL does.

The sync is the deployment's and an organization's shelf is its own, so each
has a router with its own gate rather than a route overriding one in place.
Listed in ``CORE_FEATURES`` as ``FEATURE`` below.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import (
    CurrentIdentity,
    McpCatalogServiceDep,
    build_mcp_catalog_service,
    require_deployment_operator,
    verify_master_key,
)
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.schemas.mcp_catalog import (
    McpCatalogFeaturedListPublic,
    McpCatalogFeaturedPublic,
    McpCatalogFeatureRequest,
    McpCatalogServersPublic,
    McpCatalogSyncStatus,
)
from gateway.services.mcp_catalog import run_mcp_catalog_sync_worker

router = APIRouter(prefix="/mcp-catalog", tags=["mcp-catalog"], dependencies=[Depends(verify_master_key)])

featured_router = APIRouter(
    prefix="/organizations/me/mcp-catalog/featured",
    tags=["mcp-catalog"],
    dependencies=[Depends(verify_master_key)],
)

sync_router = APIRouter(
    prefix="/mcp-catalog/sync",
    tags=["mcp-catalog"],
    dependencies=[Depends(require_deployment_operator)],
)


@router.get("/servers")
async def search_mcp_catalog(
    service: McpCatalogServiceDep,
    q: Annotated[str, Query(max_length=200, description="Text to find in a server's title, name or description")] = "",
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 25,
) -> McpCatalogServersPublic:
    """Search the catalog of remote MCP servers, or page through all of it with no `q`.

    A title that starts with `q` comes first, then a title that contains it, then a
    match in the name or description alone. The catalog is
    empty until a sync has run, and stays empty on a deployment whose
    `mcp_catalog_sync` is `off`. Listing a server here permits nothing: a
    workspace reaches a server only once it is added to that workspace.
    """
    return await service.search(query=q, skip=skip, limit=limit)


@featured_router.get("")
async def list_featured_mcp_servers(
    service: McpCatalogServiceDep, current_identity: CurrentIdentity
) -> McpCatalogFeaturedListPublic:
    """The servers the caller's organization recommends, by title. Any member of the organization.

    Each entry is the copy taken when it was featured, and `catalog_status`
    says whether the catalog still lists it at that URL.
    """
    return await service.list_featured(user=current_identity)


@featured_router.post(
    "",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "The catalog holds no server with this id."},
        status.HTTP_409_CONFLICT: {"description": "The organization already features 50 servers; remove one first."},
    },
)
async def feature_mcp_server(
    service: McpCatalogServiceDep, current_identity: CurrentIdentity, body: McpCatalogFeatureRequest
) -> McpCatalogFeaturedPublic:
    """Recommend a catalog server to the caller's organization. Organization owners and admins only.

    The entry is copied as the catalog lists it now, and a later sync neither
    changes nor removes the copy. Featuring a server that is already featured
    takes the copy again. An organization features at most 50 servers; one more
    is refused with a 409.
    """
    return await service.feature(user=current_identity, request=body)


@featured_router.delete(
    "/{featured_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "The organization features no entry with this id."}},
)
async def unfeature_mcp_server(
    service: McpCatalogServiceDep, current_identity: CurrentIdentity, featured_id: uuid.UUID
) -> None:
    """Stop recommending a server. Organization owners and admins only.

    Servers already added to a workspace from it are left as they are.
    """
    await service.unfeature(user=current_identity, featured_id=featured_id)


@sync_router.get("")
async def get_mcp_catalog_sync(service: McpCatalogServiceDep) -> McpCatalogSyncStatus:
    """Whether the catalog syncs from the MCP Registry, and what the sync in progress or the last one did."""
    return await service.sync_status()


@sync_router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_409_CONFLICT: {"description": "The sync is switched off on this deployment; nothing was started."}
    },
)
async def request_mcp_catalog_sync(service: McpCatalogServiceDep) -> McpCatalogSyncStatus:
    """Start a sync within a minute. Refused with a 409 while `mcp_catalog_sync` is `off`.

    A sync that is already running is left to finish, and this one follows it.
    """
    return await service.request_sync()


async def _poll(config: GatewayConfig) -> None:
    await run_mcp_catalog_sync_worker(lambda uow: build_mcp_catalog_service(uow, config), config)


FEATURE = CoreFeature(
    name="mcp_catalog",
    surface=Surface("mcp_catalog"),
    enabled=lambda config: config.mcp_catalog_enabled,
    routers=lambda config: (router, featured_router, sync_router),
    worker=_poll,
)
