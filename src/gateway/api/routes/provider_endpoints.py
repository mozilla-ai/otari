"""Provider endpoints a workspace or one of its users owns.

An app built on otari lets its users bring their own model server. The app
registers each one here, then its users reach it as ``<name>:<model>`` on the
chat, responses and messages routes, with their own key and without using
anybody's budget. See ``services/providers`` for the rules an endpoint is held to.

Deployment operators only, like ``/api/v1/aliases``: an endpoint names its owner
explicitly, so writing one is acting across the deployment. Every route
answers 403 until ``provider_endpoints_enabled`` is on. No route here catches a
domain error, because each error carries its own status.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import ProviderEndpointServiceDep, require_deployment_operator
from gateway.schemas.providers import (
    ProviderEndpointCreateRequest,
    ProviderEndpointPublic,
    ProviderEndpointsPublic,
    ProviderEndpointUpdateRequest,
)

router = APIRouter(
    prefix="/provider-endpoints",
    tags=["provider-endpoints"],
    dependencies=[Depends(require_deployment_operator)],
)


@router.get("")
async def list_provider_endpoints(
    service: ProviderEndpointServiceDep,
    workspace_id: Annotated[uuid.UUID | None, Query(description="Only endpoints this workspace owns.")] = None,
    user_id: Annotated[str | None, Query(description="Only endpoints this user owns.")] = None,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> ProviderEndpointsPublic:
    """List owned provider endpoints, narrowed by owner when given. Keys are never returned."""
    return await service.list_endpoints(workspace_id=workspace_id, user_id=user_id, skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_provider_endpoint(
    service: ProviderEndpointServiceDep,
    body: ProviderEndpointCreateRequest,
) -> ProviderEndpointPublic:
    """Register an endpoint for a workspace, or for one user in it.

    It is reachable as ``<name>:<model>`` by its owner as soon as this returns on
    this worker, and on the others within 30 seconds.
    """
    return await service.create_endpoint(body)


@router.get("/{endpoint_id}")
async def get_provider_endpoint(
    service: ProviderEndpointServiceDep,
    endpoint_id: uuid.UUID,
) -> ProviderEndpointPublic:
    """Read one endpoint. The key is never returned, only its last four characters."""
    return await service.get_endpoint(endpoint_id)


@router.patch("/{endpoint_id}")
async def update_provider_endpoint(
    service: ProviderEndpointServiceDep,
    endpoint_id: uuid.UUID,
    body: ProviderEndpointUpdateRequest,
) -> ProviderEndpointPublic:
    """Change an endpoint's name, provider, base URL, key or default fields. The owner cannot change."""
    return await service.update_endpoint(endpoint_id, body)


@router.delete("/{endpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider_endpoint(
    service: ProviderEndpointServiceDep,
    endpoint_id: uuid.UUID,
) -> None:
    """Delete an endpoint. Its name stops resolving at once on this worker, within 30 seconds elsewhere."""
    await service.delete_endpoint(endpoint_id)
