"""Members' requests for a server of the MCP catalog, and the managers' answer.

A member who cannot add MCP servers asks the workspace's owners and admins for one the catalog
lists. A manager answers by adding the server, through the ordinary create, which approves every
pending request for it, or by declining here. Thin composition over
`gateway.services.tools.McpServerRequestService`; see `docs/mcp-catalog.md`.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import CurrentIdentity, McpServerRequestServiceDep, verify_master_key
from gateway.schemas.tools import (
    McpServerRequestCreate,
    McpServerRequestDecline,
    McpServerRequestGroupsPublic,
    McpServerRequestPublic,
    McpServerRequestsDeclined,
    McpServerRequestsPublic,
)
from gateway.services.tools import McpServerRequestService

# Auth is declared on the router, as `routes/workspace_mcp_servers.py` does.
router = APIRouter(
    prefix="/workspaces/{workspace_id}/mcp-server-requests",
    tags=["mcp-servers"],
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def list_mcp_server_requests(
    service: McpServerRequestServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of catalog servers to skip")] = 0,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=McpServerRequestService.MAX_PENDING_SERVERS_PAGE,
            description="Maximum number of catalog servers to return",
        ),
    ] = McpServerRequestService.MAX_PENDING_SERVERS_PAGE,
) -> McpServerRequestGroupsPublic:
    """The workspace's pending requests, one entry per catalog server, the longest-waiting first.

    Organization owners/admins or this workspace's owners/admins. Each entry names who asked and
    carries their notes, and the catalog's entry for the server as it is now. A page holds `limit`
    servers after the first `skip`, and at most 500 requests; `count` says how many wait in all.
    """
    return await service.pending(user=current_identity, workspace_id=workspace_id, skip=skip, limit=limit)


@router.get("/mine")
async def list_my_mcp_server_requests(
    service: McpServerRequestServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=McpServerRequestService.MAX_OWN_PAGE, description="Maximum number of records to return")
    ] = 50,
) -> McpServerRequestsPublic:
    """The caller's own requests in this workspace: the pending ones, and the answers they have not acknowledged.

    Any member who can see the workspace. An answer, approved or declined, is listed until the
    caller acknowledges it. A declined request also leaves the list when the caller asks for the
    server again or the server is added, and an approved one when its server is deleted.
    """
    return await service.mine(user=current_identity, workspace_id=workspace_id, skip=skip, limit=limit)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "The catalog lists no server by that name."},
        status.HTTP_409_CONFLICT: {
            "description": "The workspace already has the server, or the caller's request for it is already waiting."
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "The caller has 10 requests waiting in this workspace, or made 10 there in the last day."
        },
    },
)
async def request_mcp_server(
    service: McpServerRequestServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: McpServerRequestCreate,
) -> McpServerRequestPublic:
    """Ask the workspace's owners and admins to add a server the MCP catalog lists.

    Any member who can see the workspace, a viewer included. The request keeps the title and URL
    the catalog gives the server now. Where this deployment can send mail, the workspace's owners
    and admins and its organization's are told, at most 20 of them; otherwise they find the
    request on the MCP servers page. A caller holds one pending request per server.
    """
    return await service.create(user=current_identity, workspace_id=workspace_id, request=body)


@router.post(
    "/decline",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Nobody is waiting for that server in this workspace."}},
)
async def decline_mcp_server_requests(
    service: McpServerRequestServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: McpServerRequestDecline,
) -> McpServerRequestsDeclined:
    """Decline every pending request for one catalog server.

    Organization owners/admins or this workspace's owners/admins. Each requester sees the
    `reason`, and is mailed it where this deployment can send mail. To approve instead, add the
    server to the workspace: storing a server at the requested URL approves every pending request
    for it.
    """
    return await service.decline(user=current_identity, workspace_id=workspace_id, request=body)


@router.delete(
    "/{request_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_409_CONFLICT: {"description": "The request was already answered or withdrawn."}},
)
async def withdraw_mcp_server_request(
    service: McpServerRequestServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    request_id: uuid.UUID,
) -> None:
    """Withdraw one of the caller's own pending requests. Someone else's request is a 404."""
    await service.withdraw(user=current_identity, workspace_id=workspace_id, request_id=request_id)


@router.post(
    "/{request_id}/acknowledge",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_409_CONFLICT: {"description": "The request is still waiting for an answer."}},
)
async def acknowledge_mcp_server_request(
    service: McpServerRequestServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    request_id: uuid.UUID,
) -> None:
    """Record that the caller has seen the answer to one of their own requests, so it is no longer listed for them.

    Repeating it changes nothing. Someone else's request is a 404.
    """
    await service.acknowledge(user=current_identity, workspace_id=workspace_id, request_id=request_id)
