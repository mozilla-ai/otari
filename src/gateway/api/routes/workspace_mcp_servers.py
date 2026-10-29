"""Workspace-scoped MCP server management.

Thin composition over
`gateway.services.tenancy.workspace_mcp_server_service`, following
`routes/workspace_member_budget_policies.py`'s shape: master key on the
router, plus the caller's tenancy identity for the per-workspace role check
the service runs. A request reaches these servers by naming their ids in
``mcp_server_ids``; see `docs/mcp.md`.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import CurrentIdentity, WorkspaceMcpServerServiceDep, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerAccess,
    WorkspaceMcpServerAccessCheck,
    WorkspaceMcpServerCheckRequest,
    WorkspaceMcpServerCreate,
    WorkspaceMcpServerGrantsCreate,
    WorkspaceMcpServerGrantUpdate,
    WorkspaceMcpServerHealth,
    WorkspaceMcpServerKeyAccess,
    WorkspaceMcpServerPublic,
    WorkspaceMcpServersPublic,
    WorkspaceMcpServerUpdate,
)

# Auth is declared on the router, matching `routes/workspaces.py`: every
# handler here needs the master key, and a future one that forgot the
# decorator would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/workspaces/{workspace_id}/mcp-servers",
    tags=["mcp-servers"],
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def list_workspace_mcp_servers(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> WorkspaceMcpServersPublic:
    """List the MCP servers configured for a workspace.

    Readable by any member who can see the workspace: these servers act on
    every request a member sends through it, so what they are is a member's
    to view (otari-ai#1942). Changing them stays with organization
    owners/admins or this workspace's owners/admins. Authorization tokens are
    never included; each server reports only whether it has one.
    """
    return await service.list_servers(user=current_identity, workspace_id=workspace_id, skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_workspace_mcp_server(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceMcpServerCreate,
) -> WorkspaceMcpServerPublic:
    """Register an MCP server for a workspace. Organization owners/admins or this workspace's owners/admins.

    The authorization token is encrypted at rest and never returned. The URL
    is checked for SSRF safety here as well as on the request path, and must
    use https when a token is set. A name already used in this workspace is
    refused with a 409.

    Once stored, the server is connected to and its tools listed, and the
    outcome is returned as `health`. A server that fails this check is still
    stored, and `health.error` says why.
    """
    return await service.create_server(user=current_identity, workspace_id=workspace_id, request=body)


@router.patch("/{server_id}")
async def update_workspace_mcp_server(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    body: WorkspaceMcpServerUpdate,
) -> WorkspaceMcpServerPublic:
    """Update a server. Organization owners/admins or this workspace's owners/admins.

    Only the fields sent are applied. Omit `authorization_token` to leave the
    stored token alone, send an empty string to clear it, or send a value to
    rotate it. A change to the URL, the token or the allow-list checks the
    server again, as creating it does. `everyone_tools` is what every key of
    the workspace may call; the server's grants override it per key or user.
    """
    return await service.update_server(
        user=current_identity,
        workspace_id=workspace_id,
        server_id=server_id,
        request=body,
    )


@router.get("/keys/{api_key_id}/access")
async def get_workspace_mcp_key_access(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    api_key_id: str,
) -> list[WorkspaceMcpServerKeyAccess]:
    """What one key reaches on each of the workspace's MCP servers.

    Organization owners/admins or this workspace's owners/admins.

    For each server, oldest first: the rule that decides (the key's own grant, its owner's, or the
    server's `everyone_tools`) and that rule's tool patterns, beside the server's own `allowed_tools`
    ceiling. A key of another workspace is refused with a 400.
    """
    return await service.key_access(user=current_identity, workspace_id=workspace_id, api_key_id=api_key_id)


@router.get("/{server_id}")
async def get_workspace_mcp_server(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerPublic:
    """One MCP server of a workspace. Readable by any member who can see the workspace, as the list is."""
    return await service.get_server(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.get("/{server_id}/access")
async def get_workspace_mcp_server_access(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerAccess:
    """A server's access list. Organization owners/admins or this workspace's owners/admins.

    `everyone_tools` is what every key of the workspace may call. Each grant overrides it for one
    key, or for one user and so every key they own; a key's own grant decides over its owner's.
    Tool patterns are names, `*` or `prefix*`, and `[]` denies. The server's `allowed_tools` is a
    ceiling over all of them. A server holds at most 1000 grants, returned oldest first.
    """
    return await service.access(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.post("/{server_id}/access/grants")
async def add_workspace_mcp_server_grants(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    body: WorkspaceMcpServerGrantsCreate,
) -> WorkspaceMcpServerAccess:
    """Give keys and users the same tools on a server.

    Organization owners/admins or this workspace's owners/admins.

    A key or user that already has a grant on the server gets these tools instead. A key must
    belong to this workspace, and a user must own a key in it; either is otherwise refused with
    a 400. Answers with the server's whole access list.
    """
    return await service.add_grants(user=current_identity, workspace_id=workspace_id, server_id=server_id, request=body)


@router.patch("/{server_id}/access/grants/{grant_id}")
async def update_workspace_mcp_server_grant(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    grant_id: uuid.UUID,
    body: WorkspaceMcpServerGrantUpdate,
) -> WorkspaceMcpServerAccess:
    """Change the tools of one grant. Organization owners/admins or this workspace's owners/admins."""
    return await service.update_grant(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, grant_id=grant_id, request=body
    )


@router.delete("/{server_id}/access/grants/{grant_id}")
async def delete_workspace_mcp_server_grant(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    grant_id: uuid.UUID,
) -> WorkspaceMcpServerAccess:
    """Remove one grant, so its key or user falls back to the next rule.

    Organization owners/admins or this workspace's owners/admins.
    """
    return await service.delete_grant(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, grant_id=grant_id
    )


@router.get("/{server_id}/access/check")
async def check_workspace_mcp_server_access(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    api_key_id: Annotated[str, Query(min_length=1, max_length=64, description="The key to test")],
    tool: Annotated[str, Query(min_length=1, max_length=256, description="The tool name to test")],
) -> WorkspaceMcpServerAccessCheck:
    """Whether a key may call a tool on a server, and which rule decides.

    Organization owners/admins or this workspace's owners/admins.

    Applies the same order a request does: the server must be enabled and its `allowed_tools`
    must include the tool, then the key's own grant decides, else its owner's, else the
    server's `everyone_tools`. A key of another workspace is refused with a 400.
    """
    return await service.check_access(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, api_key_id=api_key_id, tool=tool
    )


@router.post(
    "/{server_id}/check",
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Every check slot on this gateway was busy; nothing was recorded. Try again shortly."
        }
    },
)
async def check_workspace_mcp_server(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerPublic:
    """Check a server's connection now. Organization owners/admins or this workspace's owners/admins.

    Opens one session, lists the server's tools, and stores the outcome as
    `health`. A server that fails is still a 200, with `health.status` set to
    `failing`. A 429 means every check slot on this gateway was busy and
    nothing was recorded.
    """
    return await service.check_server(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.post(
    "/check",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "The URL failed the SSRF or TLS check; nothing was sent to it."},
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Every check slot on this gateway was busy. Try again shortly."
        },
    },
)
async def check_workspace_mcp_server_connection(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceMcpServerCheckRequest,
) -> WorkspaceMcpServerHealth:
    """Check connection details without storing them. Organization owners/admins or this workspace's owners/admins.

    What a form's Test button calls before it saves. Answers with what a stored
    server's `health` would say, and records nothing. Pass `server_id` and omit
    `authorization_token` to test with that server's stored token. The URL is
    checked for SSRF safety first, with the same 400 a create gets.
    """
    return await service.check_connection(user=current_identity, workspace_id=workspace_id, request=body)


@router.delete("/{server_id}")
async def delete_workspace_mcp_server(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> Message:
    """Delete a server and the token stored with it. Organization owners/admins or this workspace's owners/admins."""
    await service.delete_server(user=current_identity, workspace_id=workspace_id, server_id=server_id)
    return Message(message="MCP server deleted")
