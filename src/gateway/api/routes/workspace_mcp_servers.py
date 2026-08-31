"""Workspace-scoped MCP server management (standalone mode only).

Thin composition over
`gateway.services.tenancy.workspace_mcp_server_service`, following
`routes/workspace_member_budget_policies.py`'s shape: master key on the
router, plus the caller's tenancy identity for the per-workspace role check
the service runs. A request reaches these servers by naming their ids in
``mcp_server_ids``; see `docs/mcp.md`.

Hybrid mode does not mount this router. There the same rows live on the
platform and are managed from otari.ai, which is also what resolves them for a
request.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerCreate,
    WorkspaceMcpServerPublic,
    WorkspaceMcpServerService,
    WorkspaceMcpServersPublic,
    WorkspaceMcpServerUpdate,
)

# Auth is declared on the router, matching `routes/workspaces.py`: every
# handler here needs the master key, and a future one that forgot the
# decorator would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/mcp-servers",
    tags=["mcp-servers"],
    dependencies=[Depends(verify_master_key)],
)


def get_workspace_mcp_server_service(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceMcpServerService:
    """Build the service on the request's session."""
    return WorkspaceMcpServerService(db)


WorkspaceMcpServerServiceDep = Annotated[WorkspaceMcpServerService, Depends(get_workspace_mcp_server_service)]


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
    rotate it.
    """
    return await service.update_server(
        user=current_identity,
        workspace_id=workspace_id,
        server_id=server_id,
        request=body,
    )


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
