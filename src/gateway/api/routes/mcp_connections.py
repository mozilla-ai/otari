"""A user's own connections to per-user MCP servers.

A stored server in ``per_user`` mode sends each caller's own token, which a
user gets by signing in to the server. These routes start that sign-in, finish
it, list the caller's connections and end one, and let a workspace manager see
and end the connections of others. Thin composition over
`gateway.services.tenancy.workspace_mcp_server_service`, which settles who may
ask, and `gateway.services.tools`, which runs the sign-in; see
`docs/connect-your-account-to-an-mcp-server.md`.
"""

import uuid
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import RedirectResponse

from gateway.api.deps import (
    CurrentIdentity,
    McpConnectionServiceDep,
    WorkspaceMcpServerServiceDep,
    get_config,
    verify_master_key,
)
from gateway.core.config import GatewayConfig
from gateway.schemas.tools import (
    McpConnectionComplete,
    McpConnectionCompleted,
    McpConnectionsPublic,
    McpConnectionStart,
    McpConnectionStarted,
    McpServerConnectionsPublic,
)

# The dashboard route that finishes a sign-in a dashboard session started.
DASHBOARD_CALLBACK_ROUTE = "/#/mcp-connections/callback"

# Auth is declared on the router, as `routes/workspace_mcp_servers.py` does.
router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["mcp-connections"],
    dependencies=[Depends(verify_master_key)],
)

# Finishing a sign-in names no workspace: the state does.
session_router = APIRouter(
    prefix="/mcp-connections",
    tags=["mcp-connections"],
    dependencies=[Depends(verify_master_key)],
)

# Where an authorization server sends the browser back to. It carries no credential of ours, because
# the session cookie is SameSite=Strict and does not travel on a redirect from another site.
callback_router = APIRouter(prefix="/mcp-connections", tags=["mcp-connections"])


@router.get("/mcp-connections")
async def list_my_mcp_connections(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> McpConnectionsPublic:
    """List the caller's own connections to this workspace's MCP servers.

    A connection is the caller's sign-in to a server whose `credential_mode`
    is `per_user`. Any member who can see the workspace may call this. Tokens
    are never included.
    """
    return await service.my_connections(user=current_identity, workspace_id=workspace_id)


@router.post("/mcp-connections/{server_id}")
async def start_mcp_connection(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    body: McpConnectionStart,
) -> McpConnectionStarted:
    """Start the caller's own sign-in to a per-user MCP server.

    Returns the server's authorization page to send the browser to. Otari
    discovers the server's authorization server, registers itself as a client
    where the server holds no client ID, and uses the authorization code flow
    with S256 PKCE. The sign-in is bound to the caller: it finishes only in
    their own dashboard session, once, within ten minutes. The connection then
    belongs to the caller's user, so every key they own in this workspace
    reaches the server as them.

    Any member who can see the workspace may call this. A server in `shared`
    mode is refused with a 409, and a deployment with no `public_base_url`
    with a 400, because the address the authorization server returns to is
    derived from it.
    """
    return await service.start_connection(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, return_to=body.return_to
    )


@router.delete("/mcp-connections/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_my_mcp_connection(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> None:
    """End the caller's own connection to an MCP server.

    The stored tokens are deleted, and the authorization server is asked to
    revoke them where it publishes a revocation endpoint. The caller's keys
    then get `mcp_sign_in_required` from this server until they sign in again.
    """
    await service.disconnect(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.get("/mcp-servers/{server_id}/connections")
async def list_mcp_server_connections(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> McpServerConnectionsPublic:
    """List who is connected to a per-user MCP server. Organization owners/admins or this workspace's owners/admins.

    Each entry names the user the connection belongs to and, when that user
    is a dashboard account, its email and name, oldest connection first.
    `count` is the total. Tokens are never included.
    """
    return await service.server_connections(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, skip=skip, limit=limit
    )


@router.delete("/mcp-servers/{server_id}/connections/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_mcp_server_connection(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    user_id: str,
) -> None:
    """End one user's connection to an MCP server. Organization owners/admins or this workspace's owners/admins.

    The stored tokens are deleted and the authorization server is asked to
    revoke them. That user's keys get `mcp_sign_in_required` from this server
    until they sign in again.
    """
    await service.remove_connection(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, user_id=user_id
    )


@session_router.post("/complete")
async def complete_mcp_connection(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    body: McpConnectionComplete,
) -> McpConnectionCompleted:
    """Finish a sign-in to a per-user MCP server, from the code its authorization server returned.

    The dashboard calls this from the page the callback lands on. The state
    is consumed whatever happens next, so it finishes one sign-in at most, and
    it must be one the caller started: another person's is refused with a 400.
    Send the redirect's `iss` too when it had one: an answer naming another
    authorization server is refused (RFC 9207).
    """
    return await service.complete_connection(user=current_identity, state=body.state, code=body.code, iss=body.iss)


@callback_router.get("/callback", include_in_schema=False)
async def mcp_connection_callback(
    connections: McpConnectionServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    state: Annotated[str | None, Query(max_length=512)] = None,
    code: Annotated[str | None, Query(max_length=4096)] = None,
    error: Annotated[str | None, Query(max_length=256)] = None,
    iss: Annotated[str | None, Query(max_length=2048)] = None,
) -> RedirectResponse:
    """Receive an authorization server's redirect and send the browser on into the dashboard.

    The browser always ends on the dashboard's own page: a person is here
    because an authorization server sent them, so a refusal is shown there too.
    """
    outcome = await connections.receive_callback(state=state, code=code, error=error, iss=iss)
    return RedirectResponse(
        url=config.ui_link(f"{DASHBOARD_CALLBACK_ROUTE}?{urlencode(outcome.query())}"),
        status_code=status.HTTP_303_SEE_OTHER,
    )
