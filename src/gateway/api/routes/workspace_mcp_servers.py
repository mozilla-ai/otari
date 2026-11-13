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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response, status
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect

from gateway.api.deps import (
    CurrentIdentity,
    McpServerLogoServiceDep,
    PersonGrowthSignalPortDep,
    WorkspaceMcpServerServiceDep,
    verify_master_key,
)
from gateway.api.routes._helpers import bounded_body
from gateway.api.routes._logo_response import LOGO_RESPONSES, logo_response
from gateway.api.routes.organizations import Message
from gateway.schemas.tools import (
    McpConnectionStart,
    McpConnectionStarted,
    WorkspaceMcpServerLogoPublic,
    WorkspaceMcpServerLogosPublic,
)
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerAccess,
    WorkspaceMcpServerAccessCheck,
    WorkspaceMcpServerCheckRequest,
    WorkspaceMcpServerCreate,
    WorkspaceMcpServerGrantsCreate,
    WorkspaceMcpServerGrantUpdate,
    WorkspaceMcpServerHealth,
    WorkspaceMcpServerKeyAccess,
    WorkspaceMcpServerMyTools,
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
    growth: PersonGrowthSignalPortDep,
    background_tasks: BackgroundTasks,
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
    return await service.create_server(
        user=current_identity,
        workspace_id=workspace_id,
        request=body,
        growth=growth,
        background_tasks=background_tasks,
    )


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
    ceiling. A key of another workspace is answered with a 404.
    """
    return await service.key_access(user=current_identity, workspace_id=workspace_id, api_key_id=api_key_id)


@router.get("/logos")
async def list_workspace_mcp_server_logos(
    service: McpServerLogoServiceDep, current_identity: CurrentIdentity, workspace_id: uuid.UUID
) -> WorkspaceMcpServerLogosPublic:
    """The logo of each of the workspace's MCP servers, and whether it needs a tile to stay visible.

    Readable by any member who can see the workspace, as the servers are. A logo is
    an admin's upload, else the catalog's for the server's URL, else one the gateway
    found on the server's host. Listing them makes the gateway fetch nothing.
    """
    return await service.list_logos(user=current_identity, workspace_id=workspace_id)


@router.get("/{server_id}")
async def get_workspace_mcp_server(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerPublic:
    """One MCP server of a workspace. Readable by any member who can see the workspace, as the list is."""
    return await service.get_server(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.get("/{server_id}/my-tools")
async def get_my_workspace_mcp_server_tools(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerMyTools:
    """The tools the caller may use on a server. Any member who can see the workspace.

    Applies the caller's own rule: their user's grant, else the server's `everyone_tools`, within
    its `allowed_tools`; a disabled server admits nothing. The tool list comes from the server's
    last passing check; where there is none and the server is per-user, it is listed with the
    caller's own connection and not stored. `known` is false when neither is available.
    """
    return await service.my_tools(user=current_identity, workspace_id=workspace_id, server_id=server_id)


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
    Tool patterns are names, `*`, `prefix*`, or `@read` for the tools the server's `read_tools`
    labels read, and `[]` denies. The server's `allowed_tools` is a ceiling over all of them. A
    server holds at most 1000 grants, returned oldest first.
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
    belong to this workspace, and a user must own a key in it or be someone the Playground lets in
    (a member of the workspace, or an owner or admin of its organization), named by their
    identity's canonical id; a deleted user, or anyone else, is refused with a 400. Answers with the
    server's whole access list.
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
    server's `everyone_tools`. A key of another workspace is answered with a 404.
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
    nothing was recorded. A per-user server holds no token to check with and
    is refused with a 409: start its check with `POST .../check/sign-in`.
    """
    return await service.check_server(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.post("/{server_id}/check/sign-in")
async def start_workspace_mcp_server_check_sign_in(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    body: McpConnectionStart,
) -> McpConnectionStarted:
    """Start a sign-in made only to check a per-user server. Organization or workspace owners/admins.

    Returns the authorization page to send the browser to, as starting a connection does. The
    caller signs in with any account; finishing the sign-in (`POST /mcp-connections/complete`)
    lists the server's tools once with the token it returns, stores the outcome as `health`, and
    drops the token, revoking it where the authorization server allows. No connection is created,
    and the caller's own connection, if they have one, is left as it was. A server in `shared` mode
    is refused with a 409.
    """
    return await service.start_check_sign_in(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, return_to=body.return_to
    )


@router.post("/{server_id}/shared-account")
async def start_workspace_mcp_server_shared_sign_in(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    body: McpConnectionStart,
) -> McpConnectionStarted:
    """Start a sign-in for everyone, to a shared server. Organization or workspace owners/admins.

    Returns the authorization page to send the browser to, as starting a connection does. The
    caller signs in with the account the whole workspace should act as, so a service account is
    the better choice. Finishing the sign-in (`POST /mcp-connections/complete`) stores that
    account as the server's `shared_account`: every caller's call then carries its token, which
    Otari renews as it does a user's own, and a stored `authorization_token` is removed. A server
    in `per_user` mode is refused with a 409.
    """
    return await service.start_shared_sign_in(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, return_to=body.return_to
    )


@router.delete("/{server_id}/shared-account")
async def end_workspace_mcp_server_shared_account(
    service: WorkspaceMcpServerServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerPublic:
    """End a shared server's account. Organization owners/admins or this workspace's owners/admins.

    The stored tokens are deleted, and the authorization server is asked to revoke them where it
    publishes a revocation endpoint. The server is then checked again with no credential, and
    comes back with `needs_setup` when it asks for one. A server with no shared account is a 404.
    """
    return await service.disconnect_shared(user=current_identity, workspace_id=workspace_id, server_id=server_id)


@router.post(
    "/check",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": (
                "The URL failed the SSRF or TLS check, or is on another origin than the stored server "
                "whose credential was asked for; nothing was sent to it."
            )
        },
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
    `authorization_token` to test with that server's stored credential, which is
    only sent to a URL on that server's own origin (scheme, host and port): any
    other is a 400. The URL is checked for SSRF safety first, with the same 400
    a create gets.
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


# A 256 px PNG from the dashboard is far smaller; this bounds the body, multipart framing included.
_MAX_LOGO_BODY_BYTES = 520 * 1024
_LOGO_TOO_LARGE = "The logo is larger than 512 KB"
_LOGO_UPLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "tile": {
            "type": "string",
            "enum": ["auto", "always", "never"],
            "description": "Whether the logo sits on a tile: where it would fade out, always, or never.",
        },
        "file": {
            "type": "string",
            "contentMediaType": "image/png",
            "description": "A PNG of at most 512 KB to show instead of the server's own logo. Leave out to keep it.",
        },
    },
    "required": ["tile"],
}


@router.get("/{server_id}/logo", include_in_schema=False, responses=LOGO_RESPONSES)
async def get_workspace_mcp_server_logo(
    request: Request,
    service: McpServerLogoServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> Response:
    """A server's logo, for any member who can see the workspace; `logo_url` from the logo list points here."""
    logo = await service.logo(user=current_identity, workspace_id=workspace_id, server_id=server_id)
    return logo_response(request, logo)


@router.put(
    "/{server_id}/logo",
    responses={
        413: {"description": "The body is larger than the logo limit."},
        422: {"description": "The file is not a PNG, or the tile is not one of the choices."},
    },
    openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": _LOGO_UPLOAD_SCHEMA}}}
    },
)
async def set_workspace_mcp_server_logo(
    request: Request,
    service: McpServerLogoServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerLogoPublic:
    """Set how a server's logo shows: the tile choice, and an uploaded PNG in place of its own logo.

    Organization owners/admins or this workspace's owners/admins. The upload is kept
    apart from the logo the gateway fetches, so no fetch or catalog sync replaces it.
    """
    try:
        async with bounded_body(request, _MAX_LOGO_BODY_BYTES, _LOGO_TOO_LARGE).form(max_files=1, max_fields=1) as form:
            tile = form.get("tile")
            file = form.get("file")
            if not isinstance(tile, str):
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Send the tile choice in the field 'tile'.")
            content = await file.read() if isinstance(file, UploadFile) else None
    except ClientDisconnect:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The upload was interrupted.") from None
    return await service.set_logo(
        user=current_identity, workspace_id=workspace_id, server_id=server_id, tile=tile, content=content
    )


@router.delete("/{server_id}/logo")
async def remove_workspace_mcp_server_logo(
    service: McpServerLogoServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> WorkspaceMcpServerLogoPublic:
    """Remove a server's uploaded logo, so it shows its own again. The tile choice stays.

    Organization owners/admins or this workspace's owners/admins.
    """
    return await service.remove_upload(user=current_identity, workspace_id=workspace_id, server_id=server_id)
