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
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse

from gateway.api.deps import (
    CurrentIdentity,
    McpConnectionServiceDep,
    WorkspaceMcpServerServiceDep,
    get_config,
    get_session_identity,
    verify_master_key,
)
from gateway.api.routes._mcp_sign_in_pages import (
    SignInPage,
    confirm_page,
    conversation_page,
    leaving_page,
    other_user_page,
)
from gateway.core.config import GatewayConfig
from gateway.exceptions import TenancyError
from gateway.exceptions.tools_exceptions import McpConnectionFlowError
from gateway.log_config import logger
from gateway.models.tenancy import User as TenancyUser
from gateway.schemas.tools import (
    McpConnectionComplete,
    McpConnectionCompleted,
    McpConnectionsPublic,
    McpConnectionStart,
    McpConnectionStarted,
    McpServerConnectionsPublic,
)
from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServerService
from gateway.services.tools import McpConnectionService

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

# Where an authorization server sends the browser back to, and where a conversation's sign-in link
# opens. A request arriving from another site carries no credential of ours, because the session
# cookie is SameSite=Strict; the link's own page posts from this origin, so its POST does.
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


async def _sign_in_link_page(
    service: WorkspaceMcpServerService, ticket: str, opened_by: TenancyUser | None
) -> tuple[Response, bool]:
    """The page a sign-in link shows before anything starts, and whether it offers to continue."""
    try:
        preview = await service.sign_in_link_preview(ticket=ticket, opened_by=opened_by)
    except McpConnectionFlowError:
        return conversation_page(SignInPage.LINK_EXPIRED), False
    except TenancyError as exc:
        # The server is gone or changed mode: the person can only try later.
        logger.info("MCP sign-in link refused reason=%s", type(exc).__name__)
        return conversation_page(SignInPage.CANNOT_START), False
    if preview.other_user and preview.opened_as is not None:
        logger.warning("MCP sign-in link refused reason=other_identity")
        return (
            other_user_page(
                server_name=preview.server_name, user_label=preview.user_label, opened_as=preview.opened_as
            ),
            False,
        )
    return (
        confirm_page(
            ticket=ticket,
            server_name=preview.server_name,
            server_host=preview.server_host,
            user_label=preview.user_label,
            opened_as=preview.opened_as,
        ),
        True,
    )


def _posted_from_this_gateway(request: Request, config: GatewayConfig) -> bool:
    """Whether a POST came from a page of this gateway's own origin, so no other site can press Continue.

    A browser says so itself in ``Sec-Fetch-Site``; one too old to send it is held to its ``Origin``.
    """
    fetch_site = request.headers.get("Sec-Fetch-Site")
    if fetch_site is not None:
        return fetch_site == "same-origin"
    origin, own = urlsplit(request.headers.get("Origin", "")), urlsplit(config.public_base_url or "")
    return bool(origin.netloc) and (origin.scheme, origin.netloc.lower()) == (own.scheme, own.netloc.lower())


@callback_router.get("/sign-in", include_in_schema=False)
async def open_mcp_sign_in_link(
    service: WorkspaceMcpServerServiceDep,
    opened_by: Annotated[TenancyUser | None, Depends(get_session_identity)],
    ticket: Annotated[str, Query(max_length=1024)] = "",
) -> Response:
    """Open a sign-in link a conversation showed: say whose it is and which server, and ask before starting.

    Nothing starts and nothing is stored here, so a link previewer or a scanner that fetches the
    link does nothing. The link names its server and user itself, so no session is needed; one
    opened in the dashboard session of another user is refused. A link that is forged, altered or
    expired gets a plain page saying to ask for a new one.
    """
    page, _ = await _sign_in_link_page(service, ticket, opened_by)
    return page


@callback_router.post("/sign-in", include_in_schema=False)
async def start_mcp_sign_in_from_link(
    request: Request,
    service: WorkspaceMcpServerServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    opened_by: Annotated[TenancyUser | None, Depends(get_session_identity)],
    ticket: Annotated[str, Form(max_length=1024)] = "",
) -> Response:
    """Start the sign-in a link names, once the person pressed Continue on the page the link opened.

    Only that page can ask: a POST from anywhere else is answered with the page itself, as a GET
    would be. A dashboard session it arrives in must be the link's own user, and the sign-in is
    then bound to that session.
    """
    page, offered = await _sign_in_link_page(service, ticket, opened_by)
    if not offered or not _posted_from_this_gateway(request, config):
        return page
    try:
        url = await service.start_from_sign_in_link(ticket=ticket, opened_by=opened_by)
    except McpConnectionFlowError:
        return conversation_page(SignInPage.LINK_EXPIRED)
    except TenancyError as exc:
        # The deployment cannot run a sign-in, or the authorization server could not be reached.
        logger.info("MCP sign-in link refused reason=%s", type(exc).__name__)
        return conversation_page(SignInPage.CANNOT_START)
    if urlsplit(url).scheme not in ("https", "http"):
        return conversation_page(SignInPage.CANNOT_START)
    return leaving_page(url)


@callback_router.get("/client-metadata", include_in_schema=False)
async def mcp_client_metadata(config: Annotated[GatewayConfig, Depends(get_config)]) -> Response:
    """Serve Otari's client ID metadata document, which an authorization server reads in place of a registration.

    It holds nothing secret: the client's name, the one redirect URI and that it is a public PKCE
    client. A deployment with no public https address serves none.
    """
    document = McpConnectionService.client_metadata(config)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return JSONResponse(document, headers={"Cache-Control": "public, max-age=3600"})


@callback_router.get("/callback", include_in_schema=False)
async def mcp_connection_callback(
    connections: McpConnectionServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    state: Annotated[str | None, Query(max_length=512)] = None,
    code: Annotated[str | None, Query(max_length=4096)] = None,
    error: Annotated[str | None, Query(max_length=256)] = None,
    iss: Annotated[str | None, Query(max_length=2048)] = None,
) -> Response:
    """Receive an authorization server's redirect and send the browser on to where the sign-in began.

    A sign-in started from the dashboard ends on the dashboard's own page, a
    refusal included. One started from a conversation's link ends on a plain
    page saying to go back to the conversation, since that person may have no
    dashboard session at all.
    """
    outcome = await connections.receive_callback(state=state, code=code, error=error, iss=iss)
    if outcome.conversation:
        if outcome.connected is not None:
            return conversation_page(SignInPage.CONNECTED, server_name=outcome.server_name)
        return conversation_page(SignInPage.FAILED)
    return RedirectResponse(
        url=config.ui_link(f"{McpConnectionService.DASHBOARD_CALLBACK_ROUTE}?{urlencode(outcome.query())}"),
        status_code=status.HTTP_303_SEE_OTHER,
    )
