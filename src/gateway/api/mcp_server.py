"""Otari's own MCP server at ``/mcp``: what a person sees on the dashboard, for their agents to read.

Every tool is one read of the management API, made in-process with the caller's personal access
token, so what a tool answers is exactly what the route answers that person: the same scope, the
same role checks and the same refusals. Nothing here decides who may see what.

The transport is stateless streamable HTTP with JSON responses, so any replica answers any call.
A call without a live token is answered 401 naming this server's protected-resource metadata
(RFC 9728), which names Otari's OAuth endpoints (RFC 8414), so an MCP client can register
itself and have its person sign in rather than be handed a token.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware, get_access_token
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.authentication import AuthCredentials
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from gateway.api.deps import build_personal_access_token_service, oauth_app_issuer, reading_through_mcp
from gateway.auth.models import MCP_ACCESS_TOKEN_PREFIX, PERSONAL_ACCESS_TOKEN_PREFIX
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import create_unit_of_work
from gateway.version import __version__

MCP_PATH = "/mcp"
_RESOURCE_METADATA_PATH = "/.well-known/oauth-protected-resource"
_AUTHORIZATION_SERVER_METADATA_PATH = "/.well-known/oauth-authorization-server"
# How much of one answer a tool hands the model. A page of usage rows can run long, and a
# truncated answer that says so beats one that fills the caller's context.
_MAX_RESULT_CHARS = 100_000
# The loopback request's origin. Only the in-process transport ever sees it.
_LOOPBACK_BASE_URL = "http://otari.internal"
_LOOPBACK_TIMEOUT_SECONDS = 60

_INSTRUCTIONS = """\
Read-only access to an Otari AI gateway, as the person whose access token you hold: their \
organization's spend, usage, requests, budgets, keys and workspaces, exactly as their dashboard \
shows them. Dates are ISO 8601. Money is USD. Start with get_overview or get_usage_summary; use \
api_get for any other GET endpoint of the management API."""

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

WorkspaceId = Annotated[
    str | None, Field(description="Narrow to one workspace (a UUID from list_workspaces). Omit for every one.")
]
StartDate = Annotated[str | None, Field(description="Window start, inclusive (ISO 8601). Omit for the default.")]
EndDate = Annotated[str | None, Field(description="Window end, exclusive (ISO 8601). Omit for now.")]
Models = Annotated[list[str] | None, Field(description="Only these models, as provider:model.")]
ApiKeyIds = Annotated[list[str] | None, Field(description="Only these API key IDs.")]
UserIds = Annotated[list[str] | None, Field(description="Only these user IDs.")]
Skip = Annotated[int, Field(ge=0, description="Rows to skip, for paging.")]
Limit = Annotated[int, Field(ge=1, le=200, description="Rows to return.")]


class _AccessTokenVerifier:
    """Admits a bearer that is a live personal access token or MCP client access token, as its account."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith((PERSONAL_ACCESS_TOKEN_PREFIX, MCP_ACCESS_TOKEN_PREFIX)):
            return None
        async with create_unit_of_work() as uow:
            owner = await build_personal_access_token_service(uow).authenticate(token)
        if owner is None:
            return None
        return AccessToken(token=token, client_id="personal-access-token", scopes=[], subject=str(owner.id))


def _params(**values: Any) -> dict[str, Any]:
    return {name: value for name, value in values.items() if value is not None}


def _truncate(text: str) -> str:
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + f"\n[truncated: the answer was {len(text)} characters; page or filter it]"


def _document(response: httpx.Response) -> Any:
    """A response's JSON body, or its text where it answered something else."""
    try:
        return response.json()
    except ValueError:
        return response.text


def _render(document: Any) -> str:
    """One answer as the model reads it, truncated whole rather than in pieces."""
    text = document if isinstance(document, str) else json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    return _truncate(text)


def _refusal(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return f"Otari answered {response.status_code}: {detail if isinstance(detail, str) else response.reason_phrase}"


def build_mcp_server(app: ASGIApp) -> MCPServer[Any]:
    """The MCP server, whose tools read through ``app`` with the caller's own token."""
    # MCPServer calls logging.basicConfig with a Rich handler on the root logger, which would print every
    # library's INFO lines beside Otari's own logging, so the root logger is put back as it was.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    server: MCPServer[Any] = MCPServer(
        name="otari",
        title="Otari",
        instructions=_INSTRUCTIONS,
        version=__version__,
    )
    root.handlers[:] = handlers
    root.setLevel(level)

    async def fetch(path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """One management GET, answered in-process by the app with the token this call came in with."""
        access = get_access_token()
        if access is None:
            raise ToolError("Not authenticated")
        # ASGITransport awaits the app directly, so httpx's own timeout never fires here.
        try:
            with reading_through_mcp():
                async with (
                    asyncio.timeout(_LOOPBACK_TIMEOUT_SECONDS),
                    httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=_LOOPBACK_BASE_URL) as client,
                ):
                    response = await client.get(
                        f"{API_ROOT}{path}", params=params, headers={"Authorization": f"Bearer {access.token}"}
                    )
        except TimeoutError:
            raise ToolError(f"Otari did not answer within {_LOOPBACK_TIMEOUT_SECONDS} seconds") from None
        if response.is_error:
            raise ToolError(_refusal(response))
        return response

    async def get(path: str, params: dict[str, Any] | None = None) -> str:
        return _render(_document(await fetch(path, params)))

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def get_context() -> str:
        """Who the token speaks for: the person, their active organization and their role in it."""
        return await get("/organizations/me")

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def list_workspaces(skip: Skip = 0, limit: Limit = 100) -> str:
        """The workspaces of the organization the person can see, with their IDs."""
        return await get("/workspaces", _params(skip=skip, limit=limit))

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def get_overview(workspace_id: WorkspaceId = None) -> str:
        """The dashboard overview: active keys and members, and how budgets and spend ceilings stand."""
        return await get("/overview", _params(workspace_id=workspace_id))

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def get_usage_summary(
        start_date: StartDate = None,
        end_date: EndDate = None,
        workspace_id: WorkspaceId = None,
        model: Models = None,
        api_key_id: ApiKeyIds = None,
        user_id: UserIds = None,
        bucket: Annotated[str, Field(pattern="^(hour|day)$", description="Time-series granularity.")] = "day",
        dimensions: Annotated[
            list[str] | None,
            Field(
                description="Breakdowns to compute, each filling by_<value>: model, user, api_key, source, "
                "source_label, endpoint, provider, status_code, tool; or none for totals only. Omit for all."
            ),
        ] = None,
    ) -> str:
        """Spend, tokens and request counts for the organization over a window, with breakdowns and a series."""
        return await get(
            "/organizations/me/usage/summary",
            _params(
                start_date=start_date,
                end_date=end_date,
                workspace_id=workspace_id,
                model=model,
                api_key_id=api_key_id,
                user_id=user_id,
                bucket=bucket,
                dimensions=dimensions,
            ),
        )

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def list_requests(
        start_date: StartDate = None,
        end_date: EndDate = None,
        workspace_id: WorkspaceId = None,
        model: Models = None,
        api_key_id: ApiKeyIds = None,
        user_id: UserIds = None,
        status: Annotated[
            str | None, Field(pattern="^(success|error|absorbed)$", description="Only this outcome.")
        ] = None,
        skip: Skip = 0,
        limit: Limit = 50,
    ) -> str:
        """Individual requests, newest first: model, key, user, tokens, cost, status and request ID."""
        return await get(
            "/organizations/me/usage",
            _params(
                start_date=start_date,
                end_date=end_date,
                workspace_id=workspace_id,
                model=model,
                api_key_id=api_key_id,
                user_id=user_id,
                status=status,
                skip=skip,
                limit=limit,
            ),
        )

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def get_request(
        request_id: Annotated[str, Field(min_length=1, max_length=200, description="An Otari-Request-ID.")],
    ) -> str:
        """The trace of one request by its Otari-Request-ID: its spans, timings, tool calls and errors."""
        return await get(f"/organizations/me/traces/requests/{quote(request_id, safe='')}")

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def list_budgets(skip: Skip = 0, limit: Limit = 100) -> str:
        """The organization's budgets and spend ceilings, with what each has spent of its limit."""
        params = _params(skip=skip, limit=limit)
        budgets = await fetch("/organizations/me/budgets", params)
        ceilings = await fetch("/organizations/me/spend-ceilings", params)
        # One document, rendered and truncated as one: two rendered answers spliced together would be
        # cut apart mid-object whenever either of them ran long.
        return _render({"budgets": _document(budgets), "spend_ceilings": _document(ceilings)})

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def list_api_keys(workspace_id: WorkspaceId = None, skip: Skip = 0, limit: Limit = 100) -> str:
        """The API keys the person can see, with their owners, scopes and status. Never the keys themselves."""
        return await get("/organizations/me/keys", _params(workspace_id=workspace_id, skip=skip, limit=limit))

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    async def api_get(
        path: Annotated[
            str,
            Field(
                pattern=r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*$",
                max_length=500,
                description="A management API path below /api/v1, such as /organizations/me/members.",
            ),
        ],
        query: Annotated[
            dict[str, str | list[str]] | None, Field(description="Query parameters; a list repeats the name.")
        ] = None,
    ) -> str:
        """GET any endpoint of Otari's management API (reference: docs/public/openapi.json), as the person."""
        if "/../" in f"{path}/" or "//" in path:
            raise ToolError("The path must be a plain path below /api/v1")
        return await get(path, query)

    return server


class McpEndpoint:
    """The ASGI app mounted at ``/mcp``, which answers only while the lifespan runs ``serving``.

    It authenticates every call itself, because the 401 it answers names metadata at the address the
    caller reached, which only the request knows. The SDK's session manager runs once per instance,
    and an app's lifespan can run more than once, so each run builds the transport afresh.
    """

    def __init__(self, server: MCPServer[Any], config: GatewayConfig) -> None:
        self._server, self._config = server, config
        self._verifier = _AccessTokenVerifier()
        self._app: ASGIApp | None = None

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        app = self._server.streamable_http_app(
            streamable_http_path=MCP_PATH,
            json_response=True,
            stateless_http=True,
            # A bearer token, never a cookie, authenticates every call, so there is no ambient
            # credential for a rebinding page to ride; and the SDK's default allows only localhost.
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        async with self._server.session_manager.run():
            self._app = AuthContextMiddleware(app)
            try:
                yield
            finally:
                self._app = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._app is None:
            await JSONResponse({"detail": "The MCP server is not running"}, status_code=503)(scope, receive, send)
            return
        request = Request(scope)
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        access = await self._verifier.verify_token(token.strip()) if scheme.lower() == "bearer" else None
        if access is None:
            metadata = oauth_app_issuer(request, self._config) + _RESOURCE_METADATA_PATH + MCP_PATH
            challenge = f'Bearer resource_metadata="{metadata}"' + (', error="invalid_token"' if token else "")
            refusal = JSONResponse(
                {"error": "invalid_token", "error_description": "A live Otari access token is required"},
                status_code=401,
                headers={"WWW-Authenticate": challenge},
            )
            await refusal(scope, receive, send)
            return
        scope["user"], scope["auth"] = AuthenticatedUser(access), AuthCredentials(access.scopes)
        await self._app(scope, receive, send)


def mount_mcp_server(app: FastAPI, config: GatewayConfig) -> McpEndpoint:
    """Serve ``/mcp`` and the metadata an MCP client discovers Otari's sign-in from, on ``app``."""
    endpoint = McpEndpoint(build_mcp_server(app), config)
    app.router.routes.append(Route(MCP_PATH, endpoint, methods=["GET", "POST", "DELETE"], include_in_schema=False))

    async def resource_metadata(request: Request) -> JSONResponse:
        issuer = oauth_app_issuer(request, config)
        return JSONResponse(
            {
                "resource": issuer + MCP_PATH,
                "authorization_servers": [issuer],
                "bearer_methods_supported": ["header"],
                "resource_name": "Otari",
            }
        )

    async def authorization_server_metadata(request: Request) -> JSONResponse:
        issuer = oauth_app_issuer(request, config)
        oauth = f"{issuer}{API_ROOT}/oauth/apps"
        return JSONResponse(
            {
                "issuer": issuer,
                "authorization_endpoint": f"{oauth}/authorize",
                "token_endpoint": f"{oauth}/token",
                "registration_endpoint": f"{oauth}/register",
                "revocation_endpoint": f"{oauth}/revoke",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none"],
                "revocation_endpoint_auth_methods_supported": ["none"],
                "authorization_response_iss_parameter_supported": True,
            }
        )

    # The resource's own path first (RFC 9728 section 3.1), and the bare path for clients that try it.
    for path in (_RESOURCE_METADATA_PATH + MCP_PATH, _RESOURCE_METADATA_PATH):
        app.add_api_route(path, resource_metadata, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        _AUTHORIZATION_SERVER_METADATA_PATH, authorization_server_metadata, methods=["GET"], include_in_schema=False
    )
    return endpoint
