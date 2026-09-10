"""Caller-orchestrated MCP: stored-server tool discovery and one-shot execution.

For an application that uses Otari as its routing layer and owns its own
authorization policy. It asks Otari for the tool definitions a stored MCP server
exposes, lets a model, a person, or a workflow propose a call, applies whatever
policy it has (a human approval prompt, an administrator rule, trusted
read-only auto-authorization), and then asks Otari to execute that one exact
call.

**Otari does not verify a human approval and does not claim to** (R-AUTH-4).
The calling application is the authorization boundary. What Otari enforces
independently is authentication, that the stored server is one the authenticated
workspace may reach, the stored tool allowlist, URL safety, and its own
execution bounds.

Nothing is held open across a caller's decision: no model stream, no MCP
session, no database session, no worker-local state. That is the whole reason
these are two separate requests rather than a pause inside the managed tool
loop.

The managed loop in ``services/mcp_loop.py`` remains a separate supported
integration mode, for a client that wants Otari to own the model/tool loop.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.exceptions import HTTPException as StarletteHTTPException

from gateway.api.deps import get_config, get_db_if_needed, verify_api_key_or_master_key
from gateway.api.routes._platform import (
    _extract_platform_user_token,
    _resolve_platform_mcp_server,
)

# ``GatewayConfig`` and ``AsyncSession`` are imported at runtime rather than
# under ``TYPE_CHECKING``: this module uses postponed annotations, and FastAPI
# resolves a route signature at import time to decide what each parameter is.
# Left as strings it cannot resolve, it reads both dependencies as query
# parameters and every request fails validation before the handler runs.
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import release_session
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import check_rate_limit
from gateway.repositories.users_repository import get_active_user

# The module as well as the names below, so the whole-request deadline is read
# off it at call time and stays tunable in the one place that owns the ceilings.
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import (
    ARGUMENTS_MAX_DEPTH,
    CODE_AUTHENTICATION_FAILED,
    CODE_CAPACITY_UNAVAILABLE,
    CODE_CONNECTION_FAILED,
    CODE_CREDENTIALS_UNAVAILABLE,
    CODE_DISCOVERY_CAPACITY_UNAVAILABLE,
    CODE_DISCOVERY_LIMIT_EXCEEDED,
    CODE_FORBIDDEN,
    CODE_INVALID_REQUEST,
    CODE_OUTCOME_UNKNOWN,
    CODE_PAYMENT_REQUIRED,
    CODE_RATE_LIMIT_EXCEEDED,
    CODE_RESOLUTION_FAILED,
    CODE_RESULT_TOO_LARGE,
    CODE_SERVER_CHANGED,
    CODE_SERVER_NOT_FOUND,
    CODE_SERVICE_UNAVAILABLE,
    CODE_TOOL_NOT_ALLOWED,
    CODE_UNSAFE_URL,
    DISCOVERY_MAX_TOOLS,
    REVISION_PATTERN,
    TOOL_NAME_MAX_LENGTH,
    ExecutionState,
    McpExecutionError,
    arguments_within_bounds,
    discover_stored_tools,
    discovery_response_exceeds_bound,
    execute_stored_tool,
)
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError
from gateway.services.tenancy.workspace_mcp_server_service import resolve_workspace_mcp_server
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url
from gateway.services.workspace_scope import resolve_workspace_id

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from fastapi import Response

    from gateway.models.mcp import ResolvedMcpServer

EXECUTE_ENDPOINT = "/v1/mcp/execute"
TOOLS_ENDPOINT = "/v1/mcp/servers/{mcp_server_id}/tools"

# The in-flight registry describes work by model, and these endpoints run no
# model. The label names what is actually being done instead, so an operator
# watching /v1/usage/in-flight sees a stateless MCP call rather than a blank.
EXECUTE_LABEL = "mcp.execute"
TOOLS_LABEL = "mcp.list_tools"

# One fixed safe message per error code (R-ERR-1). Nothing here varies with the
# request, the resolver answer, or the remote server: a message that varied
# would be the leak the whole error contract exists to prevent.
SAFE_DETAILS: dict[str, str] = {
    CODE_INVALID_REQUEST: "MCP request is invalid",
    CODE_AUTHENTICATION_FAILED: "Authentication failed",
    CODE_PAYMENT_REQUIRED: "Payment required",
    CODE_FORBIDDEN: "Request forbidden",
    CODE_RATE_LIMIT_EXCEEDED: "Rate limit exceeded",
    CODE_SERVICE_UNAVAILABLE: "MCP service is unavailable",
    CODE_SERVER_NOT_FOUND: "MCP server not found",
    CODE_SERVER_CHANGED: "MCP server configuration changed, rediscover its tools",
    CODE_TOOL_NOT_ALLOWED: "The requested tool is not allowed for this MCP server",
    CODE_UNSAFE_URL: "MCP server URL is unsafe",
    CODE_RESOLUTION_FAILED: "MCP server resolution failed",
    CODE_CREDENTIALS_UNAVAILABLE: "MCP server credentials are unavailable",
    CODE_DISCOVERY_LIMIT_EXCEEDED: "MCP discovery exceeded its limits",
    CODE_DISCOVERY_CAPACITY_UNAVAILABLE: "MCP discovery capacity is unavailable",
    CODE_CAPACITY_UNAVAILABLE: "MCP execution capacity is unavailable",
    CODE_CONNECTION_FAILED: "MCP server connection failed",
    CODE_OUTCOME_UNKNOWN: "MCP execution outcome is unknown",
    CODE_RESULT_TOO_LARGE: "MCP result exceeds the response limit",
}


# The two failures that are transient by construction: a per-process slot did
# not free up inside the admission deadline. Everything else this contract
# returns is a decision, a bound, or an outcome Otari cannot know, and none of
# those get better by being sent again (R-ERR-4). The one other status carrying
# this header is the 429, which keeps the limiter's own value.
_RETRYABLE_CAPACITY_CODES = frozenset({CODE_CAPACITY_UNAVAILABLE, CODE_DISCOVERY_CAPACITY_UNAVAILABLE})
RETRY_AFTER_ONE = {"Retry-After": "1"}


class McpErrorBody(BaseModel):
    """The one error shape both stored-server endpoints return (R-ERR-1)."""

    detail: str
    code: str
    execution_state: ExecutionState
    request_id: str


class _McpRoute(APIRoute):
    """Gives every route on this router the shared error contract and request id.

    A route class rather than an application exception handler, for two reasons.
    The 422 has to be normalized too, and a ``RequestValidationError`` is raised
    inside the route handler where only this wrapper can see it; and the
    contract is deliberately local to ``/api/v1/mcp``, so nothing else on the
    deployment picks up an error shape it never published.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            request_id = f"req_{uuid.uuid4().hex}"
            request.state.otari_request_id = request_id
            try:
                response = await original(request)
            except RequestValidationError:
                # Normalized rather than forwarded: FastAPI's body echoes the
                # rejected value, which here is the caller's own arguments.
                return _error_response(CODE_INVALID_REQUEST, ExecutionState.NOT_STARTED, 422, request_id)
            except McpExecutionError as exc:
                return _error_response(
                    exc.code,
                    exc.execution_state,
                    exc.status_code,
                    request_id,
                    headers=RETRY_AFTER_ONE if exc.code in _RETRYABLE_CAPACITY_CODES else None,
                )
            except StarletteHTTPException as exc:
                code, execution_state, status_code = _classify(exc)
                retry_after = (exc.headers or {}).get("Retry-After")
                return _error_response(
                    code,
                    execution_state,
                    status_code,
                    request_id,
                    headers={"Retry-After": retry_after} if retry_after else None,
                )
            response.headers["X-Otari-Request-ID"] = request_id
            return response

        return handler


def _error_response(
    code: str,
    execution_state: ExecutionState,
    status_code: int,
    request_id: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = McpErrorBody(
        detail=SAFE_DETAILS.get(code, "MCP request failed"),
        code=code,
        execution_state=execution_state,
        request_id=request_id,
    )
    response_headers = {"X-Otari-Request-ID": request_id}
    if headers:
        response_headers.update(headers)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"), headers=response_headers)


def _classify(exc: StarletteHTTPException) -> tuple[str, ExecutionState, int]:
    """Map an authentication or platform refusal onto this contract's enums.

    Every one of these is raised before dispatch, so all of them are
    ``not_started``. The platform's own detail is dropped rather than forwarded:
    it may describe a workspace, a plan, or a stored server, and R-ERR-1 lets
    nothing platform-side through.
    """
    if exc.status_code in {400, 422}:
        return CODE_INVALID_REQUEST, ExecutionState.NOT_STARTED, 422
    if exc.status_code == 401:
        return CODE_AUTHENTICATION_FAILED, ExecutionState.NOT_STARTED, 401
    if exc.status_code == 402:
        return CODE_PAYMENT_REQUIRED, ExecutionState.NOT_STARTED, 402
    if exc.status_code == 403:
        return CODE_FORBIDDEN, ExecutionState.NOT_STARTED, 403
    if exc.status_code == 404:
        return CODE_SERVER_NOT_FOUND, ExecutionState.NOT_STARTED, 404
    if exc.status_code == 429:
        return CODE_RATE_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 429
    if exc.status_code == 503:
        return CODE_SERVICE_UNAVAILABLE, ExecutionState.NOT_STARTED, 503
    return CODE_RESOLUTION_FAILED, ExecutionState.NOT_STARTED, 502


router = APIRouter(prefix="/mcp", tags=["mcp"], route_class=_McpRoute)


class McpExecuteRequest(BaseModel):
    """One stored server, and the exact call the application authorized.

    No inline server fields (R-REQ-4): a caller registers a remote MCP server
    through the control plane once and refers to it by id afterwards, which
    keeps URLs, credentials, revocation and allowlist policy on Otari's side of
    the boundary instead of in every request.

    Extras are forbidden rather than ignored, so a caller still sending the old
    inline ``server`` block is told its configuration was not used instead of
    watching Otari quietly execute against a different server than the one it
    named.
    """

    model_config = ConfigDict(extra="forbid")

    mcp_server_id: uuid.UUID = Field(description="The stored MCP server to execute against.")
    tool_name: str = Field(
        min_length=1,
        max_length=TOOL_NAME_MAX_LENGTH,
        description="The remote MCP tool name the caller authorized for this one execution.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="The exact caller-authorized JSON-object arguments.",
    )
    server_revision: str = Field(
        pattern=REVISION_PATTERN,
        description=(
            "The stored-server revision returned by tool discovery. Required, and compared "
            "against the current one so a configuration change since the caller authorized "
            "this call is refused rather than executed. Not an approval credential."
        ),
    )
    client_execution_id: uuid.UUID = Field(
        description=(
            "A caller-generated UUID, for correlation only. It is not proof of approval and "
            "not an idempotency key: repeating a request with the same value may execute the "
            "tool again, so this request must never be retried automatically."
        ),
    )

    @field_validator("arguments")
    @classmethod
    def _bounded_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not arguments_within_bounds(value):
            raise ValueError(f"arguments exceed the size or depth limit (depth {ARGUMENTS_MAX_DEPTH})")
        return value


@dataclass(frozen=True, slots=True)
class _Principal:
    """Who the request authenticated as, in whichever mode answered it."""

    user_token: str | None
    workspace_id: uuid.UUID | None


async def _authenticate(
    raw_request: Request,
    db: AsyncSession | None,
    config: GatewayConfig,
) -> _Principal:
    """Authenticate, and rate-limit the principal before any outbound access.

    Hybrid mode carries the user token through to the platform resolver, which
    is both the authorization and the rate-limit boundary there (R-ADM-2); Otari
    preserves whatever 429 it returns. Standalone authenticates the key and
    charges the request to that key's own bucket here.

    A master-key request is refused. It holds no workspace of its own, so it can
    only ever resolve a stored server through the deployment's default
    workspace, which would let operator credentials reach one tenant's
    configured servers. No stored server is accessible to it, and that is what
    the 404 says.

    A blocked user is refused too, and has to be refused here. ``users.blocked``
    is read in ``budget_service.reserve_budget``, which is the only place any
    request plane consults it, and these routes never reach it because they
    reserve nothing. Without this, blocking someone would stop their completions
    and leave them driving mutating MCP tools through the gateway.
    """
    if config.is_hybrid_mode:
        return _Principal(user_token=_extract_platform_user_token(raw_request), workspace_id=None)

    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        )
    api_key, _is_master = await verify_api_key_or_master_key(raw_request, db, config)
    if api_key is None:
        raise McpExecutionError(CODE_SERVER_NOT_FOUND, ExecutionState.NOT_STARTED, 404)
    # The key's own bucket, falling back to the key when it names no user, so
    # every user-less key does not share one bucket keyed on ``"None"``.
    check_rate_limit(raw_request, api_key.user_id or api_key.id)
    await _refuse_blocked_user(db, api_key)
    return _Principal(user_token=None, workspace_id=await resolve_workspace_id(db, api_key))


async def _refuse_blocked_user(db: AsyncSession, api_key: APIKey) -> None:
    """Refuse a key whose user an operator has blocked.

    Reported as an authentication failure rather than a code of its own: the
    caller's credential no longer authorizes anything, which is what the caller
    needs to know, and the alternative would tell an unauthenticated probe that
    a given key names a real but blocked user. Why it was refused goes to the
    log instead.

    A key with no user has no block to check, which is the bootstrap and
    convenience path (``users_repository.DEFAULT_USER_ID``).
    """
    if api_key.user_id is None:
        return
    user = await get_active_user(db, api_key.user_id)
    if user is not None and user.blocked:
        logger.warning("Stateless MCP request refused reason=user_blocked api_key_id=%s", api_key.id)
        raise McpExecutionError(CODE_AUTHENTICATION_FAILED, ExecutionState.NOT_STARTED, 401)


async def _resolve_server(
    principal: _Principal,
    db: AsyncSession | None,
    config: GatewayConfig,
    mcp_server_id: uuid.UUID,
) -> ResolvedMcpServer:
    """Resolve the stored server, applying the outcome ladder both modes share (R-RES-1).

    No MCP network access happens here or in anything it raises, so a refusal on
    these grounds never reaches the remote server.
    """
    if config.is_hybrid_mode:
        server = await _resolve_platform_mcp_server(config, principal.user_token or "", mcp_server_id)
    else:
        if db is None or principal.workspace_id is None:  # pragma: no cover - _authenticate settles both
            raise McpExecutionError(CODE_RESOLUTION_FAILED, ExecutionState.NOT_STARTED, 502)
        try:
            resolved = await resolve_workspace_mcp_server(
                db,
                workspace_id=principal.workspace_id,
                server_id=mcp_server_id,
            )
        except (SecretDecryptionError, SecretBoxUnavailableError):
            # Connecting without the credential the workspace configured would
            # send an unauthenticated request to a server that expects one.
            logger.warning("Stateless MCP stored credential unavailable server_id=%s", mcp_server_id)
            raise McpExecutionError(CODE_CREDENTIALS_UNAVAILABLE, ExecutionState.NOT_STARTED, 500) from None
        if resolved is None:
            raise McpExecutionError(CODE_SERVER_NOT_FOUND, ExecutionState.NOT_STARTED, 404)
        server = resolved

    if not server.enabled:
        # Indistinguishable from an id naming no server, on purpose: a caller
        # learns that it cannot reach this server, not why.
        raise McpExecutionError(CODE_SERVER_NOT_FOUND, ExecutionState.NOT_STARTED, 404)
    return server


def _require_allowed(server: ResolvedMcpServer, tool_name: str) -> None:
    """Apply the stored allowlist to one tool name (R-RES-4).

    ``None`` admits every live tool, the meaning the column already carries in
    the managed loop; an explicit empty list is a deny-all, so it refuses here
    as well.
    """
    if server.allowed_tools is not None and tool_name not in server.allowed_tools:
        raise McpExecutionError(CODE_TOOL_NOT_ALLOWED, ExecutionState.NOT_STARTED, 403)


async def _require_safe_url(server: ResolvedMcpServer) -> None:
    """Run the existing MCP SSRF policy over the resolved URL (R-TRANSPORT-3)."""
    try:
        await validate_mcp_url(server.url, has_authorization_token=bool(server.authorization_token))
    except UnsafeURLError:
        raise McpExecutionError(CODE_UNSAFE_URL, ExecutionState.NOT_STARTED, 400) from None


class McpToolDefinition(BaseModel):
    """One live tool a caller-orchestrated application may expose to its model.

    ``annotations`` is the remote server's own metadata, passed through as
    untrusted data. Otari never turns ``readOnlyHint`` into an authorization
    decision (R-RISK-1); each application owns its risk policy, and a server
    cannot waive an application's approval gate by labeling itself read-only.
    """

    name: str = Field(description=f"The remote MCP tool name to send back to {API_ROOT}/mcp/execute.")
    description: str | None = Field(default=None, description="The server's own description, untrusted.")
    input_schema: dict[str, Any] = Field(description="The tool's MCP inputSchema, unmodified.")
    annotations: dict[str, Any] | None = Field(
        default=None,
        description="The server's MCP annotations, untrusted metadata rather than policy.",
    )


class McpToolWarning(BaseModel):
    """One tool that was omitted, and the code that omitted it (R-SCHEMA-3)."""

    tool_name: str
    code: str


class McpToolsResponse(BaseModel):
    """The authorized catalog for one stored server.

    Carries no server URL, no credential, and no allowlist entry that the live
    catalog did not return (R-DISC-2). ``server_revision`` is what an
    application persists with a proposed call and sends back to
    ``/api/v1/mcp/execute``, so a stored-configuration change between the two is
    refused rather than executed.
    """

    server_id: uuid.UUID
    server_revision: str = Field(
        description=(
            "An opaque revision of the stored server's URL, credential, enabled state and "
            "allowlist. It detects Otari-side and platform-side configuration changes only: "
            "a remote server that changes its own catalog or a tool's behavior behind an "
            "unchanged URL will not move it."
        ),
    )
    tools: list[McpToolDefinition] = Field(max_length=DISCOVERY_MAX_TOOLS)
    warnings: list[McpToolWarning]


@router.get(
    "/servers/{mcp_server_id}/tools",
    response_model=McpToolsResponse,
    responses={
        400: {"model": McpErrorBody},
        401: {"model": McpErrorBody},
        402: {"model": McpErrorBody},
        403: {"model": McpErrorBody},
        404: {"model": McpErrorBody},
        422: {"model": McpErrorBody},
        429: {"model": McpErrorBody},
        500: {"model": McpErrorBody},
        502: {"model": McpErrorBody},
        503: {"model": McpErrorBody},
    },
)
async def list_mcp_tools(
    raw_request: Request,
    mcp_server_id: uuid.UUID,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> McpToolsResponse:
    """List the tools a stored MCP server exposes to the authenticated workspace.

    Call this once per server when preparing a model or workflow run, and reuse
    the answer for every tool from that server for the length of the run: there
    is no cross-run cache in this version, so a later run rediscovers and
    staleness stays bounded without any invalidation state to keep.

    The response is the whole authorized catalog or an error. It is never
    partial, because a caller would read a short catalog as the complete input
    to its own authorization and risk policy. The single exception is
    ``warnings``, which names a tool whose descriptor Otari could not carry.

    A tool the server removes after discovery may still be proposed from the
    run's snapshot; execution then returns the remote server's own typed error.
    """
    started = time.monotonic()
    try:
        async with asyncio.timeout(mcp_stateless.DISCOVERY_TOTAL_TIMEOUT_S):
            principal = await _authenticate(raw_request, db, config)
            server = await _resolve_server(principal, db, config, mcp_server_id)

            if server.allowed_tools == []:
                # An operator's explicit deny-all is a complete answer already,
                # so this opens no connection at all (R-RES-4).
                response = McpToolsResponse(
                    server_id=server.id,
                    server_revision=server.revision,
                    tools=[],
                    warnings=[],
                )
                if discovery_response_exceeds_bound(response):  # pragma: no cover - fixed-size response
                    raise McpExecutionError(CODE_DISCOVERY_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 502)
                return response

            await release_session(db)
            await _require_safe_url(server)
            track_request(raw_request, endpoint=TOOLS_ENDPOINT, model=TOOLS_LABEL)
            catalog = await discover_stored_tools(server)

            response = McpToolsResponse(
                server_id=server.id,
                server_revision=server.revision,
                tools=[
                    McpToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.inputSchema,
                        annotations=tool.annotations.model_dump(mode="json", exclude_none=True)
                        if tool.annotations is not None
                        else None,
                    )
                    for tool in catalog.tools
                ],
                warnings=[McpToolWarning(tool_name=name, code=code) for name, code in catalog.warnings],
            )
            if discovery_response_exceeds_bound(response):
                raise McpExecutionError(CODE_DISCOVERY_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 502)
    except TimeoutError:
        exc = McpExecutionError(CODE_DISCOVERY_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 502)
        logger.info(
            "Stateless MCP discovery request_id=%s server_id=%s outcome=%s duration_ms=%.2f",
            getattr(raw_request.state, "otari_request_id", "-"),
            mcp_server_id,
            exc.code,
            (time.monotonic() - started) * 1000,
        )
        raise exc from None
    except McpExecutionError as exc:
        logger.info(
            "Stateless MCP discovery request_id=%s server_id=%s outcome=%s duration_ms=%.2f",
            getattr(raw_request.state, "otari_request_id", "-"),
            mcp_server_id,
            exc.code,
            (time.monotonic() - started) * 1000,
        )
        raise

    logger.info(
        "Stateless MCP discovery request_id=%s server_id=%s outcome=ok tools=%d omitted=%d duration_ms=%.2f",
        getattr(raw_request.state, "otari_request_id", "-"),
        mcp_server_id,
        len(catalog.tools),
        len(catalog.warnings),
        (time.monotonic() - started) * 1000,
    )
    return response


@router.post(
    "/execute",
    response_model=CallToolResult,
    response_model_exclude_none=True,
    responses={
        400: {"model": McpErrorBody},
        401: {"model": McpErrorBody},
        402: {"model": McpErrorBody},
        403: {"model": McpErrorBody},
        404: {"model": McpErrorBody},
        409: {"model": McpErrorBody},
        422: {"model": McpErrorBody},
        429: {"model": McpErrorBody},
        500: {"model": McpErrorBody},
        502: {"model": McpErrorBody},
        503: {"model": McpErrorBody},
        504: {"model": McpErrorBody},
    },
)
async def execute_mcp_tool(
    raw_request: Request,
    request: McpExecuteRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CallToolResult:
    """Execute one caller-authorized tool call against a stored MCP server.

    The calling application owns any user approval, argument editing,
    cancellation and action history; Otari executes exactly the tool name and
    arguments it is given, once, and returns the remote server's native result.
    A result with ``isError: true`` is a definitive outcome and comes back as an
    HTTP 200.

    **This request must never be retried automatically.** ``client_execution_id``
    is correlation, not idempotency: once the call has been dispatched Otari
    cannot know whether the tool ran, and an ``outcome_unknown`` response means
    exactly that. Proxies, service meshes and SDKs on this path have to disable
    retries for it, including on connection resets and 5xx responses.
    """
    started = time.monotonic()
    dispatched = False
    timings: dict[str, float] = {}

    def mark_dispatched() -> None:
        nonlocal dispatched
        dispatched = True

    try:
        async with asyncio.timeout(mcp_stateless.EXECUTION_TOTAL_TIMEOUT_S):
            principal = await _authenticate(raw_request, db, config)
            server = await _resolve_server(principal, db, config, request.mcp_server_id)
            _require_allowed(server, request.tool_name)
            if request.server_revision != server.revision:
                # In memory, over the resolution both modes already needed, so this
                # costs no database, platform or MCP round trip (R-RES-2).
                raise McpExecutionError(CODE_SERVER_CHANGED, ExecutionState.NOT_STARTED, 409)

            # Before DNS, before the concurrency wait, and before any MCP network I/O:
            # a pooled connection must not be pinned for the length of a remote call.
            await release_session(db)
            await _require_safe_url(server)
            timings["resolve_ms"] = (time.monotonic() - started) * 1000
            track_request(raw_request, endpoint=EXECUTE_ENDPOINT, model=EXECUTE_LABEL)

            result = await execute_stored_tool(
                server,
                request.tool_name,
                request.arguments,
                on_dispatch=mark_dispatched,
                timings=timings,
            )
    except TimeoutError:
        # The total deadline can only be reached before dispatch here: past it,
        # the call's own deadline is the shorter of the two and classifies the
        # failure itself.
        exc = McpExecutionError(
            CODE_OUTCOME_UNKNOWN if dispatched else CODE_CONNECTION_FAILED,
            ExecutionState.OUTCOME_UNKNOWN if dispatched else ExecutionState.NOT_STARTED,
            504 if dispatched else 502,
        )
        _log_outcome(raw_request, request, exc.execution_state, exc.code, started, timings)
        raise exc from None
    except McpExecutionError as exc:
        _log_outcome(raw_request, request, exc.execution_state, exc.code, started, timings)
        raise

    _log_outcome(raw_request, request, ExecutionState.COMPLETED, "ok", started, timings)
    return result


def _log_outcome(
    raw_request: Request,
    request: McpExecuteRequest,
    execution_state: ExecutionState,
    outcome: str,
    started: float,
    timings: dict[str, float],
) -> None:
    """Record timings, outcome and correlation ids, and nothing else (R-OBS-1, R-OBS-2).

    Both ids are opaque: the caller's ``client_execution_id`` is a canonical
    UUID validated at parse time, and the Otari request id is generated here.
    Neither carries user content, approval content, arguments or credentials.
    """
    logger.info(
        "Stateless MCP execution request_id=%s client_execution_id=%s server_id=%s "
        "outcome=%s execution_state=%s duration_ms=%.2f phases=%s",
        getattr(raw_request.state, "otari_request_id", "-"),
        request.client_execution_id,
        request.mcp_server_id,
        outcome,
        execution_state.value,
        (time.monotonic() - started) * 1000,
        # Durations and one byte count. The tool name, the arguments and the
        # result are deliberately absent from every field above and here.
        " ".join(f"{phase}={value:.2f}" for phase, value in sorted(timings.items())),
    )
