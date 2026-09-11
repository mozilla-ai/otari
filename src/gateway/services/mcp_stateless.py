"""Stateless, single-server MCP discovery and execution.

The primitive behind the stored-server endpoints in ``api/routes/mcp.py``, and
deliberately not :class:`~gateway.services.mcp_client.MCPClientPool`. The pool
exists to hold several sessions open across a model's whole tool loop and to
publish the union of their tools; these endpoints connect to exactly one server,
do exactly one thing, and close. An application pauses for a person between
discovery and execution, so nothing here may be kept alive across that wait.

The bounds are the point. A caller-orchestrated application hands Otari a tool
name and arguments a model proposed, and the remote server is untrusted: its
catalog can be unbounded, its schemas hostile, its response arbitrarily large.
Every ceiling in the design's limits table is enforced here or in the route, and
each is named in the constant that carries it.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from gateway.log_config import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, MutableMapping

    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPTool

    from gateway.models.mcp import ResolvedMcpServer


# --------------------------------------------------------------------------- #
# Limits
#
# Every ceiling the design's limits table names, in one block so a deployment
# tuning one can see what it sits beside. They are gateway-owned, not
# caller-controlled request fields.
# --------------------------------------------------------------------------- #

# Request body (L-TOOLNAME, L-ARGS). Enforced at parse time by the request
# model, so a caller's own text is bounded before resolution, logging or
# telemetry sees any of it (R-REQ-3).
TOOL_NAME_MAX_LENGTH = 256
ARGUMENTS_MAX_BYTES = 256 * 1024
ARGUMENTS_MAX_DEPTH = 32

# The wire format of a stored-server revision (R-DISC-3).
REVISION_PATTERN = r"^[A-Za-z0-9._:\-]{1,128}$"

# Per tool (L-DISC-DESC, L-DISC-SCHEMA, L-DISC-ANNOT). A breach of one of these
# omits the single tool and reports a warning (R-SCHEMA-3), rather than failing
# the whole catalog: one hostile descriptor must not take a server's other
# tools away from an application.
TOOL_DESCRIPTION_MAX_BYTES = 4 * 1024
SCHEMA_MAX_BYTES = 64 * 1024
SCHEMA_MAX_DEPTH = 32
TOOL_ANNOTATIONS_MAX_BYTES = 16 * 1024

# Not in the design's table, and bounded for the same reason the three above
# are: an object whose size and depth both pass can still hold hundreds of
# thousands of one-character properties, and everything downstream of discovery
# (an application's validator, a model's context) pays for each.
SCHEMA_MAX_PROPERTIES = 1000

# Whole catalog (L-DISC-PAGES, L-DISC-EXAMINED, L-DISC-TOOLS,
# L-DISC-RESPONSE). A breach of one of these refuses the entire response
# (R-DISC-5). Returning what was collected so far would hand an application a
# catalog it would read as complete and then base an authorization decision
# on.
DISCOVERY_MAX_PAGES = 20
DISCOVERY_MAX_EXAMINED = 1000
DISCOVERY_MAX_TOOLS = 200
DISCOVERY_RESPONSE_MAX_BYTES = 1024 * 1024

# Transport (L-TRANSPORT-BYTES, L-RESULT-BYTES). Compressed responses are
# refused before reading, and uncompressed response streams are counted as
# they are consumed, including chunked JSON and SSE (R-TRANSPORT-1).
TRANSPORT_MAX_BYTES = 1024 * 1024
RESULT_MAX_BYTES = 1024 * 1024

# Phase deadlines (L-CONNECT, L-CALL, L-CLEANUP, L-DISC-TOTAL, L-TOTAL). They
# are distinct on purpose: the call deadline starts at dispatch and excludes the
# capacity wait, and the total ones cap the whole operation so repeated slow
# phases cannot exceed the intended request lifetime.
CONNECT_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 30.0
CLEANUP_TIMEOUT_S = 2.0
DISCOVERY_TOTAL_TIMEOUT_S = 15.0
EXECUTION_TOTAL_TIMEOUT_S = 45.0

# Admission (L-DISC-CONC, L-DISC-ADMIT, L-EXEC-CONC, L-EXEC-ADMIT).
DISCOVERY_CONCURRENCY = 4
DISCOVERY_ADMISSION_TIMEOUT_S = 5.0
EXECUTION_CONCURRENCY = 8
EXECUTION_ADMISSION_TIMEOUT_S = 5.0


# --------------------------------------------------------------------------- #
# Stable wire codes
# --------------------------------------------------------------------------- #

# Per-tool omission codes, reported in a discovery response's ``warnings``.
WARN_NAME_UNSUPPORTED = "mcp_tool_name_unsupported"
WARN_SCHEMA_UNSUPPORTED = "mcp_tool_schema_unsupported"
WARN_DESCRIPTION_TOO_LARGE = "mcp_tool_description_too_large"
WARN_ANNOTATIONS_TOO_LARGE = "mcp_tool_annotations_too_large"

# Error codes. The route owns the one safe ``detail`` string each maps to; these
# are the stable enum a caller branches on.
CODE_CONNECTION_FAILED = "mcp_connection_failed"
CODE_OUTCOME_UNKNOWN = "mcp_outcome_unknown"
CODE_RESULT_TOO_LARGE = "mcp_result_too_large"
CODE_CAPACITY_UNAVAILABLE = "mcp_capacity_unavailable"
CODE_DISCOVERY_LIMIT_EXCEEDED = "mcp_discovery_limit_exceeded"
CODE_DISCOVERY_CAPACITY_UNAVAILABLE = "mcp_discovery_capacity_unavailable"
CODE_RESOLUTION_FAILED = "mcp_resolution_failed"
CODE_SERVER_NOT_FOUND = "mcp_server_not_found"
CODE_SERVER_CHANGED = "mcp_server_changed"
CODE_TOOL_NOT_ALLOWED = "mcp_tool_not_allowed"
CODE_UNSAFE_URL = "unsafe_mcp_url"
CODE_CREDENTIALS_UNAVAILABLE = "mcp_credentials_unavailable"
CODE_AUTHENTICATION_FAILED = "authentication_failed"
CODE_PAYMENT_REQUIRED = "payment_required"
CODE_FORBIDDEN = "forbidden"
CODE_RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
CODE_SERVICE_UNAVAILABLE = "service_unavailable"
CODE_INVALID_REQUEST = "invalid_request"


# --------------------------------------------------------------------------- #
# Failure types
# --------------------------------------------------------------------------- #


T = TypeVar("T")


class McpDiscoveryRefused(Exception):
    """A whole-catalog discovery failure (R-DISC-5).

    Raised for a pagination validation failure or a discovery ceiling breach,
    both of which refuse the entire response. Carries no message on purpose:
    every caller answers it with one fixed safe detail, and the only detail
    worth attaching would describe the remote server's catalog.
    """


class TransportResponseTooLarge(Exception):
    """A transport response that exceeded the byte ceiling."""


class McpCapacityUnavailable(Exception):
    """A concurrency slot did not free up inside the admission deadline."""


class ExecutionState(StrEnum):
    """Whether the remote tool may have run (R-ERR-2).

    A retry-safety classification, not a description of how the HTTP request
    went. ``NOT_STARTED`` is Otari saying it knows the tool did not run;
    ``OUTCOME_UNKNOWN`` is Otari saying it cannot know, which is the only honest
    answer once the transport has begun writing ``tools/call``.
    """

    NOT_STARTED = "not_started"
    OUTCOME_UNKNOWN = "outcome_unknown"
    COMPLETED = "completed"


class McpExecutionError(Exception):
    """A stateless MCP failure, already classified for the wire.

    Carries the status, the stable error code, and the execution state rather
    than a message: the detail a caller sees is fixed per category (R-ERR-1),
    and anything the remote server or the exception said about why is exactly
    what must not travel.
    """

    def __init__(self, code: str, execution_state: ExecutionState, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.execution_state = execution_state
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# Measuring untrusted JSON
# --------------------------------------------------------------------------- #


# Nesting depth followed when naming a grouped failure. anyio nests one level,
# so this only exists so a pathological group cannot recurse without end.
_FAILURE_GROUP_MAX_DEPTH = 4


def failure_class(exc: BaseException, _depth: int = 0) -> str:
    """Name a failure by its leaf types, carrying nothing the exception said.

    The MCP SDK yields inside ``anyio.create_task_group()``, so a transport
    failure usually arrives wrapped: ``ExceptionGroup('unhandled errors in a
    TaskGroup', [ConnectError(...)])``. Logging the outer type alone records
    ``ExceptionGroup`` for nearly every real failure, which is no diagnostic at
    all, so the leaves are named instead.

    Types only, never a message: an exception's message can hold the server URL,
    the credential, or the caller's arguments (R-OBS-2).
    """
    if isinstance(exc, BaseExceptionGroup) and _depth < _FAILURE_GROUP_MAX_DEPTH:
        leaves = sorted({failure_class(leaf, _depth + 1) for leaf in exc.exceptions})
        return "+".join(leaves) if leaves else type(exc).__name__
    return type(exc).__name__


def _oversized(value: Any, ceiling: int) -> bool:
    """Whether ``value`` exceeds ``ceiling`` once encoded.

    Encoded rather than measured in place, because that is the size the value
    occupies in the discovery response an application receives, and a value that
    will not encode at all is oversized by the only definition that matters
    here: Otari cannot carry it.
    """
    if value is None:
        return False
    try:
        encoded = _encode(value)
    except (TypeError, ValueError):
        return True
    return len(encoded) > ceiling


def _encode(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), default=_as_jsonable).encode()


def _as_jsonable(value: Any) -> Any:
    """Render a Pydantic model the way the response would, or refuse to size it."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    raise TypeError(type(value).__name__)


def _walk(schema: dict[str, Any]) -> tuple[int, int, bool]:
    """Measure nesting depth and property count, and spot any non-local ``$ref``.

    One traversal for all three, iteratively rather than recursively: the depth
    ceiling is what makes a deep schema safe, and a recursive walk would have to
    survive the schema long enough to enforce it.

    A ``$ref`` is local only when it is a fragment of this same schema, so it
    resolves without a request. Anything else, absolute, protocol-relative, or a
    sibling file, is refused rather than fetched.
    """
    max_depth = 0
    properties = 0
    external_ref = False
    stack: list[tuple[Any, int]] = [(schema, 1)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and not ref.startswith("#"):
                external_ref = True
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    properties += len(value)
                stack.append((value, depth + 1))
        elif isinstance(node, list):
            stack.extend((item, depth + 1) for item in node)
    return max_depth, properties, external_ref


# --------------------------------------------------------------------------- #
# Screening one live tool descriptor, and one call's arguments
# --------------------------------------------------------------------------- #


def screen_tool(tool: MCPTool) -> str | None:
    """Return the warning code that omits ``tool``, or ``None`` to admit it.

    Structural and size checks only (R-SCHEMA-1). The schema is untrusted data
    rather than something Otari executes, so an unrecognized dialect or keyword
    is carried through untouched; what is refused is a descriptor Otari cannot
    safely return or execute, or a schema that would make it fetch another
    schema over the network to be understood at all.
    """
    if not tool.name or len(tool.name) > TOOL_NAME_MAX_LENGTH:
        return WARN_NAME_UNSUPPORTED
    if _oversized(tool.description, TOOL_DESCRIPTION_MAX_BYTES):
        return WARN_DESCRIPTION_TOO_LARGE
    if tool.annotations is not None and _oversized(tool.annotations, TOOL_ANNOTATIONS_MAX_BYTES):
        return WARN_ANNOTATIONS_TOO_LARGE
    if not _schema_is_supported(tool.inputSchema):
        return WARN_SCHEMA_UNSUPPORTED
    return None


def _schema_is_supported(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    if _oversized(schema, SCHEMA_MAX_BYTES):
        return False
    depth, properties, external_ref = _walk(schema)
    return depth <= SCHEMA_MAX_DEPTH and properties <= SCHEMA_MAX_PROPERTIES and not external_ref


def arguments_within_bounds(arguments: dict[str, Any]) -> bool:
    """Whether one call's arguments are small and shallow enough to forward."""
    if _oversized(arguments, ARGUMENTS_MAX_BYTES):
        return False
    depth, _, _ = _walk(arguments)
    return depth <= ARGUMENTS_MAX_DEPTH


# --------------------------------------------------------------------------- #
# Transport and admission
# --------------------------------------------------------------------------- #


async def enforce_content_length(response: httpx.Response) -> None:
    """Install the decoded response ceiling before the body is read.

    Registered as an httpx response event hook, which runs with the headers
    available and the body still unread. Compression is refused because one
    encoded chunk can expand beyond the ceiling before a decoded-byte counter
    can inspect it. Identity-encoded streams are wrapped so missing or dishonest
    ``Content-Length`` values cannot bypass the limit.
    """
    content_encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise TransportResponseTooLarge

    declared = response.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            pass
        else:
            if length > TRANSPORT_MAX_BYTES:
                raise TransportResponseTooLarge

    if not isinstance(response.stream, httpx.AsyncByteStream):  # pragma: no cover - async client invariant
        raise TypeError("Expected an asynchronous HTTP response stream")
    response.stream = _BoundedResponseStream(response.stream, TRANSPORT_MAX_BYTES)


class _BoundedResponseStream(httpx.AsyncByteStream):
    """Count response bytes while preserving streaming and close behavior."""

    def __init__(self, stream: httpx.AsyncByteStream, ceiling: int) -> None:
        self._stream = stream
        self._ceiling = ceiling

    async def __aiter__(self) -> AsyncIterator[bytes]:
        consumed = 0
        async for chunk in self._stream:
            consumed += len(chunk)
            if consumed > self._ceiling:
                raise TransportResponseTooLarge
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


def build_http_client_factory() -> Callable[..., httpx.AsyncClient]:
    """An MCP HTTP client factory with redirects disabled and a size pre-check.

    The SDK's own factory sets ``follow_redirects=True``, which on a
    credentialed request means httpx would replay the ``Authorization`` header,
    and the request body, at whatever host the remote server names. Redirects
    are excluded from the first version entirely (R-TRANSPORT-2): a later one
    may follow them by validating each destination before anything is sent.
    """

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        client_headers = dict(headers or {})
        client_headers["Accept-Encoding"] = "identity"
        kwargs: dict[str, Any] = {
            "follow_redirects": False,
            "timeout": timeout if timeout is not None else httpx.Timeout(CONNECT_TIMEOUT_S, read=CALL_TIMEOUT_S),
            "event_hooks": {"response": [enforce_content_length]},
            "headers": client_headers,
        }
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    return factory


def result_exceeds_bound(result: CallToolResult) -> bool:
    """Whether a decoded MCP result is too large to return (L-RESULT-BYTES).

    A breach here is ``outcome_unknown`` rather than a clean failure: the tool
    ran and may have mutated something, and Otari simply cannot carry what came
    back.
    """
    return _oversized(result, RESULT_MAX_BYTES)


def discovery_response_exceeds_bound(response: Any) -> bool:
    """Whether the complete serialized discovery response exceeds its ceiling."""
    try:
        value = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        return len(_encode(value)) > DISCOVERY_RESPONSE_MAX_BYTES
    except (TypeError, ValueError):
        return True


class ConcurrencyGate:
    """A per-process slot ceiling with a bounded wait for admission.

    Discovery and execution hold separate gates (R-ADM-1). Discovery talks to a
    server that has not been authorized for anything yet and can be slow or
    hostile; execution is running a call a person or a policy already approved.
    One ceiling shared between them would let the first starve the second.
    """

    def __init__(self, *, limit: int, admission_timeout_s: float) -> None:
        self.limit = limit
        self._admission_timeout_s = admission_timeout_s
        self._semaphore = asyncio.Semaphore(limit)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one slot, or raise :class:`McpCapacityUnavailable`.

        Raises:
            McpCapacityUnavailable: the deadline passed with no slot free.
        """
        try:
            async with asyncio.timeout(self._admission_timeout_s):
                await self._semaphore.acquire()
        except TimeoutError:
            raise McpCapacityUnavailable from None
        try:
            yield
        finally:
            self._semaphore.release()


DISCOVERY_GATE = ConcurrencyGate(limit=DISCOVERY_CONCURRENCY, admission_timeout_s=DISCOVERY_ADMISSION_TIMEOUT_S)
EXECUTION_GATE = ConcurrencyGate(limit=EXECUTION_CONCURRENCY, admission_timeout_s=EXECUTION_ADMISSION_TIMEOUT_S)


async def _run_in_owner_task(operation: Coroutine[Any, Any, T]) -> T:
    """Run an anyio-backed context entirely in one task and await its cleanup."""
    owner = asyncio.create_task(operation)
    try:
        return await asyncio.shield(owner)
    except asyncio.CancelledError:
        owner.cancel()
        return await owner


@asynccontextmanager
async def open_session(server: ResolvedMcpServer) -> AsyncIterator[ClientSession]:
    """Open one size-bounded, redirect-disabled MCP session on a stored server.

    Connection and initialization share one deadline (L-CONNECT), so a server
    that accepts a socket and then never answers ``initialize`` cannot hold a
    slot for the length of the call deadline instead.
    """
    headers = {"Authorization": f"Bearer {server.authorization_token}"} if server.authorization_token else None
    async with AsyncExitStack() as stack:
        async with asyncio.timeout(CONNECT_TIMEOUT_S):
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(
                    server.url,
                    headers=headers,
                    timeout=CONNECT_TIMEOUT_S,
                    sse_read_timeout=CALL_TIMEOUT_S,
                    httpx_client_factory=build_http_client_factory(),
                )
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        yield session


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DiscoveredCatalog:
    """The tools an application may expose, and what was left out.

    ``warnings`` pairs a tool name with the code that omitted it, and carries no
    schema fragment or validator text: a warning describes that a descriptor was
    unusable, never what was in it (R-SCHEMA-3).
    """

    tools: list[MCPTool]
    warnings: list[tuple[str, str]]


async def discover_stored_tools(server: ResolvedMcpServer) -> DiscoveredCatalog:
    """List the live tools a stored server's allowlist admits.

    One bounded session, one paginated ``tools/list``, per-tool screening, and
    a ceiling on the whole serialized answer. Every failure is ``not_started``:
    discovery runs no tool.

    Raises:
        McpExecutionError: with the code and status the route returns.
    """
    try:
        async with DISCOVERY_GATE.slot():
            return await _run_in_owner_task(_discover_once(server))
    except McpCapacityUnavailable:
        raise McpExecutionError(CODE_DISCOVERY_CAPACITY_UNAVAILABLE, ExecutionState.NOT_STARTED, 503) from None
    except McpExecutionError:
        raise
    except BaseException as exc:
        task = asyncio.current_task()
        if isinstance(exc, asyncio.CancelledError) and task is not None and task.cancelling():
            raise
        # Everything the transport can throw, in the shapes it actually throws
        # them: a bare ``CancelledError`` when the SDK's task group cancels its
        # caller, or a group when it collects several. Neither is an
        # ``Exception``, so a narrower arm here would let the most ordinary
        # failure of all escape the error contract and answer nothing.
        logger.warning("Stateless MCP discovery failed error_class=%s", failure_class(exc))
        raise McpExecutionError(CODE_CONNECTION_FAILED, ExecutionState.NOT_STARTED, 502) from None


async def _discover_once(server: ResolvedMcpServer) -> DiscoveredCatalog:
    stack = AsyncExitStack()
    try:
        session = await stack.enter_async_context(open_session(server))
        try:
            listed = await collect_tools(session, allowed_tools=server.allowed_tools)
        except McpDiscoveryRefused:
            raise McpExecutionError(CODE_DISCOVERY_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 502) from None

        tools: list[MCPTool] = []
        warnings: list[tuple[str, str]] = []
        for tool in listed:
            code = screen_tool(tool)
            if code is None:
                tools.append(tool)
            else:
                warnings.append((tool.name, code))

        if len(tools) > DISCOVERY_MAX_TOOLS:
            raise McpExecutionError(CODE_DISCOVERY_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 502)
        if _oversized(tools, DISCOVERY_RESPONSE_MAX_BYTES):
            raise McpExecutionError(CODE_DISCOVERY_LIMIT_EXCEEDED, ExecutionState.NOT_STARTED, 502)
        return DiscoveredCatalog(tools=tools, warnings=warnings)
    finally:
        await _close_bounded(stack)


async def collect_tools(session: Any, *, allowed_tools: list[str] | None) -> list[MCPTool]:
    """Page through ``tools/list`` and return the tools the allowlist admits.

    ``allowed_tools`` carries the three-state meaning of the stored column
    (R-RES-4): ``None`` exposes the whole live catalog, and a list exposes its
    intersection with it. An empty list is a deny-all the caller settles without
    connecting at all, so it never reaches here.

    Pagination stops early once every allowlisted name has been seen. A server
    with thousands of tools and a three-name allowlist is then one or two pages,
    not twenty. The returned-tool ceiling is applied after descriptor screening,
    so omitted tools do not take valid siblings away from the caller.

    Raises:
        McpDiscoveryRefused: on a pagination validation failure (R-DISC-4) or a
            discovery ceiling breach.
    """
    allowed = None if allowed_tools is None else set(allowed_tools)
    collected: dict[str, MCPTool] = {}
    examined = 0
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for _ in range(DISCOVERY_MAX_PAGES):
        page = await session.list_tools(cursor)
        tools = getattr(page, "tools", None)
        if not isinstance(tools, list):
            raise McpDiscoveryRefused
        for tool in tools:
            name = getattr(tool, "name", None)
            if not isinstance(name, str):
                raise McpDiscoveryRefused
            examined += 1
            if examined > DISCOVERY_MAX_EXAMINED:
                raise McpDiscoveryRefused
            if allowed is not None and name not in allowed:
                continue
            previous = collected.get(name)
            if previous is not None:
                # One tool listed twice is harmless; two different tools under
                # one name are not. An application maps a model's proposal back
                # to a name, so which descriptor it validated against would
                # decide what actually ran.
                if previous != tool:
                    raise McpDiscoveryRefused
                continue
            collected[name] = tool

        if allowed is not None and allowed.issubset(collected):
            break
        cursor = getattr(page, "nextCursor", None)
        if cursor is None:
            break
        if not isinstance(cursor, str) or cursor in seen_cursors:
            # A server repeating a cursor is either broken or trying to keep the
            # session open past every other bound; both are the same refusal.
            raise McpDiscoveryRefused
        seen_cursors.add(cursor)
    else:
        raise McpDiscoveryRefused

    return list(collected.values())


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


async def execute_stored_tool(
    server: ResolvedMcpServer,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    on_dispatch: Callable[[], None] | None = None,
    timings: MutableMapping[str, float] | None = None,
) -> CallToolResult:
    """Execute exactly one caller-authorized tool call and return its result.

    No ``list_tools``: MCP permits calling a known tool directly, and what
    authorizes this call is the stored allowlist and the authenticated
    workspace, not a live catalog (R-DISC-6). The route has already applied
    both by the time this runs.

    ``on_dispatch`` fires immediately before the transport begins writing, which
    is the boundary every classification below turns on.

    ``timings``, when given, collects the phase durations and the result size for
    the caller to log. It carries no tool name, argument, or result content
    (R-OBS-1, R-OBS-2), and it is filled in as each phase ends, so a failure part
    way through still leaves the phases that did run.

    Raises:
        McpExecutionError: with the code, state and status the route returns.
    """
    record = timings if timings is not None else {}
    admission_started = monotonic()
    try:
        async with EXECUTION_GATE.slot():
            record["admission_ms"] = (monotonic() - admission_started) * 1000
            return await _run_in_owner_task(_execute_once(server, tool_name, arguments, on_dispatch, record))
    except McpCapacityUnavailable:
        record["admission_ms"] = (monotonic() - admission_started) * 1000
        raise McpExecutionError(CODE_CAPACITY_UNAVAILABLE, ExecutionState.NOT_STARTED, 503) from None


async def _execute_once(
    server: ResolvedMcpServer,
    tool_name: str,
    arguments: dict[str, Any],
    on_dispatch: Callable[[], None] | None,
    timings: MutableMapping[str, float],
) -> CallToolResult:
    stack = AsyncExitStack()
    phase_started = monotonic()
    try:
        try:
            session = await stack.enter_async_context(open_session(server))
        except BaseException as exc:
            # ``BaseException``, not ``Exception``, because that is what a dead
            # server actually raises: the SDK yields inside an anyio task group,
            # so a closed port surfaces as a bare ``CancelledError`` and a
            # collected failure as a group, and neither is an ``Exception``. A
            # narrower arm let the commonest failure of all escape the contract
            # and return no response at all.
            #
            # Still ``not_started``: nothing was written, so the caller's own
            # execution claim is safe to release or retry (R-ERR-2). The
            # enclosing total deadline reaches here as a cancellation too, and
            # is reported the same way, which is the same status and state.
            logger.warning("Stateless MCP connection failed error_class=%s", failure_class(exc))
            raise McpExecutionError(CODE_CONNECTION_FAILED, ExecutionState.NOT_STARTED, 502) from None

        timings["connect_ms"] = (monotonic() - phase_started) * 1000
        if on_dispatch is not None:
            on_dispatch()
        phase_started = monotonic()
        try:
            async with asyncio.timeout(CALL_TIMEOUT_S):
                result = await session.call_tool(tool_name, arguments)
        except TimeoutError:
            raise McpExecutionError(CODE_OUTCOME_UNKNOWN, ExecutionState.OUTCOME_UNKNOWN, 504) from None
        except BaseException as exc:
            # Deliberately conservative (R-ERR-3), cancellation and grouped
            # cancellation included: a cancelled local transport says nothing
            # about whether the remote server ran the tool to completion, and
            # letting either escape would answer a possibly-completed mutation
            # with no execution state at all.
            logger.warning("Stateless MCP call failed after dispatch error_class=%s", failure_class(exc))
            raise McpExecutionError(CODE_OUTCOME_UNKNOWN, ExecutionState.OUTCOME_UNKNOWN, 502) from None

        finally:
            timings["call_ms"] = (monotonic() - phase_started) * 1000

        if result_exceeds_bound(result):
            # The tool ran, so this is not a clean refusal; Otari simply cannot
            # carry what came back.
            raise McpExecutionError(CODE_RESULT_TOO_LARGE, ExecutionState.OUTCOME_UNKNOWN, 502)
        timings["result_bytes"] = len(_encode(result))
        return result
    finally:
        cleanup_started = monotonic()
        await _close_bounded(stack)
        timings["cleanup_ms"] = (monotonic() - cleanup_started) * 1000


async def _close_bounded(stack: AsyncExitStack) -> None:
    """Close the transport without letting shutdown outlive or replace a result.

    Bounded and run in the same owner task that entered the transport
    (R-EXEC-1). A definitive result is already in hand by the time this runs, so
    a transport that will not close must not replace it. The timeout cancels and
    awaits cleanup in place, so no transport task is detached.
    """
    try:
        async with asyncio.timeout(CLEANUP_TIMEOUT_S):
            await stack.aclose()
    except BaseException as exc:
        logger.warning("Stateless MCP cleanup failed error_class=%s", failure_class(exc))
