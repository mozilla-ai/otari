"""MCP client pool for the gateway.

Holds a set of live MCP sessions for the duration of a single chat-completion
request, exposes the union of their tools in OpenAI tool format, and routes
tool calls back to the owning server.

Use as an async context manager so sessions are cleaned up when the request
ends or the loop exits:

    async with MCPClientPool(configs) as pool:
        tools = pool.openai_tools
        result_text = await pool.call_tool(name, arguments)
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import anyio
import anyio.lowlevel
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from gateway.exceptions.tools_exceptions import McpSessionsInterruptedError
from gateway.log_config import logger
from gateway.services.mcp_stateless import failure_class
from gateway.services.tool_usage import ToolUsageTally

if TYPE_CHECKING:
    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPTool

    from gateway.models.mcp import McpServerConfig


def _effective_port(url: httpx.URL) -> int | None:
    if url.port is not None:
        return url.port
    return {"http": 80, "https": 443}.get(url.scheme)


def _is_allowed_mcp_redirect(base: httpx.URL, target: httpx.URL) -> bool:
    same_host = base.host == target.host
    same_origin = (
        same_host and base.scheme == target.scheme and _effective_port(base) == _effective_port(target)
    )
    https_upgrade = (
        same_host
        and base.scheme == "http"
        and _effective_port(base) == 80
        and target.scheme == "https"
        and _effective_port(target) == 443
    )
    return same_origin or https_upgrade


def _origin_bound_http_client(
    base_url: str,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Create an MCP HTTP client that refuses redirects outside the vetted origin."""
    base = httpx.URL(base_url)

    async def enforce_origin(request: httpx.Request) -> None:
        if not _is_allowed_mcp_redirect(base, request.url):
            raise httpx.RequestError("MCP redirect target is outside the validated origin", request=request)

    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout if timeout is not None else httpx.Timeout(30.0),
        auth=auth,
        follow_redirects=True,
        event_hooks={"request": [enforce_origin]},
    )


CLOSE_TIMEOUT_SECONDS = 10.0
_CANCEL_GRACE_SECONDS = 1.0


def mcp_tool_to_openai(tool: MCPTool) -> dict[str, Any]:
    """Convert an MCP Tool descriptor to an OpenAI-format function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


@dataclass(frozen=True)
class MCPToolCallOutcome:
    """Model-facing and client-facing renderings of one MCP result."""

    content: str
    activity_content: str
    is_error: bool
    transport_error: bool = False


@dataclass
class _ConnectedServer:
    name: str
    session: ClientSession
    tools: list[dict[str, Any]] = field(default_factory=list)
    purpose_hint: str | None = None


class MCPClientPool:
    """Manages concurrent MCP sessions for one request lifetime.

    The sessions open and close in a task the pool owns, so any task may close the pool.
    The MCP transport holds an anyio task group, which must exit in the task that entered it.
    The owned task belongs to no task group, and it runs in a copy of the opening task's context.
    """

    def __init__(self, configs: list[McpServerConfig], *, tally: ToolUsageTally | None = None):
        self._configs = configs
        self._stack = AsyncExitStack()
        self._servers: dict[str, _ConnectedServer] = {}
        self._tool_owner: dict[str, str] = {}
        # Per-request accounting, owned by the route and passed in. None when the
        # pool runs outside a billed request (tests, direct use).
        self._tally = tally
        self._owner: asyncio.Task[None] | None = None
        self._release = asyncio.Event()

    async def __aenter__(self) -> MCPClientPool:
        if self._owner is not None:
            raise RuntimeError("An MCP client pool opens only once")
        opened: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        owner = asyncio.create_task(self._hold_sessions(opened), name="mcp-client-pool")
        self._owner = owner
        try:
            await asyncio.wait((opened, owner), return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            # The release comes first, so an owner that ignores the cancellation still ends once it opens.
            self._release.set()
            owner.cancel()
            try:
                await _wait_for_owner(owner)
            finally:
                _retrieve_failure(owner)
            raise
        if not opened.done():
            if owner.cancelled():
                raise McpSessionsInterruptedError
            # The owner ends before opening only by raising, so this re-raises why.
            owner.result()
        return self

    async def __aexit__(self, *exc: object) -> None:
        owner = self._owner
        if owner is None:
            return
        self._release.set()
        try:
            in_time = await _wait_for_owner(owner)
        except asyncio.CancelledError:
            if (failure := _retrieve_failure(owner)) is not None:
                logger.warning("MCP sessions failed to close: %s", failure_class(failure))
            raise
        if not in_time:
            _retrieve_failure(owner)
            return
        if owner.cancelled():
            logger.warning("MCP sessions were canceled before they closed")
            return
        owner.result()

    async def _hold_sessions(self, opened: asyncio.Future[None]) -> None:
        async with self._stack:
            for cfg in self._configs:
                if cfg.name in self._servers:
                    raise ValueError(f"Duplicate MCP server name {cfg.name!r}")
                self._servers[cfg.name] = await self._connect(cfg)
            opened.set_result(None)
            await self._release.wait()

    async def _connect(self, cfg: McpServerConfig) -> _ConnectedServer:
        headers: dict[str, str] | None = None
        if cfg.authorization_token:
            headers = {"Authorization": f"Bearer {cfg.authorization_token}"}

        transport = await self._stack.enter_async_context(
            streamablehttp_client(
                cfg.url,
                headers=headers,
                httpx_client_factory=partial(_origin_bound_http_client, cfg.url),
            )
        )
        read, write, _ = transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        listed = await session.list_tools()
        allowed = set(cfg.allowed_tools) if cfg.allowed_tools is not None else None
        openai_tools: list[dict[str, Any]] = []
        for tool in listed.tools:
            if allowed is not None and tool.name not in allowed:
                continue
            if tool.name in self._tool_owner:
                # Tool-name collision policy: **first server wins**. Subsequent servers'
                # tools with the same name are dropped entirely — they're not added to
                # `_tool_owner` or `openai_tools`, so the model never sees them and the
                # loop never tries to dispatch to them. Logged as a warning; not an error
                # because legitimate setups can have overlapping tool names across servers
                # (e.g. two filesystem MCPs both exposing `read_file`).
                logger.warning(
                    "MCP tool name collision on %r; %s already owns it, %s skipped",
                    tool.name,
                    self._tool_owner[tool.name],
                    cfg.name,
                )
                continue
            openai_tools.append(mcp_tool_to_openai(tool))
            self._tool_owner[tool.name] = cfg.name

        return _ConnectedServer(name=cfg.name, session=session, tools=openai_tools, purpose_hint=cfg.purpose_hint)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [t for s in self._servers.values() for t in s.tools]

    def owns_tool(self, name: str) -> bool:
        return name in self._tool_owner

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(s.name, s.purpose_hint) for s in self._servers.values() if s.purpose_hint]

    def server_name_for_tool(self, name: str) -> str | None:
        """Return the configured server that owns ``name``, if connected."""
        return self._tool_owner.get(name)

    async def _call_tool_result(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Execute an MCP call and return the server's native typed result."""
        owner = self._tool_owner.get(name)
        if owner is None:
            raise KeyError(f"No MCP server owns tool {name!r}")
        try:
            return await self._servers[owner].session.call_tool(name, arguments)
        except Exception:
            if self._tally is not None:
                self._tally.record_failure(name)
            raise

    async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
        """Execute an MCP call and preserve its explicit ``isError`` status.

        The Messages stream uses this richer form for server-owned activity
        events. Other API formats keep using :meth:`call_tool`, which returns the
        model-facing text. Transport failures are converted here so their URLs,
        headers, or credentials cannot reach logs or a model provider through any
        API format.
        """
        if name not in self._tool_owner:
            raise KeyError(f"No MCP server owns tool {name!r}")
        try:
            result = await self._call_tool_result(name, arguments)
        except Exception as exc:
            logger.warning("MCP tool %s execution failed: %s", name, type(exc).__name__)
            return MCPToolCallOutcome(
                content="[tool error] MCP tool execution failed",
                activity_content="MCP tool execution failed",
                is_error=True,
                transport_error=True,
            )
        parts = [_render_content_block(block) for block in result.content]
        flattened = "\n".join(p for p in parts if p)
        rendered = f"[tool error] {flattened}" if result.isError else flattened
        if self._tally is not None:
            self._tally.record_result(name, rendered)
        return MCPToolCallOutcome(
            content=rendered,
            activity_content=flattened,
            is_error=bool(result.isError),
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call against its owning MCP server and flatten the result to text.

        MCP supports rich content blocks (image, embedded resource, structured data). This
        flattener concatenates text blocks verbatim and renders non-text blocks as a brief
        ``[type=…]`` placeholder so the model can at least see that *something* came back.
        For tools that return images or large embedded resources, this is intentionally
        lossy — fine for the current text-tool use case, but a future improvement is to
        pass image content through as multimodal message blocks when the model supports it.

        The call is recorded on the request's tally (see
        :class:`gateway.services.tool_usage.ToolUsageTally`); an ``isError`` result
        is counted and never billed.
        """
        return (await self.call_tool_outcome(name, arguments)).content


async def _wait_for_owner(owner: asyncio.Task[None]) -> bool:
    """Wait for ``owner`` to end, and only then raise a cancellation of the waiting task.

    Canceling the wait never cancels ``owner``, so a canceled close still closes every session.
    Returns whether ``owner`` ended in time; one still running then is canceled, and abandoned if it ignores that.
    """
    loop = asyncio.get_running_loop()
    # The shield stops an anyio scope from redelivering its cancellation on every loop pass.
    with anyio.CancelScope(shield=True):
        cancellation = await _wait_until_done(owner, loop.time() + CLOSE_TIMEOUT_SECONDS)
        in_time = owner.done()
        if not in_time:
            logger.warning("MCP sessions did not close within %s seconds", CLOSE_TIMEOUT_SECONDS)
            owner.cancel()
            cancellation = await _wait_until_done(owner, loop.time() + _CANCEL_GRACE_SECONDS) or cancellation
            if not owner.done():
                owner.add_done_callback(_log_late_failure)
    if cancellation is not None:
        raise cancellation
    await anyio.lowlevel.checkpoint_if_cancelled()
    return in_time


async def _wait_until_done(task: asyncio.Task[None], deadline: float) -> asyncio.CancelledError | None:
    """Wait for ``task`` to end or ``deadline`` to pass, returning a cancellation of the wait instead of raising it."""
    loop = asyncio.get_running_loop()
    cancellation: asyncio.CancelledError | None = None
    while not task.done() and (remaining := deadline - loop.time()) > 0:
        try:
            await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError as exc:
            cancellation = exc
    return cancellation


def _log_late_failure(task: asyncio.Task[None]) -> None:
    """Log by type why an abandoned owner failed, so asyncio never logs its message."""
    if (failure := _retrieve_failure(task)) is not None:
        logger.warning("Abandoned MCP sessions failed to close: %s", failure_class(failure))


def _retrieve_failure(task: asyncio.Task[None]) -> BaseException | None:
    """Return why ``task`` failed, if it ended by raising.

    Reading the exception marks it retrieved, so asyncio does not log it as lost.
    """
    if not task.done() or task.cancelled():
        return None
    return task.exception()


def _render_content_block(block: Any) -> str:
    """Render an MCP content block as a single string for inclusion in a tool message."""
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    btype = getattr(block, "type", None) or type(block).__name__.lower()
    # ImageContent has `data` (base64) and `mimeType`; just summarize.
    if btype in ("image", "image_content", "imagecontent"):
        mime = getattr(block, "mimeType", None) or getattr(block, "mime_type", None) or "image"
        data = getattr(block, "data", None)
        size = len(data) if isinstance(data, (str, bytes)) else "?"
        return f"[image type={mime} bytes_b64={size}]"
    # EmbeddedResource has a `resource` with `uri` and either `text` or `blob`.
    if btype in ("resource", "embedded_resource", "embeddedresource"):
        resource = getattr(block, "resource", None)
        uri = getattr(resource, "uri", None) if resource else None
        return f"[resource uri={uri or '?'}]"
    return f"[content type={btype}]"
