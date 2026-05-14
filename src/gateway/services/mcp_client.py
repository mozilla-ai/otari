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

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from gateway.log_config import logger

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from gateway.models.mcp import McpServerConfig


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


@dataclass
class _ConnectedServer:
    name: str
    session: ClientSession
    tools: list[dict[str, Any]] = field(default_factory=list)
    purpose_hint: str | None = None


class MCPClientPool:
    """Manages concurrent MCP sessions for one request lifetime."""

    def __init__(self, configs: list[McpServerConfig]):
        self._configs = configs
        self._stack = AsyncExitStack()
        self._servers: dict[str, _ConnectedServer] = {}
        self._tool_owner: dict[str, str] = {}

    async def __aenter__(self) -> MCPClientPool:
        try:
            for cfg in self._configs:
                self._servers[cfg.name] = await self._connect(cfg)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    async def _connect(self, cfg: McpServerConfig) -> _ConnectedServer:
        headers: dict[str, str] | None = None
        if cfg.authorization_token:
            headers = {"Authorization": f"Bearer {cfg.authorization_token}"}

        transport = await self._stack.enter_async_context(streamablehttp_client(cfg.url, headers=headers))
        read, write, _ = transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        listed = await session.list_tools()
        allowed = set(cfg.allowed_tools) if cfg.allowed_tools else None
        openai_tools: list[dict[str, Any]] = []
        for tool in listed.tools:
            if allowed is not None and tool.name not in allowed:
                continue
            if tool.name in self._tool_owner:
                # Name collision across servers: keep the first server's tool.
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call against its owning MCP server. Returns flattened text."""
        owner = self._tool_owner.get(name)
        if owner is None:
            raise KeyError(f"No MCP server owns tool {name!r}")
        result = await self._servers[owner].session.call_tool(name, arguments)
        parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else str(block))
        flattened = "\n".join(parts)
        if result.isError:
            return f"[tool error] {flattened}"
        return flattened
