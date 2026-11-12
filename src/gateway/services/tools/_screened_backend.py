"""A tool backend whose calls and results pass the request's guardrails."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from gateway.services.mcp_client import MCPToolCallOutcome

if TYPE_CHECKING:
    from gateway.services._tool_loop import ToolBackend


class ToolCallGuard(Protocol):
    """What a request's guardrails do around one call to a tool the gateway runs."""

    async def before_tool_call(self, tool: str, arguments: dict[str, Any]) -> str | None:
        """The text the model reads instead when the call is refused, or ``None`` to run it."""
        ...

    async def after_tool_call(self, tool: str, text: str) -> str:
        """What the model reads of the tool's result: the same text, or what replaces it when withheld."""
        ...


def screened_backend(backend: ToolBackend, guard: ToolCallGuard) -> ScreenedToolBackend:
    """``backend`` with each call passed through ``guard``, keeping an MCP pool recognizable as one."""
    if callable(getattr(backend, "call_tool_outcome", None)):
        return _ScreenedMcpBackend(backend, guard)
    return ScreenedToolBackend(backend, guard)


class ScreenedToolBackend:
    """``backend``, with each call and the result it returns passed through ``guard``.

    Every tool loop runs a call through ``call_tool`` (or, for an MCP pool,
    ``call_tool_outcome``), so wrapping the backend covers each loop and each
    call site in one place. Everything else is the wrapped backend's own: the
    loops and the native renderings read its tally, trace and result buffers.
    What the client is shown of a call is left as the backend reported it.
    """

    def __init__(self, backend: ToolBackend, guard: ToolCallGuard) -> None:
        self._backend = backend
        self._guard = guard

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return self._backend.openai_tools

    def owns_tool(self, name: str) -> bool:
        return self._backend.owns_tool(name)

    def purpose_hints(self) -> list[tuple[str, str]]:
        return self._backend.purpose_hints()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if (refused := await self._guard.before_tool_call(name, arguments)) is not None:
            return refused
        return await self._guard.after_tool_call(name, await self._backend.call_tool(name, arguments))


class _ScreenedMcpBackend(ScreenedToolBackend):
    """An MCP pool's wrapper, which declares the pool's own members.

    A runtime protocol check (``MCPToolBackend``) reads a class's attributes
    statically and never reaches ``__getattr__``, so the members it looks for
    are spelled out here, and only on the wrapper of a backend that has them.
    """

    def server_name_for_tool(self, name: str) -> str | None:
        return self._backend.server_name_for_tool(name)  # type: ignore[attr-defined, no-any-return]

    async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> Any:
        if (refused := await self._guard.before_tool_call(name, arguments)) is not None:
            return MCPToolCallOutcome(content=refused, activity_content=refused, is_error=True)
        outcome = await self._backend.call_tool_outcome(name, arguments)  # type: ignore[attr-defined]
        content = await self._guard.after_tool_call(name, outcome.content)
        return outcome if content is outcome.content else replace(outcome, content=content, is_error=True)
