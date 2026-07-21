"""Combine several tool backends behind one tool-loop ``pool``.

The tool loop (:mod:`gateway.services.mcp_loop`) drives a single ``pool`` implementing the
``ToolBackend`` protocol. Historically exactly one gateway tool backend was active per request
(MCP pool, sandbox, or web search — mutually exclusive). The memory tool is designed to run
*alongside* whichever of those is active, so this wrapper presents a list of backends as one:
it merges their advertised tools and purpose hints and routes each tool call to the owning
backend. The loop itself is unchanged.

It is also an async context manager: entering it enters every member backend (in order, via an
``AsyncExitStack``) and exiting closes them in reverse. Wrapping even a single backend is fine
and lets the dispatch code treat "one backend" and "several" uniformly; where the caller enters
the context (eagerly before streaming, or lazily inside the stream generator) is what preserves
each member's original open-time semantics.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from gateway.services.mcp_loop import ToolBackend


class CompositeToolBackend:
    """A ``ToolBackend`` that fans out to an ordered list of member backends."""

    def __init__(self, backends: list[ToolBackend]) -> None:
        if not backends:
            raise ValueError("CompositeToolBackend requires at least one backend")
        self._backends = backends
        self._entered = backends
        self._stack: AsyncExitStack = AsyncExitStack()

    async def __aenter__(self) -> CompositeToolBackend:
        entered: list[ToolBackend] = []
        try:
            for backend in self._backends:
                entered.append(await self._stack.enter_async_context(backend))  # type: ignore[arg-type] # noqa: PERF401
        except BaseException:
            # A member failed to open (e.g. a backend is unreachable). Close the ones that did
            # so eager-open failures don't leak half-entered backends, then propagate.
            await self._stack.aclose()
            raise
        self._entered = entered
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self._stack.aclose()

    # ----- duck-typed protocol the MCP loop uses on `pool` -----

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for backend in self._entered:
            tools.extend(backend.openai_tools)
        return tools

    def owns_tool(self, name: str) -> bool:
        return any(backend.owns_tool(name) for backend in self._entered)

    def purpose_hints(self) -> list[tuple[str, str]]:
        hints: list[tuple[str, str]] = []
        for backend in self._entered:
            # Not every ToolBackend advertises purpose hints (it's outside the loop's
            # protocol), so only collect from those that do.
            hint_fn = getattr(backend, "purpose_hints", None)
            if callable(hint_fn):
                hints.extend(hint_fn())
        return hints

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        for backend in self._entered:
            if backend.owns_tool(name):
                return await backend.call_tool(name, arguments)
        raise KeyError(f"No backend owns tool {name!r}")
