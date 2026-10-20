"""The tool backends one request declared, driven by the tool loop as one pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gateway.services._tool_loop import ToolBackend


class CombinedToolBackend:
    """Several tool backends of one request, driven by the tool loop as one pool.

    A call runs on the first backend that owns its name, so put the gateway's own
    backends ahead of MCP. Each backend keeps its own tally, trace and result
    buffers, which is why a native rendering is handed :func:`owning_backend`'s
    answer rather than this pool.
    """

    def __init__(self, backends: Sequence[ToolBackend]) -> None:
        self._backends = tuple(backends)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool for backend in self._backends for tool in backend.openai_tools]

    def owns_tool(self, name: str) -> bool:
        return self.backend_for(name) is not None

    def backend_for(self, name: str) -> ToolBackend | None:
        """The backend that runs ``name``, or ``None`` when none of them owns it."""
        return next((backend for backend in self._backends if backend.owns_tool(name)), None)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        backend = self.backend_for(name)
        if backend is None:
            raise KeyError(f"No tool backend owns tool {name!r}")
        return await backend.call_tool(name, arguments)

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [hint for backend in self._backends for hint in backend.purpose_hints()]


def owning_backend(pool: ToolBackend, name: str) -> ToolBackend:
    """The backend that runs ``name``: the member of a combined pool that owns it, else ``pool`` itself."""
    if isinstance(pool, CombinedToolBackend):
        return pool.backend_for(name) or pool
    return pool
