"""Memory as tools the gateway runs inside a request's tool loop, for agents on any model.

A :class:`ToolBackend` like the sandbox and the web tools: it offers memory's six
tools, and runs each call with its own Unit of Work, because the request's
session is released before the provider is called. The caller is the one the
request's API key names, fixed when the backend is built; no argument changes it.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Self

from gateway.services.memory._service import MemoryCaller
from gateway.services.memory._tools import (
    INSTRUCTIONS,
    TOOL_NAMES,
    UNTRUSTED_NOTICE,
    ToolOutcome,
    openai_tool_definitions,
)
from gateway.services.tool_usage import TOOL_ERROR_SENTINEL

# Runs one call for the caller on a Unit of Work of its own; built by the API layer, which owns the wiring.
RunMemoryTool = Callable[[MemoryCaller, str, dict[str, Any]], Awaitable[ToolOutcome | None]]


class MemoryToolBackend:
    """Memory's tools for one request's caller."""

    def __init__(self, caller: MemoryCaller, run: RunMemoryTool) -> None:
        self.caller = caller
        self._run = run

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return openai_tool_definitions()

    def owns_tool(self, name: str) -> bool:
        return name in TOOL_NAMES

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """The call's answer as the model reads it: data with a notice that it is data, or a refusal to act on.

        Raises:
            KeyError: ``name`` is none of memory's tools.
        """
        outcome = await self._run(self.caller, name, arguments)
        if outcome is None:
            raise KeyError(name)
        if outcome.error is not None:
            return f"{TOOL_ERROR_SENTINEL} {outcome.error}"
        return f"{UNTRUSTED_NOTICE}\n{outcome.text}"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [("agent memory", INSTRUCTIONS)]
