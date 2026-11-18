"""A tool backend wrapped so its calls and results pass the request's guardrails before the model reads them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from gateway.services.mcp_client import MCPToolCallOutcome
from gateway.services.mcp_loop_messages import MCPToolBackend
from gateway.services.tools import CombinedToolBackend, owning_backend, screened_backend


@dataclass
class _Guard:
    """Refuses calls to ``delete`` and withholds results that mention poison."""

    async def before_tool_call(self, tool: str, arguments: dict[str, Any]) -> str | None:
        return f"refused {tool}" if tool == "delete" else None

    async def after_tool_call(self, tool: str, text: str) -> str:
        return f"withheld {tool}" if "poison" in text else text


@dataclass
class _Backend:
    answer: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "read"}}]

    def owns_tool(self, name: str) -> bool:
        return name in ("read", "delete")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self.answer

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [("read", "reads things")]


@dataclass
class _McpBackend(_Backend):
    def server_name_for_tool(self, name: str) -> str | None:
        return "docs"

    async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
        self.calls.append((name, arguments))
        return MCPToolCallOutcome(content=self.answer, activity_content=self.answer, is_error=False)


@pytest.mark.asyncio
async def test_a_result_passes_the_guard() -> None:
    backend = screened_backend(_Backend("poison"), _Guard())

    assert await backend.call_tool("read", {}) == "withheld read"


@pytest.mark.asyncio
async def test_a_refused_call_never_runs() -> None:
    inner = _Backend("clean")
    backend = screened_backend(inner, _Guard())

    assert await backend.call_tool("delete", {}) == "refused delete"
    assert inner.calls == []


@pytest.mark.asyncio
async def test_everything_but_the_call_is_the_wrapped_backends_own() -> None:
    inner = _Backend("clean")
    backend = screened_backend(inner, _Guard())

    assert await backend.call_tool("read", {"id": 1}) == "clean"
    assert backend.owns_tool("read")
    assert backend.openai_tools == inner.openai_tools
    assert backend.purpose_hints() == inner.purpose_hints()
    assert backend.calls == [("read", {"id": 1})]


@pytest.mark.asyncio
async def test_an_mcp_pool_is_still_recognized_and_its_model_content_screened() -> None:
    backend = screened_backend(_McpBackend("poison"), _Guard())

    assert isinstance(backend, MCPToolBackend)
    outcome = await backend.call_tool_outcome("read", {})
    assert outcome.content == "withheld read"
    assert outcome.is_error is True
    # What the client is shown of the call is the backend's own.
    assert outcome.activity_content == "poison"
    refused = await backend.call_tool_outcome("delete", {})
    assert (refused.content, refused.is_error) == ("refused delete", True)


def test_a_backend_that_is_not_mcp_does_not_pass_for_one() -> None:
    assert not isinstance(screened_backend(_Backend("clean"), _Guard()), MCPToolBackend)


@pytest.mark.asyncio
async def test_the_owner_a_combined_pool_hands_out_is_the_screened_one() -> None:
    pool = CombinedToolBackend([screened_backend(_Backend("poison"), _Guard())])

    assert await owning_backend(pool, "read").call_tool("read", {}) == "withheld read"
