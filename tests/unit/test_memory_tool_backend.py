"""Memory's tools in the gateway's tool loop: what the model is offered, and what it reads back."""

import uuid
from typing import Any

import pytest

from gateway.services.memory import MemoryCaller, MemoryToolBackend, ToolOutcome
from gateway.services.memory._tools import TOOL_NAMES, UNTRUSTED_NOTICE
from gateway.services.tool_usage import is_tool_error

pytestmark = pytest.mark.asyncio

CALLER = MemoryCaller(uuid.uuid4(), uuid.uuid4(), is_manager=False)


def _backend(outcome: ToolOutcome | None) -> tuple[MemoryToolBackend, list[tuple[MemoryCaller, str, dict[str, Any]]]]:
    calls: list[tuple[MemoryCaller, str, dict[str, Any]]] = []

    async def run(caller: MemoryCaller, name: str, arguments: dict[str, Any]) -> ToolOutcome | None:
        calls.append((caller, name, arguments))
        return outcome

    return MemoryToolBackend(CALLER, run), calls


async def test_the_model_is_offered_the_six_tools_as_functions() -> None:
    backend, _ = _backend(None)

    names = {tool["function"]["name"] for tool in backend.openai_tools}
    assert names == TOOL_NAMES and all(tool["type"] == "function" for tool in backend.openai_tools)
    assert all(backend.owns_tool(name) for name in names) and not backend.owns_tool("web_search")
    assert backend.purpose_hints()[0][0] == "agent memory"


async def test_a_result_comes_back_as_data_with_a_notice_and_for_the_backends_own_caller() -> None:
    backend, calls = _backend(ToolOutcome(data={"results": []}))

    answer = await backend.call_tool("memory_search", {"query": "deploy", "workspace_id": "elsewhere"})

    assert answer == f'{UNTRUSTED_NOTICE}\n{{"results": []}}'
    assert calls == [(CALLER, "memory_search", {"query": "deploy", "workspace_id": "elsewhere"})]


async def test_a_refusal_comes_back_as_a_tool_error_the_model_can_act_on() -> None:
    backend, _ = _backend(ToolOutcome(error="Memory is off for this workspace."))

    answer = await backend.call_tool("memory_propose", {})

    assert is_tool_error(answer) and answer.endswith("Memory is off for this workspace.")


async def test_a_name_memory_does_not_own_is_not_answered() -> None:
    backend, _ = _backend(None)

    with pytest.raises(KeyError):
        await backend.call_tool("memory_delete_everything", {})
