"""Bounded stored-server discovery (R-DISC-5, R-SCHEMA-3, R-ADM-1).

Discovery either returns the complete authorized catalog or nothing. The one
permitted partial result is the labeled per-tool omission: a single unusable
descriptor must not take its siblings with it.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from mcp import ClientSession
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from gateway.models.mcp import ResolvedMcpServer
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import (
    DISCOVERY_MAX_TOOLS,
    DISCOVERY_RESPONSE_MAX_BYTES,
    SCHEMA_MAX_BYTES,
    ConcurrencyGate,
    ExecutionState,
    McpExecutionError,
    discover_stored_tools,
)

SERVER = ResolvedMcpServer(
    id=uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e"),
    name="github",
    url="https://mcp.example.com/mcp",
    authorization_token="server-secret",
)


def _tool(name: str, **overrides: Any) -> MCPTool:
    fields: dict[str, Any] = {"name": name, "description": name, "inputSchema": {"type": "object"}}
    fields.update(overrides)
    return MCPTool(**fields)


class _FakeSession:
    def __init__(self, tools: list[MCPTool]) -> None:
        self._tools = tools
        self.delay = 0.0

    async def list_tools(self, cursor: str | None = None) -> ListToolsResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        return ListToolsResult(tools=self._tools)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"session": _FakeSession([_tool("create_issue")]), "connect_error": None}

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> Any:
        if state["connect_error"] is not None:
            raise state["connect_error"]
        yield state["session"]

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    return state


@pytest.mark.asyncio
async def test_the_admitted_catalog_is_returned(opened: dict[str, Any]) -> None:
    catalog = await discover_stored_tools(SERVER)

    assert [t.name for t in catalog.tools] == ["create_issue"]
    assert catalog.warnings == []


@pytest.mark.asyncio
async def test_a_real_client_session_closes_without_cancelling_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_tools(_session: ClientSession, cursor: str | None = None) -> ListToolsResult:
        assert cursor is None
        return ListToolsResult(tools=[_tool("create_issue")])

    @asynccontextmanager
    async def open_real_session(*args: Any, **kwargs: Any) -> AsyncIterator[ClientSession]:
        incoming_writer, incoming_reader = anyio.create_memory_object_stream(1)
        outgoing_writer, outgoing_reader = anyio.create_memory_object_stream(1)
        async with incoming_writer, incoming_reader, outgoing_writer, outgoing_reader:
            async with ClientSession(incoming_reader, outgoing_writer) as real_session:
                yield real_session

    monkeypatch.setattr(ClientSession, "list_tools", list_tools)
    monkeypatch.setattr(mcp_stateless, "open_session", open_real_session)

    catalog = await discover_stored_tools(SERVER)

    assert [tool.name for tool in catalog.tools] == ["create_issue"]
    task = asyncio.current_task()
    assert task is not None and task.cancelling() == 0


@pytest.mark.asyncio
async def test_one_unusable_descriptor_is_omitted_and_labeled(opened: dict[str, Any]) -> None:
    huge = {"type": "object", "properties": {"x": {"description": "y" * (SCHEMA_MAX_BYTES + 1)}}}
    opened["session"] = _FakeSession([_tool("create_issue"), _tool("broken", inputSchema=huge)])

    catalog = await discover_stored_tools(SERVER)

    assert [t.name for t in catalog.tools] == ["create_issue"]
    assert catalog.warnings == [("broken", "mcp_tool_schema_unsupported")]


@pytest.mark.asyncio
async def test_an_omitted_descriptor_does_not_count_toward_the_returned_tool_ceiling(
    opened: dict[str, Any],
) -> None:
    external_schema = {"type": "object", "$ref": "https://example.com/schema"}
    opened["session"] = _FakeSession(
        [_tool(f"valid{i}") for i in range(DISCOVERY_MAX_TOOLS)]
        + [_tool("broken", inputSchema=external_schema)]
    )

    catalog = await discover_stored_tools(SERVER)

    assert len(catalog.tools) == DISCOVERY_MAX_TOOLS
    assert catalog.warnings == [("broken", "mcp_tool_schema_unsupported")]


@pytest.mark.asyncio
async def test_more_than_the_returned_tool_ceiling_refuses_the_whole_response(
    opened: dict[str, Any],
) -> None:
    opened["session"] = _FakeSession([_tool(f"tool{i}") for i in range(DISCOVERY_MAX_TOOLS + 1)])

    with pytest.raises(McpExecutionError) as raised:
        await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_discovery_limit_exceeded"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED


@pytest.mark.asyncio
async def test_a_pagination_failure_refuses_the_whole_response(
    opened: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise mcp_stateless.McpDiscoveryRefused

    monkeypatch.setattr(mcp_stateless, "collect_tools", refuse)

    with pytest.raises(McpExecutionError) as raised:
        await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_discovery_limit_exceeded"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_an_oversized_serialized_response_refuses_the_whole_response(opened: dict[str, Any]) -> None:
    filler = "z" * 4000
    count = DISCOVERY_RESPONSE_MAX_BYTES // len(filler) + 2
    opened["session"] = _FakeSession([_tool(f"t{i}", description=filler) for i in range(count)])

    with pytest.raises(McpExecutionError) as raised:
        await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_discovery_limit_exceeded"


@pytest.mark.asyncio
async def test_a_connection_failure_is_reported_as_one(opened: dict[str, Any]) -> None:
    opened["connect_error"] = RuntimeError("server-secret refused")

    with pytest.raises(McpExecutionError) as raised:
        await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_connection_failed"
    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_no_free_discovery_slot_before_the_deadline_is_refused(
    opened: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ConcurrencyGate(limit=1, admission_timeout_s=0.01)
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_GATE", gate)

    async with gate.slot():
        with pytest.raises(McpExecutionError) as raised:
            await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_discovery_capacity_unavailable"
    assert raised.value.status_code == 503


@pytest.mark.asyncio
async def test_a_task_group_cancelling_the_caller_is_a_connection_failure(opened: dict[str, Any]) -> None:
    """A dead server raises a bare CancelledError, which is not an ``Exception``."""
    opened["connect_error"] = asyncio.CancelledError()

    with pytest.raises(McpExecutionError) as raised:
        await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_connection_failed"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED


@pytest.mark.asyncio
async def test_a_server_dying_mid_pagination_is_a_connection_failure(
    opened: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def die(cursor: str | None = None) -> Any:
        raise BaseExceptionGroup("unhandled errors in a TaskGroup", [asyncio.CancelledError()])

    monkeypatch.setattr(opened["session"], "list_tools", die)

    with pytest.raises(McpExecutionError) as raised:
        await discover_stored_tools(SERVER)

    assert raised.value.code == "mcp_connection_failed"
    assert raised.value.status_code == 502
