"""One bounded stateless execution (execution steps 9-13, R-EXEC-1, R-ERR-2, R-ERR-3).

The dispatch boundary is what these cover. Before the transport starts writing
``tools/call``, Otari knows the tool did not run and says ``not_started``. From
that moment on it cannot know, so every failure is ``outcome_unknown`` and no
path may invite a retry of a call that may already have mutated something.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
from unittest.mock import Mock

import anyio
import pytest
from httpx import ConnectError
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

from gateway import log_config
from gateway.models.mcp import ResolvedMcpServer
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import (
    RESULT_MAX_BYTES,
    ConcurrencyGate,
    ExecutionState,
    McpExecutionError,
    execute_stored_tool,
)

SERVER = ResolvedMcpServer(
    id=uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e"),
    name="github",
    url="https://mcp.example.com/mcp",
    authorization_token="server-secret",
    allowed_tools=["create_issue"],
)

RESULT = CallToolResult(content=[TextContent(type="text", text="Created issue #42")])


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listed = 0
        self.call_error: BaseException | None = None
        self.result = RESULT
        # Set by the fixture below, so a test can fail the connect or the
        # cleanup half of the substituted session independently.
        self.transport: dict[str, BaseException | None] = {}

    async def list_tools(self, cursor: str | None = None) -> Any:
        self.listed += 1
        raise AssertionError("execution must never call list_tools")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return self.result


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    """Substitute the real transport with a session that records what it is asked."""
    fake = _FakeSession()
    state: dict[str, BaseException | None] = {"connect_error": None, "cleanup_error": None}
    fake.transport = state

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> Any:
        if state["connect_error"] is not None:
            raise state["connect_error"]
        yield fake
        if state["cleanup_error"] is not None:
            raise state["cleanup_error"]

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    return fake


@pytest.mark.asyncio
async def test_the_exact_tool_is_called_once_without_live_discovery(session: _FakeSession) -> None:
    result = await execute_stored_tool(SERVER, "create_issue", {"title": "Approved"})

    assert result == RESULT
    assert session.calls == [("create_issue", {"title": "Approved"})]
    assert session.listed == 0


@pytest.mark.asyncio
async def test_a_real_client_session_closes_without_cancelling_its_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def call_tool(
        _session: ClientSession,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        assert (name, arguments) == ("create_issue", {"title": "Approved"})
        return RESULT

    @asynccontextmanager
    async def open_real_session(*args: Any, **kwargs: Any) -> AsyncIterator[ClientSession]:
        incoming_writer, incoming_reader = anyio.create_memory_object_stream(1)
        outgoing_writer, outgoing_reader = anyio.create_memory_object_stream(1)
        async with incoming_writer, incoming_reader, outgoing_writer, outgoing_reader:
            async with ClientSession(incoming_reader, outgoing_writer) as real_session:
                yield real_session

    monkeypatch.setattr(ClientSession, "call_tool", call_tool)
    monkeypatch.setattr(mcp_stateless, "open_session", open_real_session)

    result = await execute_stored_tool(SERVER, "create_issue", {"title": "Approved"})

    assert result == RESULT
    task = asyncio.current_task()
    assert task is not None and task.cancelling() == 0


@pytest.mark.asyncio
async def test_a_connection_failure_is_not_started(session: _FakeSession) -> None:
    session.transport["connect_error"] = RuntimeError("server-secret refused")

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_connection_failed"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 502
    assert session.calls == []


@pytest.mark.asyncio
async def test_a_deadline_after_dispatch_is_an_unknown_outcome(
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_stateless, "CALL_TIMEOUT_S", 0.01)

    async def never_answer(name: str, arguments: dict[str, Any]) -> CallToolResult:
        session.calls.append((name, arguments))
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    monkeypatch.setattr(session, "call_tool", never_answer)

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_outcome_unknown"
    assert raised.value.execution_state is ExecutionState.OUTCOME_UNKNOWN
    assert raised.value.status_code == 504


@pytest.mark.asyncio
async def test_a_transport_failure_after_dispatch_is_an_unknown_outcome(session: _FakeSession) -> None:
    session.call_error = RuntimeError("connection reset")

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_outcome_unknown"
    assert raised.value.execution_state is ExecutionState.OUTCOME_UNKNOWN
    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_cancellation_after_dispatch_is_an_unknown_outcome(session: _FakeSession) -> None:
    """Cancelling local work cannot assert the remote server stopped the tool."""
    session.call_error = asyncio.CancelledError()

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.execution_state is ExecutionState.OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_an_oversized_result_is_refused_as_an_unknown_outcome(session: _FakeSession) -> None:
    session.result = CallToolResult(content=[TextContent(type="text", text="x" * (RESULT_MAX_BYTES + 1))])

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_result_too_large"
    assert raised.value.execution_state is ExecutionState.OUTCOME_UNKNOWN
    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_a_definitive_result_survives_a_cleanup_failure(session: _FakeSession) -> None:
    """R-EXEC-1: transport shutdown must not turn a completed mutation into a retry."""
    session.transport["cleanup_error"] = RuntimeError("server-secret cleanup failure")

    assert await execute_stored_tool(SERVER, "create_issue", {"title": "Approved"}) == RESULT


@pytest.mark.asyncio
async def test_a_server_reported_error_is_a_definitive_result(session: _FakeSession) -> None:
    session.result = CallToolResult(content=[TextContent(type="text", text="denied")], isError=True)

    result = await execute_stored_tool(SERVER, "create_issue", {})

    assert result.isError is True


@pytest.mark.asyncio
async def test_no_free_slot_before_the_deadline_is_not_started(
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ConcurrencyGate(limit=1, admission_timeout_s=0.01)
    monkeypatch.setattr(mcp_stateless, "EXECUTION_GATE", gate)

    async with gate.slot():
        with pytest.raises(McpExecutionError) as raised:
            await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_capacity_unavailable"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 503
    assert session.calls == []


@pytest.mark.asyncio
async def test_phase_timings_and_result_size_are_recorded_without_content(session: _FakeSession) -> None:
    """R-OBS-1: the route logs phases and sizes, and R-OBS-2 leaves out everything else."""
    timings: dict[str, float] = {}

    await execute_stored_tool(SERVER, "create_issue", {"title": "Approved"}, timings=timings)

    assert set(timings) == {"admission_ms", "connect_ms", "call_ms", "cleanup_ms", "result_bytes"}
    assert all(value >= 0 for value in timings.values())
    assert timings["result_bytes"] > 0


@pytest.mark.asyncio
async def test_phase_timings_survive_a_failure_after_dispatch(session: _FakeSession) -> None:
    session.call_error = RuntimeError("connection reset")
    timings: dict[str, float] = {}

    with pytest.raises(McpExecutionError):
        await execute_stored_tool(SERVER, "create_issue", {}, timings=timings)

    assert "connect_ms" in timings
    assert "call_ms" in timings
    assert "result_bytes" not in timings


# --------------------------------------------------------------------------- #
# The shapes a real MCP transport failure actually arrives in
#
# The SDK yields inside ``anyio.create_task_group()``, so a dead server does not
# raise the tidy ``Exception`` a mocked pool does. Against the pinned ``mcp``,
# ``session.initialize()`` against a closed port raises a bare
# ``CancelledError``, which is not an ``Exception`` at all, and transport
# shutdown separately raises an ``ExceptionGroup``. Both have to land inside the
# error contract rather than escaping it.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_task_group_cancelling_the_caller_is_a_connection_failure(session: _FakeSession) -> None:
    """The shape a closed port actually produces: a bare CancelledError."""
    session.transport["connect_error"] = asyncio.CancelledError()

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_connection_failed"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 502
    assert session.calls == []


@pytest.mark.asyncio
async def test_a_grouped_connection_failure_is_reported_by_its_leaf(
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ExceptionGroup`` alone says nothing, so the log names what was inside it."""
    warning = Mock()
    monkeypatch.setattr(log_config.logger, "warning", warning)
    session.transport["connect_error"] = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [ConnectError("All connection attempts failed")],
    )

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_connection_failed"
    warning.assert_called_once_with("Stateless MCP connection failed error_class=%s", "ConnectError")


@pytest.mark.asyncio
async def test_a_group_holding_a_cancellation_after_dispatch_is_an_unknown_outcome(
    session: _FakeSession,
) -> None:
    """A ``BaseExceptionGroup`` is not an ``Exception``, and must not escape the contract."""
    session.call_error = BaseExceptionGroup("unhandled errors in a TaskGroup", [asyncio.CancelledError()])

    with pytest.raises(McpExecutionError) as raised:
        await execute_stored_tool(SERVER, "create_issue", {})

    assert raised.value.code == "mcp_outcome_unknown"
    assert raised.value.execution_state is ExecutionState.OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_a_grouped_cleanup_failure_still_preserves_the_result(session: _FakeSession) -> None:
    """The shape transport shutdown actually produces, over R-EXEC-1."""
    session.transport["cleanup_error"] = BaseExceptionGroup(
        "unhandled errors in a TaskGroup",
        [asyncio.CancelledError()],
    )

    assert await execute_stored_tool(SERVER, "create_issue", {"title": "Approved"}) == RESULT


@pytest.mark.asyncio
async def test_a_cleanup_timeout_cancels_and_awaits_the_closer(monkeypatch: pytest.MonkeyPatch) -> None:
    finished = asyncio.Event()

    async def stalled_cleanup() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    stack = AsyncExitStack()
    stack.push_async_callback(stalled_cleanup)
    monkeypatch.setattr(mcp_stateless, "CLEANUP_TIMEOUT_S", 0.01)

    await mcp_stateless._close_bounded(stack)

    assert finished.is_set(), "the timed-out closer must not remain detached"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError(), "RuntimeError"),
        (ExceptionGroup("g", [ConnectError("x")]), "ConnectError"),
        (BaseExceptionGroup("g", [asyncio.CancelledError()]), "CancelledError"),
        (
            ExceptionGroup("g", [ConnectError("x"), ExceptionGroup("inner", [TimeoutError()])]),
            "ConnectError+TimeoutError",
        ),
        (ExceptionGroup("g", [ConnectError("x"), ConnectError("y")]), "ConnectError"),
    ],
)
def test_a_failure_class_names_the_leaves_and_nothing_else(exc: BaseException, expected: str) -> None:
    assert mcp_stateless.failure_class(exc) == expected


def test_a_failure_class_carries_no_message_from_the_exception() -> None:
    """R-OBS-2: an exception message can hold a URL, a credential, or an argument."""
    assert "server-secret" not in mcp_stateless.failure_class(RuntimeError("server-secret leaked"))
