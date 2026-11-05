"""Unit coverage for closing a streamed request's tool backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.api.routes._pipeline import (
    _close_tool_backend,
    _eager_backend_stream,
    _OpenedTools,
    _ToolBackendKind,
    open_stream,
)


async def _raise(exc: BaseException) -> None:
    raise exc


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        "gateway.api.routes._pipeline.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    return warnings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "logged"),
    [
        (RuntimeError("the server hung up"), "RuntimeError"),
        (ExceptionGroup("closing", [ConnectionError(), TimeoutError()]), "ConnectionError+TimeoutError"),
    ],
    ids=["exception", "exception-group"],
)
async def test_an_ordinary_close_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException, logged: str
) -> None:
    warnings = _capture_warnings(monkeypatch)

    await _close_tool_backend(_raise(failure), _ToolBackendKind.SANDBOX)

    assert warnings == [f"The sandbox tool backend failed to close: {logged}"]


@pytest.mark.asyncio
async def test_a_cancellation_inside_a_close_failure_still_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)
    mixed = BaseExceptionGroup("closing", [RuntimeError(), asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup) as raised:
        await _close_tool_backend(_raise(mixed), _ToolBackendKind.MCP)

    assert [type(exc) for exc in raised.value.exceptions] == [asyncio.CancelledError]
    assert warnings == ["The MCP tool backend failed to close: RuntimeError"]


@pytest.mark.asyncio
async def test_a_canceled_close_propagates_unlogged(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)

    with pytest.raises(asyncio.CancelledError):
        await _close_tool_backend(_raise(asyncio.CancelledError()), _ToolBackendKind.WEB_RETRIEVAL)

    assert warnings == []


class _Backend:
    def __init__(self, close_failure: BaseException | None = None) -> None:
        self.close_failure = close_failure
        self.closed = False
        self.closed_with: BaseException | None = None

    async def __aenter__(self) -> str:
        return "entered"

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True
        self.closed_with = exc[1] if isinstance(exc[1], BaseException) else None
        if self.close_failure is not None:
            raise self.close_failure


class _TwoChunkAdapter:
    """A format adapter whose tool loop streams two chunks."""

    name = "chat"

    def prepare_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def inject_hints(self, kwargs: dict[str, Any], hints: object, *, header: str | None) -> dict[str, Any]:
        return kwargs

    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure

    async def open_tool_loop_stream(self, *args: object, **kwargs: object) -> AsyncIterator[str]:
        yield "first"
        if self.failure is not None:
            raise self.failure
        yield "second"


def _eager_stream(stack: AsyncExitStack, failure: BaseException | None = None) -> AsyncIterator[str]:
    """The stream over backends entered on ``stack``, as ``open_stream`` builds it."""
    tool_ctx = SimpleNamespace(
        mcp_server_configs=[object()],
        use_sandbox=False,
        use_web_search=False,
        use_web_fetch=False,
        tools_header=None,
        max_tool_iterations=10,
        use_budget=None,
        native_tools=lambda _name: frozenset(),
    )
    opened = _OpenedTools(pool=cast(Any, SimpleNamespace(purpose_hints=lambda: [])), sandbox=None)
    return _eager_backend_stream(cast(Any, _TwoChunkAdapter(failure)), {}, opened, stack, cast(Any, tool_ctx))


@pytest.mark.asyncio
async def test_a_stream_the_client_left_closes_its_backends_with_that_exit() -> None:
    backend = _Backend()
    stack = AsyncExitStack()
    await stack.enter_async_context(backend)
    stream = cast(AsyncGenerator[str], _eager_stream(stack))

    assert await anext(stream) == "first"
    await stream.aclose()

    assert isinstance(backend.closed_with, GeneratorExit)


@pytest.mark.asyncio
async def test_a_stream_that_finished_closes_its_backends_cleanly() -> None:
    backend = _Backend()
    stack = AsyncExitStack()
    await stack.enter_async_context(backend)

    assert [chunk async for chunk in _eager_stream(stack)] == ["first", "second"]
    assert backend.closed
    assert backend.closed_with is None


@pytest.mark.asyncio
async def test_a_stream_that_failed_closes_its_backends_with_that_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend can abandon work nobody will read only if it is told the request failed."""
    warnings = _capture_warnings(monkeypatch)
    backend = _Backend(close_failure=RuntimeError("the server hung up"))
    stack = AsyncExitStack()
    await stack.enter_async_context(backend)
    failure = ValueError("the provider failed")

    with pytest.raises(ValueError, match="the provider failed"):
        _ = [chunk async for chunk in _eager_stream(stack, failure)]

    assert backend.closed_with is failure
    assert warnings == ["The MCP tool backend failed to close: RuntimeError"]


def _held_tool_ctx(hold_ms: int) -> Any:
    return SimpleNamespace(
        use_tool_loop=True,
        config=SimpleNamespace(streaming_first_chunk_hold_ms=hold_ms),
        mcp_server_configs=[object()],
        use_sandbox=False,
        use_web_search=False,
        use_web_fetch=False,
        tools_header=None,
        max_tool_iterations=10,
        use_budget=None,
        native_tools=lambda _name: frozenset(),
    )


@pytest.mark.asyncio
async def test_a_backend_slower_than_the_hold_opens_after_the_stream_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    release = asyncio.Event()

    async def slow_open(*_args: object) -> _OpenedTools:
        await release.wait()
        return _OpenedTools(pool=cast(Any, SimpleNamespace(purpose_hints=lambda: [])), sandbox=None)

    monkeypatch.setattr("gateway.api.routes._pipeline._open_tool_backends", slow_open)

    stream = await asyncio.wait_for(
        open_stream(adapter=cast(Any, _TwoChunkAdapter()), tool_ctx=_held_tool_ctx(10), call_kwargs={}), timeout=5
    )
    release.set()

    assert [chunk async for chunk in stream] == ["first", "second"]


@pytest.mark.asyncio
async def test_a_backend_that_fails_within_the_hold_raises_before_the_stream_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_open(*_args: object) -> _OpenedTools:
        raise ConnectionError("the MCP server is down")

    monkeypatch.setattr("gateway.api.routes._pipeline._open_tool_backends", failing_open)

    with pytest.raises(ConnectionError):
        await open_stream(adapter=cast(Any, _TwoChunkAdapter()), tool_ctx=_held_tool_ctx(5_000), call_kwargs={})
