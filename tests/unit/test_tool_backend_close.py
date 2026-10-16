"""Unit coverage for closing a streamed request's tool backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack
from typing import cast

import pytest

from gateway.api.routes._pipeline import (
    _close_tool_backend,
    _held_tool_backend,
    _stream_with_stack_cleanup,
    _ToolBackendKind,
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


@pytest.mark.asyncio
async def test_a_held_backend_yields_what_entering_returns_and_closes_after_a_failed_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings = _capture_warnings(monkeypatch)
    backend = _Backend(close_failure=RuntimeError("the server hung up"))

    with pytest.raises(ValueError, match="the block failed"):
        async with _held_tool_backend(backend, _ToolBackendKind.MCP) as entered:
            assert entered == "entered"
            raise ValueError("the block failed")

    assert backend.closed
    assert warnings == ["The MCP tool backend failed to close: RuntimeError"]


@pytest.mark.asyncio
async def test_a_held_backend_is_closed_with_the_error_that_ended_the_block() -> None:
    """A backend can abandon work nobody will read only if it is told the request failed."""
    backend = _Backend()
    failure = ValueError("the provider failed")

    with pytest.raises(ValueError):
        async with _held_tool_backend(backend, _ToolBackendKind.MCP):
            raise failure

    assert backend.closed_with is failure


@pytest.mark.asyncio
async def test_a_held_backend_is_closed_cleanly_after_a_block_that_finished() -> None:
    backend = _Backend()

    async with _held_tool_backend(backend, _ToolBackendKind.MCP):
        pass

    assert backend.closed
    assert backend.closed_with is None


async def _two_chunks() -> AsyncIterator[str]:
    yield "first"
    yield "second"


@pytest.mark.asyncio
async def test_a_stream_the_client_left_closes_its_backends_with_that_exit() -> None:
    backend = _Backend()
    stack = AsyncExitStack()
    await stack.enter_async_context(backend)
    stream = cast(AsyncGenerator[str], _stream_with_stack_cleanup(_two_chunks(), stack, _ToolBackendKind.MCP))

    assert await anext(stream) == "first"
    await stream.aclose()

    assert isinstance(backend.closed_with, GeneratorExit)


@pytest.mark.asyncio
async def test_a_stream_that_finished_closes_its_backends_cleanly() -> None:
    backend = _Backend()
    stack = AsyncExitStack()
    await stack.enter_async_context(backend)

    assert [chunk async for chunk in _stream_with_stack_cleanup(_two_chunks(), stack, _ToolBackendKind.MCP)] == [
        "first",
        "second",
    ]
    assert backend.closed
    assert backend.closed_with is None
