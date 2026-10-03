"""Unit coverage for closing a streamed request's tool backend."""

from __future__ import annotations

import asyncio

import pytest

from gateway.api.routes._pipeline import _close_tool_backend, _held_tool_backend, _ToolBackendKind


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

    async def __aenter__(self) -> str:
        return "entered"

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True
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
