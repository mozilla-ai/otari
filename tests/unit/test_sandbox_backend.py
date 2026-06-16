"""Unit tests for `SandboxBackend`.

Mocks the HTTP layer with `respx` so the suite needs no sandbox container.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from gateway.services.sandbox_backend import (
    CODE_EXECUTION_TOOL_NAME,
    SandboxBackend,
    SandboxNotReachableError,
)


class _MockTransport(httpx.AsyncBaseTransport):
    """Tiny in-process httpx transport that routes requests to a handler dict."""

    def __init__(self, handlers: dict[tuple[str, str], httpx.Response | Exception]) -> None:
        self._handlers = handlers
        self.captured: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured.append(request)
        key = (request.method, request.url.path)
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        if isinstance(handler, Exception):
            raise handler
        return handler


def _patched_async_client(handlers: dict[tuple[str, str], Any], monkeypatch: pytest.MonkeyPatch) -> _MockTransport:
    transport = _MockTransport(handlers)
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return transport


@pytest.mark.asyncio
async def test_creates_session_on_enter_and_destroys_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "sbx_abc"}),
            ("DELETE", "/sessions/sbx_abc"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with SandboxBackend(sandbox_url="http://sandbox:8080") as backend:
        assert backend.owns_tool(CODE_EXECUTION_TOOL_NAME)

    methods_and_paths = [(r.method, r.url.path) for r in transport.captured]
    assert ("POST", "/sessions") in methods_and_paths
    assert ("DELETE", "/sessions/sbx_abc") in methods_and_paths


@pytest.mark.asyncio
async def test_call_tool_dispatches_code_to_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shape mirrors what the sandbox actually returns — see
    # https://github.com/mozilla-ai/otari-sandbox-container/blob/main/sandbox/models.py
    # ``result_block.content`` is a single ``CodeExecutionResultContent``
    # object, not a list.
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "42\n",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with SandboxBackend(sandbox_url="http://sandbox:8080") as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(6 * 7)"})

    assert "stdout:" in result
    assert "42" in result
    exec_request = next(r for r in transport.captured if r.url.path == "/sessions/s1/exec")
    body = exec_request.read().decode()
    assert "print(6 * 7)" in body


@pytest.mark.asyncio
async def test_call_tool_surfaces_stderr_and_nonzero_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "",
            "stderr": "NameError: name 'foo' is not defined\n",
            "return_code": 1,
            "content": [],
        },
    }
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with SandboxBackend(sandbox_url="http://sandbox:8080") as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(foo)"})

    # Non-zero return_code or stderr-only output is marked as [tool error]
    # so the model gets a clear failure signal.
    assert result.startswith("[tool error]")
    assert "stderr" in result
    assert "NameError" in result
    assert "return_code: 1" in result


@pytest.mark.asyncio
async def test_stderr_only_treated_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``return_code=0`` with stderr-only output still surfaces as a tool error.

    Some runners emit warnings via stderr without a non-zero exit code; the
    model should still see them as failure-shaped so it can recover.
    """
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "",
            "stderr": "DeprecationWarning: ...",
            "return_code": 0,
            "content": [],
        },
    }
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with SandboxBackend(sandbox_url="http://sandbox:8080") as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})

    assert result.startswith("[tool error]")
    assert "DeprecationWarning" in result


@pytest.mark.asyncio
async def test_enter_raises_when_sandbox_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("POST", "/sessions"): httpx.ConnectError("connection refused")},
        monkeypatch,
    )

    with pytest.raises(SandboxNotReachableError, match="failed to create"):
        async with SandboxBackend(sandbox_url="http://sandbox:8080"):
            pass


@pytest.mark.asyncio
async def test_owns_only_code_execution() -> None:
    backend = SandboxBackend(sandbox_url="http://sandbox:8080")
    assert backend.owns_tool(CODE_EXECUTION_TOOL_NAME)
    assert not backend.owns_tool("now_utc")
    assert not backend.owns_tool("anything_else")


@pytest.mark.asyncio
async def test_openai_tools_advertises_code_execution() -> None:
    backend = SandboxBackend(sandbox_url="http://sandbox:8080")
    tools = backend.openai_tools
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == CODE_EXECUTION_TOOL_NAME
    assert "code" in tools[0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_purpose_hint_is_emitted() -> None:
    backend = SandboxBackend(sandbox_url="http://sandbox:8080")
    hints = backend.purpose_hints()
    assert len(hints) == 1
    assert hints[0][0] == CODE_EXECUTION_TOOL_NAME


@pytest.mark.asyncio
async def test_auth_token_forwarded_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``auth_token`` is set, every sandbox call carries Authorization: Bearer.

    This lets the platform-hosted /v1/sandbox proxy authenticate the caller's
    workspace token and derive tenancy + per-workspace code-exec policy from it.
    """
    result_block = {"type": "code_execution_tool_result", "content": {"stdout": "ok"}}
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "sbx_abc"}),
            ("POST", "/sessions/sbx_abc/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/sbx_abc"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with SandboxBackend(
        sandbox_url="http://sandbox:8080",
        auth_token="tk_workspace_token",  # noqa: S106 — test fixture, not a real secret
    ) as backend:
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(1)"})

    # Every request to the backend (create / exec / delete) carries the header.
    assert transport.captured, "expected at least one request"
    for request in transport.captured:
        assert request.headers["Authorization"] == "Bearer tk_workspace_token"


@pytest.mark.asyncio
async def test_no_auth_header_when_auth_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone exec-service backend (no auth_token) sends no Authorization header."""
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "sbx_abc"}),
            ("DELETE", "/sessions/sbx_abc"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with SandboxBackend(sandbox_url="http://sandbox:8080"):
        pass

    assert transport.captured, "expected at least one request"
    for request in transport.captured:
        assert "Authorization" not in request.headers
