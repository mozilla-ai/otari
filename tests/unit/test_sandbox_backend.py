"""Unit tests for `SandboxBackend`.

Mocks the HTTP layer so the suite needs no sandbox container. The backend is
driven through the protocol adapter, which is the one that speaks HTTP; what
these cover is the half above the port, and the adapter's own handling of the
contract.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from gateway.adapters.code_execution_adapter import ProtocolCodeExecutionAdapter
from gateway.services.code_execution import ContainerLease
from gateway.services.sandbox_backend import (
    CODE_EXECUTION_TOOL_NAME,
    SandboxBackend,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)


class _MockTransport(httpx.AsyncBaseTransport):
    """Tiny in-process httpx transport that routes requests to a handler dict."""

    def __init__(self, handlers: dict[tuple[str, str], httpx.Response | Exception]) -> None:
        self._handlers = handlers
        self.captured: list[httpx.Request] = []
        self.closed = False

    async def aclose(self) -> None:
        # httpx closes its transport with the client, which is how these tests
        # see that the adapter released the connection it opened.
        self.closed = True

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured.append(request)
        key = (request.method, request.url.path)
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        if isinstance(handler, Exception):
            raise handler
        return handler


SANDBOX_URL = "http://sandbox:8080"


def _sandbox(**kwargs: Any) -> SandboxBackend:
    """A backend over the protocol adapter, which is the half these tests mock."""
    return SandboxBackend(port=ProtocolCodeExecutionAdapter(SANDBOX_URL), **kwargs)


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

    async with _sandbox() as backend:
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

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(6 * 7)"})

    assert "stdout:" in result
    assert "42" in result
    exec_request = next(r for r in transport.captured if r.url.path == "/sessions/s1/exec")
    body = exec_request.read().decode()
    assert "print(6 * 7)" in body


@pytest.mark.asyncio
async def test_exec_read_timeout_exceeds_execution_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exec POST's read timeout must be larger than the sandbox's exec budget.

    Regression test for #269: the client default equals ``timeout_seconds``, so a
    legitimate near-max execution would trip the client read timeout at the same
    instant the sandbox finishes, surfacing as a spurious ``SandboxNotReachableError``
    instead of a result. The exec request carries its own longer timeout as headroom.
    """
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {"type": "code_execution_result", "stdout": "ok\n", "stderr": "", "return_code": 0, "content": []},
    }
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    timeout_s = 30.0
    async with _sandbox(timeout_s=timeout_s) as backend:
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(1)"})

    exec_request = next(r for r in transport.captured if r.url.path == "/sessions/s1/exec")
    granted_budget = json.loads(exec_request.read().decode())["timeout_seconds"]
    read_timeout = exec_request.extensions["timeout"]["read"]
    assert read_timeout > granted_budget, (
        f"exec read timeout {read_timeout} must exceed the granted budget {granted_budget}"
    )


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

    async with _sandbox() as backend:
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

    async with _sandbox() as backend:
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
        async with _sandbox():
            pass


@pytest.mark.asyncio
async def test_owns_only_code_execution() -> None:
    backend = _sandbox()
    assert backend.owns_tool(CODE_EXECUTION_TOOL_NAME)
    assert not backend.owns_tool("now_utc")
    assert not backend.owns_tool("anything_else")


@pytest.mark.asyncio
async def test_openai_tools_advertises_code_execution() -> None:
    backend = _sandbox()
    tools = backend.openai_tools
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == CODE_EXECUTION_TOOL_NAME
    assert "code" in tools[0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_purpose_hint_is_emitted() -> None:
    backend = _sandbox()
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
        port=ProtocolCodeExecutionAdapter("http://sandbox:8080"),
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

    async with _sandbox():
        pass

    assert transport.captured, "expected at least one request"
    for request in transport.captured:
        assert "Authorization" not in request.headers


# --- OTel tracing ------------------------------------------------------------


def _make_otel_provider() -> tuple[object, object]:
    """Return an (exporter, provider) pair with an in-memory span exporter."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


@pytest.mark.asyncio
async def test_call_tool_emits_span_with_code_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful code_execution call must emit a span named 'code_execution'
    with the code attribute set."""
    from opentelemetry import trace as otel_trace

    import gateway.services.sandbox_backend as sb_module

    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {"type": "code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0, "content": []},
    }
    exporter, provider = _make_otel_provider()
    original_provider = otel_trace.get_tracer_provider()
    otel_trace.set_tracer_provider(provider)  # type: ignore[arg-type]
    sb_module.tracer = provider.get_tracer(sb_module.__name__)  # type: ignore[attr-defined]

    try:
        _patched_async_client(
            {
                ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
                ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
                ("DELETE", "/sessions/s1"): httpx.Response(204),
            },
            monkeypatch,
        )

        async with _sandbox() as backend:
            await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(6 * 7)"})
    finally:
        otel_trace.set_tracer_provider(original_provider)
        sb_module.tracer = otel_trace.get_tracer(sb_module.__name__)

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(exporter, InMemorySpanExporter)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == CODE_EXECUTION_TOOL_NAME
    assert span.attributes is not None
    assert span.attributes["tool.type"] == "otari_code_execution"
    assert span.attributes["code_execution.code_size"] == len("print(6 * 7)")
    assert span.attributes["code_execution.backend"] == "http://sandbox:8080"


@pytest.mark.asyncio
async def test_call_tool_span_records_error_on_sandbox_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the sandbox exec endpoint fails the span must record the exception
    and have an ERROR status."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import StatusCode

    import gateway.services.sandbox_backend as sb_module

    exporter, provider = _make_otel_provider()
    original_provider = otel_trace.get_tracer_provider()
    otel_trace.set_tracer_provider(provider)  # type: ignore[arg-type]
    sb_module.tracer = provider.get_tracer(sb_module.__name__)  # type: ignore[attr-defined]

    try:
        _patched_async_client(
            {
                ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
                ("POST", "/sessions/s1/exec"): httpx.ConnectError("connection refused"),
                ("DELETE", "/sessions/s1"): httpx.Response(204),
            },
            monkeypatch,
        )

        from gateway.services.sandbox_backend import SandboxNotReachableError

        async with _sandbox() as backend:
            with pytest.raises(SandboxNotReachableError):
                await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})
    finally:
        otel_trace.set_tracer_provider(original_provider)
        sb_module.tracer = otel_trace.get_tracer(sb_module.__name__)

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(exporter, InMemorySpanExporter)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == CODE_EXECUTION_TOOL_NAME
    assert span.status.status_code == StatusCode.ERROR
    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1
    attrs = exception_events[0].attributes
    assert attrs is not None
    assert attrs["exception.type"] == "httpx.ConnectError"


@pytest.mark.asyncio
async def test_call_tool_span_error_status_on_tool_error_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the sandbox returns a non-zero return_code the span status must be ERROR."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import StatusCode

    import gateway.services.sandbox_backend as sb_module

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
    exporter, provider = _make_otel_provider()
    original_provider = otel_trace.get_tracer_provider()
    otel_trace.set_tracer_provider(provider)  # type: ignore[arg-type]
    sb_module.tracer = provider.get_tracer(sb_module.__name__)  # type: ignore[attr-defined]

    try:
        _patched_async_client(
            {
                ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
                ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
                ("DELETE", "/sessions/s1"): httpx.Response(204),
            },
            monkeypatch,
        )

        async with _sandbox() as backend:
            result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(foo)"})
    finally:
        otel_trace.set_tracer_provider(original_provider)
        sb_module.tracer = otel_trace.get_tracer(sb_module.__name__)

    assert result.startswith("[tool error]")

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(exporter, InMemorySpanExporter)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# ----- contract conformance (docs/code-execution-protocol.md) -----


def _exec_handlers(result_block: Any, monkeypatch: pytest.MonkeyPatch) -> _MockTransport:
    return _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(201, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )


@pytest.mark.asyncio
async def test_unknown_result_block_type_still_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool kind added to the contract later must not break an older gateway.

    The extension policy allows new result-block types additively, so `type` is
    read as an opaque string rather than validated against a closed set.
    """
    _exec_handlers(
        {
            "type": "future_code_execution_tool_result",
            "tool_use_id": "t1",
            "content": {"type": "code_execution_result", "stdout": "42\n", "return_code": 0},
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(42)"})

    assert result == "stdout:\n42\n"


@pytest.mark.asyncio
async def test_unrecognised_fields_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backends may return fields this gateway does not know; they are dropped."""
    _exec_handlers(
        {
            "type": "code_execution_tool_result",
            "tool_use_id": "t1",
            "cpu_milliseconds": 12,
            "content": {
                "type": "code_execution_result",
                "stdout": "ok",
                "return_code": 0,
                "sandbox_node": "pod-7",
            },
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})

    assert result == "stdout:\nok"


@pytest.mark.asyncio
async def test_file_refs_are_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _exec_handlers(
        {
            "type": "code_execution_tool_result",
            "tool_use_id": "t1",
            "content": {
                "type": "code_execution_result",
                "stdout": "done",
                "return_code": 0,
                "content": [
                    {"type": "code_execution_output", "file_id": "f1", "filename": "chart.png"},
                    {"type": "code_execution_output", "file_id": "f2"},
                ],
            },
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "savefig()"})

    assert "files: chart.png, ?" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="result_block-absent"),
        pytest.param({"result_block": "not-an-object"}, id="result_block-not-an-object"),
        pytest.param({"result_block": {"type": "code_execution_tool_result"}}, id="content-absent"),
        pytest.param(
            {"result_block": {"type": "code_execution_tool_result", "content": ["a", "b"]}},
            id="content-is-a-list",
        ),
    ],
)
async def test_malformed_exec_response_raises(body: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A response that violates the contract yields no usable result, so it raises.

    `content` as a list is called out explicitly: it is the shape a consumer
    reaches for when it mistakes the payload for Anthropic's mixed content-block
    list, and a backend that emits it is not conforming.
    """
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(201, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json=body),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        with pytest.raises(SandboxNotReachableError, match="sandbox exec failed"):
            await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})


@pytest.mark.asyncio
async def test_enter_raises_when_session_handle_lacks_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("POST", "/sessions"): httpx.Response(201, json={"created_at": 1.0})},
        monkeypatch,
    )

    with pytest.raises(SandboxNotReachableError, match="failed to create"):
        async with _sandbox():
            pass


@pytest.mark.asyncio
async def test_documented_exec_response_shape_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full response documented in docs/code-execution-protocol.md.

    Pinned verbatim against the spec's example, envelope included, so the
    published contract and this client cannot drift apart silently.
    """
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(201, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(
                200,
                json={
                    "tool_use_id": "srvtoolu_abc",
                    "execution_time_ms": 84,
                    "result_block": {
                        "type": "code_execution_tool_result",
                        "tool_use_id": "srvtoolu_abc",
                        "content": {
                            "type": "code_execution_result",
                            "stdout": "3.14\n",
                            "stderr": "",
                            "return_code": 0,
                            "content": [
                                {
                                    "type": "code_execution_output",
                                    "file_id": "file_1",
                                    "filename": "chart.png",
                                }
                            ],
                        },
                    },
                },
            ),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(3.14)"})

    assert result == "stdout:\n3.14\n\nfiles: chart.png"


@pytest.mark.asyncio
async def test_schema_violation_message_omits_the_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A contract violation must not carry program output into logs or spans.

    `stdout` here holds the kind of thing sandboxed code can emit (a credential
    it was handed). The error names the offending field, never its value.
    """
    secret = "sk-live-do-not-log-this"
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(201, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(
                200,
                json={
                    "result_block": {
                        "type": "code_execution_tool_result",
                        # `content` must be an object; a list violates the contract.
                        "content": [{"stdout": secret}],
                    }
                },
            ),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        with pytest.raises(SandboxNotReachableError) as excinfo:
            await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert secret not in rendered
    assert "result_block.content" in str(excinfo.value)


@pytest.mark.asyncio
async def test_session_handle_violation_message_omits_the_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-live-do-not-log-this"
    _patched_async_client(
        {("POST", "/sessions"): httpx.Response(201, json={"session_id": {"nested": secret}})},
        monkeypatch,
    )

    with pytest.raises(SandboxNotReachableError) as excinfo:
        async with _sandbox():
            pass

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert secret not in rendered
    assert "session_id" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            {"stdout": "42\n", "stderr": None, "return_code": 0},
            "stdout:\n42\n",
            id="stderr-null",
        ),
        pytest.param(
            {"stdout": None, "stderr": "boom", "return_code": 1},
            "[tool error] stderr:\nboom\nreturn_code: 1",
            id="stdout-null",
        ),
        pytest.param(
            {"stdout": "done", "return_code": 0, "content": None},
            "stdout:\ndone",
            id="file-refs-null",
        ),
        pytest.param(
            {"stdout": "file body", "return_code": 0, "content": "inline text"},
            "stdout:\nfile body",
            id="file-refs-not-a-list",
        ),
        pytest.param(
            {"stdout": "ok", "return_code": 0, "content": [{"filename": None}, "bare-id"]},
            "stdout:\nok\nfiles: ?",
            id="file-ref-unnameable",
        ),
    ],
)
async def test_renderable_fields_absorb_unusable_values(
    content: dict[str, Any], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload carrying usable output must not be discarded over a cosmetic field.

    Each case is one the pre-typing renderer absorbed (`... or ""`, an
    `isinstance` guard, a per-entry filter). Rejecting them would turn a run
    that produced real output into a failed tool call, and for the eager-open
    path into a 502.
    """
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(201, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(
                200,
                json={"result_block": {"type": "code_execution_tool_result", "content": content}},
            ),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            {"stdout": "", "stderr": {"message": "boom"}, "return_code": 0},
            "[tool error] stderr:\n{'message': 'boom'}",
            id="stderr-only-is-an-object",
        ),
        pytest.param(
            {"stdout": "ok", "stderr": {"message": "warn"}, "return_code": 0},
            "stdout:\nok\nstderr:\n{'message': 'warn'}",
            id="stderr-object-alongside-stdout",
        ),
        pytest.param(
            {"stdout": ["a", "b"], "return_code": 0},
            "stdout:\n['a', 'b']",
            id="stdout-is-a-list",
        ),
        pytest.param(
            {"stdout": "ok", "return_code": 0, "content": [{"filename": {"nested": 1}}]},
            "stdout:\nok\nfiles: {'nested': 1}",
            id="filename-is-an-object",
        ),
    ],
)
async def test_structured_values_in_render_only_fields_keep_the_signal(
    content: dict[str, Any], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structure where text was expected is rendered, not dropped.

    The first case is why stringifying beats discarding. Blanking a structured
    `stderr` would leave a stderr-only run with nothing to render, so it would
    come back `(no output)` with no `[tool error]` marker and bill as a
    successful call, which is the failure the error-variant exclusion in
    docs/code-execution-protocol.md exists to prevent.

    The second case pins the surrounding semantics: stderr *alongside* stdout at
    `return_code: 0` is not error-shaped either way, so only the stderr-only
    shape distinguishes the two coercions.
    """
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(201, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(
                200,
                json={"result_block": {"type": "code_execution_tool_result", "content": content}},
            ),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})

    assert result == expected


# ---------------------------------------------------------------------------
# The sandbox image and the exposed tool set (#740)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_omits_image_when_none_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that names no image sends the body it always sent.

    The whole compatibility claim for the additive ``image`` field rests on
    this: a backend built before the field existed must see an empty object.
    """
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "sbx_abc"}),
            ("DELETE", "/sessions/sbx_abc"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox():
        pass

    create = next(r for r in transport.captured if r.method == "POST")
    assert json.loads(create.content) == {}


@pytest.mark.asyncio
async def test_create_session_sends_the_pinned_image(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "sbx_abc"}),
            ("DELETE", "/sessions/sbx_abc"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox(image="mzdotai/otari-sandbox-container:latest"):
        pass

    create = next(r for r in transport.captured if r.method == "POST")
    assert json.loads(create.content) == {"image": "mzdotai/otari-sandbox-container:latest"}


@pytest.mark.asyncio
async def test_allowed_tools_including_code_execution_changes_nothing() -> None:
    backend = SandboxBackend(
        port=ProtocolCodeExecutionAdapter("http://sandbox:8080"),
        allowed_tools=frozenset({CODE_EXECUTION_TOOL_NAME, "bash_code_execution"}),
    )
    assert backend.owns_tool(CODE_EXECUTION_TOOL_NAME)
    assert [tool["function"]["name"] for tool in backend.openai_tools] == [CODE_EXECUTION_TOOL_NAME]
    assert backend.purpose_hints()


@pytest.mark.asyncio
async def test_allowed_tools_excluding_code_execution_offers_nothing() -> None:
    """An allow-list the backend's one tool is not in leaves it with nothing to offer.

    Admission refuses this case before a backend is built, so the assertion is
    about the two halves agreeing rather than about a path a request takes: what
    is advertised and what is dispatched come from one predicate, so a future
    caller that skips the 403 still cannot have the tool run.
    """
    backend = SandboxBackend(
        port=ProtocolCodeExecutionAdapter("http://sandbox:8080"),
        allowed_tools=frozenset({"bash_code_execution"}),
    )
    assert not backend.owns_tool(CODE_EXECUTION_TOOL_NAME)
    assert backend.openai_tools == []
    assert backend.purpose_hints() == []

    with pytest.raises(KeyError):
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "1"})


@pytest.mark.asyncio
async def test_served_tool_names_is_what_the_backend_actually_advertises() -> None:
    """The policy layer's idea of what this deployment serves must be the backend's.

    ``SERVED_TOOL_NAMES`` is the set a workspace tool list is intersected
    against, in two places: ``_require_runnable_tools`` refuses a write that
    shares nothing with it, and ``prepare_gateway_tools`` refuses a request the
    same way. Both are only correct while the tuple names what a backend really
    offers. Growing it without teaching :class:`SandboxBackend` the new kind
    would admit a policy naming only that kind, then hand the model a backend
    advertising nothing, which is the silently-successful request both guards
    exist to prevent.
    """
    from gateway.services.tenancy.workspace_code_execution_policy_service import SERVED_TOOL_NAMES

    backend = _sandbox()
    advertised = tuple(tool["function"]["name"] for tool in backend.openai_tools)

    assert advertised == SERVED_TOOL_NAMES


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after, expected", [("15", "15"), (None, None), ("invalid", None), ("9999999", None)])
async def test_session_503_preserves_retry_hint_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, retry_after: str | None, expected: str | None
) -> None:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    transport = _patched_async_client(
        {("POST", "/sessions"): httpx.Response(503, headers=headers, json={"detail": "private internals"})},
        monkeypatch,
    )
    backend = _sandbox()
    with pytest.raises(SandboxUnavailableError) as caught:
        await backend.__aenter__()
    assert caught.value.retry_after == expected
    assert "private internals" not in str(caught.value)
    assert transport.closed, "the adapter left its client open after a refused session"
    assert len(transport.captured) == 1


@pytest.mark.asyncio
async def test_exec_503_preserves_retry_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(503, headers={"Retry-After": "15"}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    async with _sandbox() as backend:
        with pytest.raises(SandboxUnavailableError) as caught:
            await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(42)"})
    assert caught.value.retry_after == "15"


class _FakeFiles:
    """A stand-in for ``SandboxFileBridge``: inputs to seed, outputs it was handed."""

    def __init__(self, inputs: list[Any], *, max_output_bytes: int = 1 << 20, max_output_files: int = 20) -> None:
        self.inputs = inputs
        self.max_output_bytes = max_output_bytes
        self.max_output_files = max_output_files
        self.stored: list[tuple[str, bytes]] = []
        # Streams the backend started and abandoned, as a store would see them.
        self.abandoned: list[str] = []

    async def read_input(self, staged: Any) -> bytes:
        return b"a,b\n1,2\n"

    async def store_output(self, filename: str, chunks: Any) -> str | None:
        data = bytearray()
        try:
            async for chunk in chunks:
                data.extend(chunk)
        except BaseException:
            self.abandoned.append(filename)
            raise
        if not data:
            return None
        self.stored.append((filename, bytes(data)))
        return f"file-{len(self.stored)}"


def _staged(file_id: str = "file-csv", filename: str = "data.csv") -> Any:
    from gateway.services.file_service import StagedFile

    return StagedFile(file_id, filename, "text/csv", f"x/{file_id}")


@pytest.mark.asyncio
async def test_staged_inputs_are_seeded_before_the_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/files"): httpx.Response(201, json={"path": "data.csv", "size": 8}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([_staged()])
    async with _sandbox(files=files):
        pass

    put = next(r for r in transport.captured if r.method == "POST" and r.url.path == "/sessions/s1/files")
    body = put.read()
    assert b'filename="data.csv"' in body
    assert b"a,b\n1,2\n" in body
    assert b'name="path"' in body


@pytest.mark.asyncio
async def test_refused_seed_is_terminal_and_releases_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/files"): httpx.Response(413, json={"error": "too large"}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    with pytest.raises(SandboxNotReachableError, match="file-csv"):
        async with _sandbox(files=_FakeFiles([_staged()])):
            pass
    assert ("DELETE", "/sessions/s1") in [(r.method, r.url.path) for r in transport.captured]


@pytest.mark.asyncio
async def test_produced_files_are_fetched_stored_and_named_with_file_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "saved\n",
            "stderr": "",
            "return_code": 0,
            "content": [{"type": "code_execution_output", "file_id": "sbx-1", "filename": "chart.png"}],
        },
    }
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("GET", "/sessions/s1/files"): httpx.Response(200, content=b"\x89PNG"),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([])
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "plt.savefig('chart.png')"})

    assert files.stored == [("chart.png", b"\x89PNG")]
    assert "chart.png (file_id: file-1)" in result


@pytest.mark.asyncio
async def test_unfetchable_output_is_still_named_and_does_not_fail_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "ok\n",
            "stderr": "",
            "return_code": 0,
            "content": [{"type": "code_execution_output", "file_id": "sbx-1", "filename": "out.csv"}],
        },
    }
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("GET", "/sessions/s1/files"): httpx.Response(404, json={"error": "gone"}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([])
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    assert files.stored == []
    assert "files: out.csv" in result
    assert "file_id" not in result


@pytest.mark.asyncio
async def test_no_bridge_leaves_outputs_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "content": [{"type": "code_execution_output", "file_id": "sbx-1", "filename": "a.txt"}],
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
    async with _sandbox() as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})
    assert result == "files: a.txt"
    assert all(r.url.path != "/sessions/s1/files" for r in transport.captured)


@pytest.mark.asyncio
async def test_executions_are_kept_in_order_and_drained_by_take(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a loop minting native result blocks reads: the code and the structured result."""
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {"type": "code_execution_result", "stdout": "1\n", "stderr": "", "return_code": 0, "content": []},
    }
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        assert backend.container_id.startswith("otari_cntr_")
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(1)"})
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(2)"})
        executions = backend.take_executions()
        assert [execution.code for execution in executions] == ["print(1)", "print(2)"]
        assert executions[0].result is not None
        assert executions[0].result.content.stdout == "1\n"
        # Drained: a later round cannot claim an earlier round's executions.
        assert backend.take_executions() == []


@pytest.mark.asyncio
async def test_an_exec_that_never_answered_is_kept_without_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(500),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )

    async with _sandbox() as backend:
        with pytest.raises(SandboxNotReachableError):
            await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "print(1)"})
        executions = backend.take_executions()

    assert len(executions) == 1
    assert executions[0].code == "print(1)"
    assert executions[0].result is None


@pytest.mark.asyncio
async def test_an_execution_carries_the_stored_ids_of_the_files_it_produced(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a native block announces: the ``/v1/files`` id, never the sandbox's own."""
    result_block = {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "content": [{"type": "code_execution_output", "file_id": "sbx-1", "filename": "chart.png"}],
        },
    }
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": result_block}),
            ("GET", "/sessions/s1/files"): httpx.Response(200, content=b"\x89PNG"),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    async with _sandbox(files=_FakeFiles([])) as backend:
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "plt.savefig('chart.png')"})
        execution = backend.take_executions()[0]

    assert execution.file_ids == {"chart.png": "file-1"}


def _result_block_naming(filename: str) -> dict[str, Any]:
    return {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {
            "type": "code_execution_result",
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "content": [{"type": "code_execution_output", "file_id": "sbx-1", "filename": filename}],
        },
    }


@pytest.mark.asyncio
async def test_an_output_declared_over_the_cap_is_refused_before_it_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CountingStream(httpx.AsyncByteStream):
        reads = 0

        async def __aiter__(self) -> Any:
            _CountingStream.reads += 1
            yield b"x" * 64

    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _result_block_naming("big.bin")}),
            ("GET", "/sessions/s1/files"): httpx.Response(
                200, headers={"content-length": "64"}, stream=_CountingStream()
            ),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([], max_output_bytes=16)
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    assert files.stored == []
    assert _CountingStream.reads == 0
    assert "files: big.bin" in result


@pytest.mark.asyncio
async def test_an_output_that_grows_past_the_cap_is_abandoned_mid_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EndlessStream(httpx.AsyncByteStream):
        chunks = 0

        async def __aiter__(self) -> Any:
            while True:
                _EndlessStream.chunks += 1
                yield b"x" * 8

    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _result_block_naming("big.bin")}),
            # No Content-Length: the cap has to hold on the bytes as they arrive.
            ("GET", "/sessions/s1/files"): httpx.Response(200, stream=_EndlessStream()),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([], max_output_bytes=32)
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    assert files.stored == []
    # Read just past the cap and no further: 32 bytes is four chunks, the fifth trips it.
    assert _EndlessStream.chunks == 5
    assert "file_id" not in result


def _empty_result_block(stdout: str = "saved\n") -> dict[str, Any]:
    """A result block that names no files, as the reference container returns."""
    return {
        "type": "code_execution_tool_result",
        "tool_use_id": "t1",
        "content": {"type": "code_execution_result", "stdout": stdout, "stderr": "", "return_code": 0, "content": []},
    }


def _listing(*entries: tuple[str, int | None, float]) -> dict[str, Any]:
    return {"files": [{"path": p, "size_bytes": s, "mime_type": None, "modified_at": m} for p, s, m in entries]}


_Handlers = dict[tuple[str, str], httpx.Response | list[httpx.Response]]


class _SequenceTransport(httpx.AsyncBaseTransport):
    """Like ``_MockTransport``, but a handler may be a list answered in order."""

    def __init__(self, handlers: _Handlers) -> None:
        self._handlers = handlers
        self.captured: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured.append(request)
        handler = self._handlers.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, json={"error": "no handler"})
        if isinstance(handler, list):
            return handler.pop(0) if len(handler) > 1 else handler[0]
        return handler


def _patched_sequence_client(handlers: _Handlers, monkeypatch: pytest.MonkeyPatch) -> _SequenceTransport:
    transport = _SequenceTransport(handlers)
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return transport


@pytest.mark.asyncio
async def test_a_file_the_block_does_not_name_is_found_by_the_workspace_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_sequence_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("POST", "/sessions/s1/files"): httpx.Response(201, json={"path": "data.csv", "size": 8}),
            # Listed once after seeding (the input only), once after the call (the output too).
            ("GET", "/sessions/s1/files/list"): [
                httpx.Response(200, json=_listing(("data.csv", 8, 1.0))),
                httpx.Response(200, json=_listing(("data.csv", 8, 1.0), ("out.txt", 5, 2.0))),
            ],
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _empty_result_block()}),
            ("GET", "/sessions/s1/files"): httpx.Response(200, content=b"hello"),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([_staged()])
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "open('out.txt','w').write('hello')"})
        execution = backend.take_executions()[0]

    # Only the new file was fetched: the unchanged seeded input was not re-read.
    fetched = [
        r.url.params.get("path") for r in transport.captured if r.method == "GET" and r.url.path.endswith("/files")
    ]
    assert fetched == ["out.txt"]
    assert files.stored == [("out.txt", b"hello")]
    assert "files: out.txt (file_id: file-1)" in result
    assert execution.file_ids == {"out.txt": "file-1"}


@pytest.mark.asyncio
async def test_the_diff_moves_forward_so_a_later_call_collects_only_its_own_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_sequence_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("GET", "/sessions/s1/files/list"): [
                httpx.Response(200, json=_listing()),
                httpx.Response(200, json=_listing(("a.txt", 1, 1.0))),
                # a.txt rewritten (new stamp) and b.txt new: both are this call's.
                httpx.Response(200, json=_listing(("a.txt", 2, 3.0), ("b.txt", 1, 3.0))),
            ],
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _empty_result_block()}),
            ("GET", "/sessions/s1/files"): httpx.Response(200, content=b"x"),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([])
    async with _sandbox(files=files) as backend:
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "one"})
        await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "two"})
        first, second = backend.take_executions()

    assert list(first.file_ids) == ["a.txt"]
    assert sorted(second.file_ids) == ["a.txt", "b.txt"]


@pytest.mark.asyncio
async def test_a_backend_without_list_files_still_collects_what_the_block_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            # No /files/list handler: the mock answers 404, as a backend without the operation would.
            ("POST", "/sessions/s1/exec"): httpx.Response(
                200, json={"result_block": _result_block_naming("chart.png")}
            ),
            ("GET", "/sessions/s1/files"): httpx.Response(200, content=b"\x89PNG"),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([])
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    assert files.stored == [("chart.png", b"\x89PNG")]
    assert "chart.png (file_id: file-1)" in result


@pytest.mark.asyncio
async def test_a_call_stores_at_most_the_configured_number_of_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_sequence_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("GET", "/sessions/s1/files/list"): [
                httpx.Response(200, json=_listing()),
                httpx.Response(200, json=_listing(("a.txt", 1, 1.0), ("b.txt", 1, 1.0), ("c.txt", 1, 1.0))),
            ],
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _empty_result_block()}),
            ("GET", "/sessions/s1/files"): httpx.Response(200, content=b"x"),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([], max_output_files=2)
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    assert [name for name, _ in files.stored] == ["a.txt", "b.txt"]
    # The third is still named, so the model and the caller know it exists.
    assert "c.txt" in result
    assert "c.txt (file_id" not in result


@pytest.mark.asyncio
async def test_a_call_stores_at_most_the_configured_bytes_across_its_files(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TwentyBytes(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            yield b"x" * 10
            yield b"y" * 10

    _patched_sequence_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("GET", "/sessions/s1/files/list"): [
                httpx.Response(200, json=_listing()),
                # No declared size either, so nothing is known until the bytes arrive.
                httpx.Response(200, json=_listing(("a.bin", None, 1.0), ("b.bin", None, 1.0))),
            ],
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _empty_result_block()}),
            # No Content-Length: the budget has to hold on the bytes as they arrive.
            ("GET", "/sessions/s1/files"): httpx.Response(200, stream=_TwentyBytes()),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([], max_output_bytes=30)
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    # The first file fits (20 of 30); the second runs past what is left and is
    # abandoned mid-stream, which the store sees as a failed stream to clean up.
    assert files.stored == [("a.bin", b"x" * 10 + b"y" * 10)]
    assert files.abandoned == ["b.bin"]
    assert "a.bin (file_id: file-1)" in result
    assert "b.bin (file_id" not in result


@pytest.mark.asyncio
async def test_an_output_the_listing_already_says_is_too_big_is_never_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _patched_sequence_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("GET", "/sessions/s1/files/list"): [
                httpx.Response(200, json=_listing()),
                httpx.Response(200, json=_listing(("huge.bin", 10_000, 1.0))),
            ],
            ("POST", "/sessions/s1/exec"): httpx.Response(200, json={"result_block": _empty_result_block()}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    files = _FakeFiles([], max_output_bytes=30)
    async with _sandbox(files=files) as backend:
        result = await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": "x"})

    # Not fetched at all: an adapter whose provider hands a file over whole would
    # otherwise hold 10kB to discover it may store 30 bytes of it.
    assert files.stored == []
    assert not any(request.url.path == "/sessions/s1/files" for request in transport.captured)
    # Still named, so the model knows the run wrote it.
    assert "huge.bin" in result
    assert "huge.bin (file_id" not in result


class _RecordingPort:
    """A port that records the lease it was asked for and runs nothing."""

    label = "recording"

    def __init__(self) -> None:
        self.session_ttl_s: float | None = None

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        auth_token: str | None = None,
        resume: str | None = None,
        keep_alive_s: float | None = None,
    ) -> AsyncIterator[Any]:
        del image, timeout_s, auth_token, resume, keep_alive_s
        self.session_ttl_s = session_ttl_s
        yield SimpleNamespace(session_id="s-recording", holds_across_requests=True, discard=lambda: None)


@pytest.mark.asyncio
async def test_the_lease_outlasts_a_round_that_batches_calls() -> None:
    """The iteration cap bounds rounds, not code calls: a model may emit several
    in one round, and a lease sized for one per round is reclaimed mid-request."""
    port = _RecordingPort()
    async with SandboxBackend(port=port, timeout_s=10.0, max_executions=4):
        pass

    # 10s a call, four calls a round, four rounds, plus the 60s of slack the
    # seeding and collecting run in. Spelled out rather than compared against
    # one call per round, which the old sizing also satisfies through the slack.
    assert port.session_ttl_s == 10.0 * 4 * 4 + 60.0


@pytest.mark.asyncio
async def test_the_lease_is_capped_rather_than_asked_for_in_full() -> None:
    """A provider refuses a creation that asks to hold a sandbox past its plan's
    ceiling, so a generous estimate is clamped rather than sent."""
    port = _RecordingPort()
    async with SandboxBackend(port=port, timeout_s=600.0, max_executions=50):
        pass

    assert port.session_ttl_s == 3600.0


# --- holding a session across requests --------------------------------------------------


@pytest.mark.asyncio
async def test_a_kept_protocol_session_is_told_its_idle_timeout_and_not_destroyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _patched_async_client(
        {("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1", "idle_timeout_seconds": 600})},
        monkeypatch,
    )
    adapter = ProtocolCodeExecutionAdapter(SANDBOX_URL)
    async with adapter.open_session(timeout_s=30, session_ttl_s=300, keep_alive_s=600) as session:
        assert session.holds_across_requests
    create = next(r for r in transport.captured if r.method == "POST")
    # The contract's one lifetime hint, and the only way a kept session ends.
    assert json.loads(create.content)["idle_timeout_seconds"] == 600
    assert [r.method for r in transport.captured] == ["POST"], "a kept session must not be destroyed"


@pytest.mark.asyncio
async def test_a_backend_that_declines_the_idle_timeout_still_gets_its_session_destroyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``idle_timeout_seconds`` is a hint, and the handle reports what was kept.

    A backend that reports none will run no idle reclaim, so skipping the DELETE
    would leak the session with nothing left to end it.
    """
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    adapter = ProtocolCodeExecutionAdapter(SANDBOX_URL)
    async with adapter.open_session(timeout_s=30, session_ttl_s=300, keep_alive_s=600) as session:
        assert not session.holds_across_requests
    assert [r.method for r in transport.captured] == ["POST", "DELETE"]


@pytest.mark.asyncio
async def test_a_session_nothing_can_resume_is_never_leased_a_container_id() -> None:
    """The backend reports no container for a session the adapter will not hold."""

    class _NonHoldingPort(_RecordingPort):
        @asynccontextmanager
        async def open_session(self, **kwargs: Any) -> AsyncIterator[Any]:
            del kwargs
            yield SimpleNamespace(session_id="s1", holds_across_requests=False, discard=lambda: None)

    containers = _FakeContainers()
    async with SandboxBackend(port=_NonHoldingPort(), containers=containers) as backend:
        assert backend.lease is None
    assert containers.recorded == []


@pytest.mark.asyncio
async def test_a_session_the_request_cannot_use_is_discarded_and_leaves_no_lease() -> None:
    """A seed that fails releases the sandbox rather than holding one nobody was told about."""
    discarded: list[bool] = []

    class _DiscardablePort(_RecordingPort):
        @asynccontextmanager
        async def open_session(self, **kwargs: Any) -> AsyncIterator[Any]:
            del kwargs
            yield SimpleNamespace(
                session_id="s1",
                holds_across_requests=True,
                discard=lambda: discarded.append(True),
            )

    class _FailingFiles(_FakeFiles):
        async def read_input(self, staged: Any) -> bytes:
            raise OSError("no such blob")

    containers = _FakeContainers()
    backend = SandboxBackend(port=_DiscardablePort(), containers=containers, files=_FailingFiles(inputs=[_staged()]))
    with pytest.raises(SandboxNotReachableError):
        async with backend:
            pass  # pragma: no cover - enter raises

    assert discarded == [True]
    assert backend.lease is None
    assert containers.recorded == []


@pytest.mark.asyncio
async def test_a_released_protocol_session_sends_the_body_it_always_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {
            ("POST", "/sessions"): httpx.Response(200, json={"session_id": "s1"}),
            ("DELETE", "/sessions/s1"): httpx.Response(204),
        },
        monkeypatch,
    )
    async with _sandbox():
        pass
    create = next(r for r in transport.captured if r.method == "POST")
    assert "idle_timeout_seconds" not in json.loads(create.content)


@pytest.mark.asyncio
async def test_a_resumed_protocol_session_is_probed_not_created(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {("GET", "/sessions/s-old/files/list"): httpx.Response(200, json={"files": []})},
        monkeypatch,
    )
    adapter = ProtocolCodeExecutionAdapter(SANDBOX_URL)
    async with adapter.open_session(timeout_s=30, session_ttl_s=300, resume="s-old", keep_alive_s=600) as session:
        assert session.session_id == "s-old"
    assert [(r.method, r.url.path) for r in transport.captured] == [("GET", "/sessions/s-old/files/list")]


@pytest.mark.asyncio
async def test_resuming_a_session_the_backend_reclaimed_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("GET", "/sessions/s-old/files/list"): httpx.Response(404, json={"detail": "no such session"})},
        monkeypatch,
    )
    adapter = ProtocolCodeExecutionAdapter(SANDBOX_URL)
    with pytest.raises(SandboxSessionGoneError):
        async with adapter.open_session(timeout_s=30, session_ttl_s=300, resume="s-old"):
            pass


class _FakeContainers:
    """A registry double: fixed clocks, and it remembers what was recorded."""

    def __init__(self, idle_s: float = 600.0) -> None:
        self.idle_s = idle_s
        self.recorded: list[ContainerLease] = []
        self.released: list[str] = []

    def keep_alive_s(self, resumed: ContainerLease | None) -> float:
        return self.idle_s

    def lease(self, container_id: str, provider_session_id: str, *, resumed: ContainerLease | None) -> ContainerLease:
        now = datetime.now(UTC)
        return ContainerLease(
            container_id=container_id,
            provider="recording",
            provider_session_id=provider_session_id,
            expires_at=now + timedelta(seconds=self.idle_s),
            hard_expires_at=resumed.hard_expires_at if resumed else now + timedelta(hours=1),
        )

    async def record(self, lease: ContainerLease) -> None:
        self.recorded.append(lease)

    async def release(self, container_id: str) -> None:
        self.released.append(container_id)


class _HoldingPort(_RecordingPort):
    """Records the resume and keep-alive it was asked for too."""

    def __init__(self) -> None:
        super().__init__()
        self.resume: str | None = None
        self.keep_alive_s: float | None = None

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        auth_token: str | None = None,
        resume: str | None = None,
        keep_alive_s: float | None = None,
    ) -> AsyncIterator[Any]:
        del image, timeout_s, auth_token
        self.session_ttl_s = session_ttl_s
        self.resume = resume
        self.keep_alive_s = keep_alive_s
        yield SimpleNamespace(
            session_id=resume or "s-fresh",
            holds_across_requests=keep_alive_s is not None,
            discard=lambda: None,
        )


@pytest.mark.asyncio
async def test_a_fresh_lease_is_published_on_enter_and_recorded_on_exit() -> None:
    port = _HoldingPort()
    containers = _FakeContainers(idle_s=600.0)
    noted: list[ContainerLease] = []
    backend = SandboxBackend(port=port, containers=containers, on_lease=noted.append)

    async with backend:
        assert port.resume is None
        assert port.keep_alive_s == 600.0
        # Reported before any code runs, so a streamed response can carry it.
        assert noted and noted[0].container_id == backend.container_id
        assert noted[0].provider_session_id == "s-fresh"
        assert containers.recorded == []
    # Written after the port held the sandbox, never before.
    assert [lease.container_id for lease in containers.recorded] == [backend.container_id]


@pytest.mark.asyncio
async def test_a_resumed_lease_keeps_its_id_and_resumes_the_providers_session() -> None:
    port = _HoldingPort()
    containers = _FakeContainers()
    now = datetime.now(UTC)
    resumed = ContainerLease(
        container_id="otari_cntr_deadbeef",
        provider="recording",
        provider_session_id="s-old",
        expires_at=now + timedelta(minutes=5),
        hard_expires_at=now + timedelta(minutes=30),
    )
    async with SandboxBackend(port=port, container=resumed, containers=containers) as backend:
        assert backend.container_id == "otari_cntr_deadbeef"
        assert port.resume == "s-old"
    assert containers.recorded[0].container_id == "otari_cntr_deadbeef"
    # The hard clock is the one the first lease set; a resume never restarts it.
    assert containers.recorded[0].hard_expires_at == resumed.hard_expires_at


@pytest.mark.asyncio
async def test_without_a_registry_nothing_is_held_or_reported() -> None:
    port = _HoldingPort()
    async with SandboxBackend(port=port) as backend:
        assert port.keep_alive_s is None
        assert backend.lease is None
