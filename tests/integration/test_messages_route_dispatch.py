"""Route-level tests for the /v1/messages endpoint wiring.

These complement :mod:`tests.unit.test_mcp_loop_messages` (which tests the
Anthropic tool loop in isolation) by exercising the FastAPI route handler:
tool extraction, mutual-exclusivity validation, error-body mapping to the
Anthropic shape, and the per-tool dispatch into the right backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
    TextDelta,
)
from fastapi.testclient import TestClient


def _text_response(text: str = "ok") -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=5,
            output_tokens=2,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


# ---------- plain amessages (no gateway tools) ----------


def test_no_tools_falls_through_to_plain_amessages(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A vanilla request with no gateway-managed tools hits ``amessages`` directly,
    bypassing the tool loop entirely.
    """
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("hi")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["content"][0]["text"] == "hi"
    # Direct amessages call: tool-loop-only fields shouldn't appear.
    assert "tools" not in captured


def test_gateway_internal_fields_are_stripped_from_upstream_kwargs(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``mcp_servers`` / ``tools_header`` / ``max_tool_iterations`` are gateway-
    only knobs; Anthropic rejects unknown kwargs with a 400. Stripping must
    happen at the boundary so they never reach ``amessages``.
    """
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "tools_header": "Tools available:",
                "max_tool_iterations": 3,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    for field in ("mcp_servers", "mcp_server_ids", "tools_header", "max_tool_iterations", "user"):
        assert field not in captured, f"gateway-internal field {field!r} leaked to upstream"


def test_user_supplied_openai_shape_tools_get_converted_to_anthropic(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A caller can mix gateway-managed tools and their own function tools; the
    OpenAI-shape ones must be converted to ``{name, description, input_schema}``
    before the call reaches Anthropic.
    """
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "do it"}],
                "max_tokens": 100,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "look stuff up",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    forwarded_tools = captured.get("tools")
    assert forwarded_tools is not None
    assert forwarded_tools[0]["name"] == "lookup"
    assert forwarded_tools[0]["input_schema"] == {"type": "object", "properties": {}}
    # The OpenAI wrapper keys must be gone.
    assert "function" not in forwarded_tools[0]
    assert forwarded_tools[0].get("type") != "function"


# ---------- gateway tool dispatch ----------


def test_mcp_servers_dispatches_through_anthropic_tool_loop(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A request with ``mcp_servers`` routes through ``anthropic_tool_loop`` rather
    than calling ``amessages`` directly. Catches regressions where the dispatch
    if/elif chain silently falls through to the plain path.
    """
    seen: dict[str, Any] = {}

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> MessageResponse:
        seen["completion_kwargs"] = completion_kwargs
        seen["pool"] = pool
        seen["max_iterations"] = max_iterations
        return _text_response("via-mcp-loop")

    plain_amessages_called = False

    async def fake_amessages(**_kwargs: Any) -> MessageResponse:
        nonlocal plain_amessages_called
        plain_amessages_called = True
        return _text_response()

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "mcp_servers": [
                    {"name": "test", "url": "http://127.0.0.1:9999/mcp"},
                ],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "via-mcp-loop"
    assert "completion_kwargs" in seen, "anthropic_tool_loop was not invoked"
    assert plain_amessages_called is False


def test_code_execution_dispatches_through_sandbox_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tools: [{"type": "otari_code_execution"}]`` routes through ``SandboxBackend``."""
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    pool_seen: list[Any] = []

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> MessageResponse:
        pool_seen.append(pool)
        return _text_response("via-sandbox-loop")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes._pipeline.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "via-sandbox-loop"
    assert pool_seen == [fake_backend], "loop didn't receive the SandboxBackend"


def test_web_search_dispatches_through_web_search_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tools: [{"type": "otari_web_search"}]`` routes through ``WebSearchBackend``."""
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")

    pool_seen: list[Any] = []

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> MessageResponse:
        pool_seen.append(pool)
        return _text_response("via-web-search-loop")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    fake_builder_result = AsyncMock(
        __aenter__=AsyncMock(return_value=fake_backend),
        __aexit__=AsyncMock(return_value=None),
    )

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline._build_web_search_backend", return_value=fake_builder_result),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_web_search"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "via-web-search-loop"
    assert pool_seen == [fake_backend], "loop didn't receive the WebSearchBackend"


# ---------- provider-named keyword passthrough ----------


@pytest.mark.parametrize("tool_type", ["code_execution", "code_interpreter", "code_execution_20250825"])
def test_provider_code_execution_passes_through_to_upstream(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tool_type: str,
) -> None:
    """Provider-named code-execution keywords are NOT intercepted. They stay in
    ``tools[]`` and reach ``amessages`` so Anthropic runs the code in its own
    native sandbox — even with no gateway sandbox configured.
    """
    monkeypatch.delenv("GATEWAY_SANDBOX_URL", raising=False)
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "tools": [{"type": tool_type}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    forwarded = captured.get("tools")
    assert forwarded is not None, "provider-named tool was dropped instead of forwarded"
    assert {t["type"] for t in forwarded} == {tool_type}


@pytest.mark.parametrize("tool_type", ["web_search", "web_search_20250305"])
def test_provider_web_search_passes_through_to_upstream(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tool_type: str,
) -> None:
    """Provider-named web_search keywords pass through to Anthropic even when
    no gateway web_search backend is configured."""
    monkeypatch.delenv("GATEWAY_WEB_SEARCH_URL", raising=False)
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": tool_type}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    forwarded = captured.get("tools")
    assert forwarded is not None, "provider-named tool was dropped instead of forwarded"
    assert {t["type"] for t in forwarded} == {tool_type}


# ---------- validation errors (Anthropic-shaped 400) ----------


def _assert_anthropic_error(body: dict[str, Any], *, error_type: str, message_substr: str) -> None:
    assert "detail" in body, body
    detail = body["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == error_type
    assert message_substr in detail["error"]["message"]


def test_code_execution_without_sandbox_env_returns_400_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GATEWAY_SANDBOX_URL", raising=False)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"type": "otari_code_execution"}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(resp.json(), error_type="invalid_request_error", message_substr="OTARI_SANDBOX_URL")


def test_code_execution_combined_with_mcp_servers_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"type": "otari_code_execution"}],
            "mcp_servers": [{"name": "x", "url": "http://127.0.0.1:9999/mcp"}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="otari_code_execution and mcp_servers cannot be combined",
    )


def test_web_search_combined_with_sandbox_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [
                {"type": "otari_code_execution"},
                {"type": "otari_web_search"},
            ],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="otari_web_search cannot be combined",
    )


# ---------- gateway-side runtime errors ----------


def test_max_tool_iterations_exceeded_returns_422_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway's own iteration cap is distinct from a provider outage —
    422 with the Anthropic error envelope lets callers tell them apart.
    """
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.mcp_loop_messages import MaxToolIterationsExceeded

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> MessageResponse:
        raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes._pipeline.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
                "max_tool_iterations": 1,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 422
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="max_tool_iterations",
    )


def test_sandbox_unreachable_returns_502_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.sandbox_backend import SandboxNotReachableError

    with patch(
        "gateway.api.routes._pipeline.SandboxBackend",
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=SandboxNotReachableError("boom"))),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 502
    _assert_anthropic_error(resp.json(), error_type="api_error", message_substr="sandbox unreachable")


# ---------- streaming dispatch ----------


def _stream_message_start() -> MessageStartEvent:
    return MessageStartEvent(
        type="message_start",
        message=cast(Any, _text_response("")),
    )


def _stream_text_delta(text: str) -> ContentBlockDeltaEvent:
    return ContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=cast(Any, TextDelta(type="text_delta", text=text)),
    )


def _stream_message_delta() -> MessageDeltaEvent:
    return MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=1,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )


def _stream_message_stop() -> MessageStopEvent:
    return MessageStopEvent(type="message_stop")


async def _stream_iter(*events: MessageStreamEvent) -> AsyncIterator[MessageStreamEvent]:
    for event in events:
        yield event


def test_stream_no_tools_returns_sse_response(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``stream: true`` with no gateway tools wraps the upstream stream in an
    SSE response. The route should NOT call the tool loop.
    """

    async def fake_amessages(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return _stream_iter(_stream_message_start(), _stream_message_stop())

    tool_loop_called = False

    async def fake_loop_stream(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        nonlocal tool_loop_called
        tool_loop_called = True
        # Async-generator shape to match anthropic_tool_loop_stream.
        return
        yield  # noqa: F811 — needed to classify this as an async generator

    with (
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert tool_loop_called is False


def test_stream_mcp_servers_dispatches_through_tool_loop_stream(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``stream: true`` with ``mcp_servers`` routes through
    ``anthropic_tool_loop_stream`` rather than calling ``amessages`` directly.
    """
    seen: dict[str, Any] = {}

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int
    ) -> AsyncIterator[MessageStreamEvent]:
        seen["pool"] = pool
        seen["max_iterations"] = max_iterations
        yield _stream_message_stop()

    plain_amessages_called = False

    async def fake_amessages(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        nonlocal plain_amessages_called
        plain_amessages_called = True
        return _stream_iter()

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:9999/mcp"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert seen.get("pool") is not None, "anthropic_tool_loop_stream was not invoked"
    assert plain_amessages_called is False


def test_stream_code_execution_dispatches_through_sandbox(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stream: true`` with ``otari_code_execution`` opens the sandbox backend
    and feeds it to ``anthropic_tool_loop_stream``.
    """
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    pool_seen: list[Any] = []

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int
    ) -> AsyncIterator[MessageStreamEvent]:
        pool_seen.append(pool)
        yield _stream_message_stop()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []
    fake_backend.__aenter__ = AsyncMock(return_value=fake_backend)
    fake_backend.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes._pipeline.SandboxBackend", return_value=fake_backend),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert pool_seen == [fake_backend], "tool loop didn't receive the SandboxBackend"


def test_stream_sandbox_unreachable_returns_502_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the eager-open error mapping bug: when the
    streaming sandbox eager-open fails, the route must return a 502 with the
    Anthropic error envelope rather than a 500 (which is what would happen
    if the streaming dispatch wasn't wrapped in the error-mapping
    try/except).
    """
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.sandbox_backend import SandboxNotReachableError

    with patch(
        "gateway.api.routes._pipeline.SandboxBackend",
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=SandboxNotReachableError("boom"))),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 502
    _assert_anthropic_error(resp.json(), error_type="api_error", message_substr="sandbox unreachable")
