"""Route-level tests for the /v1/responses endpoint wiring.

Mirror of :mod:`tests.integration.test_messages_route_dispatch` for the OpenAI
Responses API surface: tool extraction, mutual-exclusivity validation,
per-backend dispatch, and error mapping (plain HTTPException with a ``detail``
string — the Responses API doesn't use the Anthropic-style error envelope).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.responses import Response, ResponseStreamEvent
from fastapi.testclient import TestClient
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

_MODEL = "openai:gpt-4o-mini"


def _response(*, output: list[Any] | None = None, status: str = "completed") -> Response:
    return Response(
        id="resp_test",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, status),
        output=output or [],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=5,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=7,
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


# ---------- plain aresponses (no gateway tools) ----------


def test_no_tools_falls_through_to_plain_aresponses(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return _response()

    with patch("gateway.api.routes.responses.aresponses", new=fake_aresponses):
        resp = client.post(
            "/v1/responses",
            json={"model": _MODEL, "input": "hi"},
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    # No tools in the upstream kwargs since none were requested.
    assert "tools" not in captured


def test_gateway_internal_fields_are_stripped_from_upstream_kwargs(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``mcp_servers`` / ``tools_header`` / ``max_tool_iterations`` are
    gateway-only knobs. They must not be forwarded to ``aresponses``.
    """
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return _response()

    with patch("gateway.api.routes.responses.aresponses", new=fake_aresponses):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "hi",
                "tools_header": "Tools available:",
                "max_tool_iterations": 3,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    for field in ("mcp_servers", "mcp_server_ids", "tools_header", "max_tool_iterations"):
        assert field not in captured, f"gateway-internal field {field!r} leaked to upstream"


def test_user_supplied_chat_shape_tools_get_flattened_to_responses_shape(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Callers may pass tools in OpenAI Chat-Completions nested shape; they
    must be flattened to the Responses API's flat shape before forwarding.
    """
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return _response()

    with patch("gateway.api.routes.responses.aresponses", new=fake_aresponses):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "do it",
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

    assert resp.status_code == 200, resp.text
    forwarded_tools = captured.get("tools")
    assert forwarded_tools is not None
    assert forwarded_tools[0]["type"] == "function"
    assert forwarded_tools[0]["name"] == "lookup"
    assert forwarded_tools[0]["parameters"] == {"type": "object", "properties": {}}
    # No nested function key — that's Chat-Completions shape, not Responses.
    assert "function" not in forwarded_tools[0]


# ---------- gateway tool dispatch ----------


def test_mcp_servers_dispatches_through_responses_tool_loop(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    seen: dict[str, Any] = {}

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> Response:
        seen["completion_kwargs"] = completion_kwargs
        seen["pool"] = pool
        seen["max_iterations"] = max_iterations
        return _response()

    plain_aresponses_called = False

    async def fake_aresponses(**_kwargs: Any) -> Response:
        nonlocal plain_aresponses_called
        plain_aresponses_called = True
        return _response()

    with (
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch("gateway.api.routes.responses.aresponses", new=fake_aresponses),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "hi",
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:9999/mcp"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert "completion_kwargs" in seen, "responses_tool_loop was not invoked"
    assert plain_aresponses_called is False


def test_code_execution_dispatches_through_sandbox_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    pool_seen: list[Any] = []

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> Response:
        pool_seen.append(pool)
        return _response()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes.responses.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "compute",
                "tools": [{"type": "code_execution_20250825"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert pool_seen == [fake_backend]


def test_web_search_dispatches_through_web_search_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")

    pool_seen: list[Any] = []

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> Response:
        pool_seen.append(pool)
        return _response()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    fake_builder_result = AsyncMock(
        __aenter__=AsyncMock(return_value=fake_backend),
        __aexit__=AsyncMock(return_value=None),
    )

    with (
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch("gateway.api.routes.responses._build_web_search_backend", return_value=fake_builder_result),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "search",
                "tools": [{"type": "web_search_20250305"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert pool_seen == [fake_backend]


# ---------- validation errors ----------


def test_code_execution_without_sandbox_env_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GATEWAY_SANDBOX_URL", raising=False)
    resp = client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "hi",
            "tools": [{"type": "code_execution_20250825"}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    assert "GATEWAY_SANDBOX_URL" in resp.json()["detail"]


def test_code_execution_combined_with_mcp_servers_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    resp = client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "hi",
            "tools": [{"type": "code_execution_20250825"}],
            "mcp_servers": [{"name": "x", "url": "http://127.0.0.1:9999/mcp"}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    assert "code_execution and mcp_servers" in resp.json()["detail"]


def test_web_search_combined_with_sandbox_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    resp = client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "hi",
            "tools": [
                {"type": "code_execution_20250825"},
                {"type": "web_search_20250305"},
            ],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    assert "web_search cannot be combined" in resp.json()["detail"]


# ---------- gateway-side runtime errors ----------


def test_max_tool_iterations_exceeded_returns_422(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.mcp_loop_responses import MaxToolIterationsExceeded

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int) -> Response:
        raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes.responses.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "go",
                "tools": [{"type": "code_execution_20250825"}],
                "max_tool_iterations": 1,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 422
    assert "max_tool_iterations" in resp.json()["detail"]


def test_sandbox_unreachable_returns_502(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.sandbox_backend import SandboxNotReachableError

    with patch(
        "gateway.api.routes.responses.SandboxBackend",
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=SandboxNotReachableError("boom"))),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "go",
                "tools": [{"type": "code_execution_20250825"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 502
    assert "sandbox unreachable" in resp.json()["detail"]


# ---------- streaming dispatch ----------


def _stream_text_event(delta: str) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id="msg_1",
        output_index=0,
        content_index=0,
        delta=delta,
        sequence_number=0,
        logprobs=[],
    )


def _stream_completed_event() -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        type="response.completed",
        response=_response(),
        sequence_number=0,
    )


async def _stream_iter(*events: ResponseStreamEvent) -> AsyncIterator[ResponseStreamEvent]:
    for event in events:
        yield event


def test_stream_no_tools_returns_sse_response(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``stream: true`` with no gateway tools wraps the upstream stream in an
    SSE response. The route should NOT call the tool loop — this is the plain
    ``aresponses(stream=True)`` path.

    ``aresponses`` is a coroutine that returns the stream, so the fake is an
    ``async def`` returning an async iterator. The tool loop is an async
    generator, so its fake must use ``yield`` (and is a separate shape).
    """

    async def fake_aresponses(**_kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        return _stream_iter(_stream_text_event("hi"), _stream_completed_event())

    tool_loop_called = False

    async def fake_loop_stream(**_kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        nonlocal tool_loop_called
        tool_loop_called = True
        # Match the async-generator shape of responses_tool_loop_stream. The
        # yield is unreachable but tells Python this is an async generator
        # (not an async function returning an iterator) — that's what the
        # route's ``async for`` expects.
        return
        yield  # noqa: F811 — needed to flag this as an async generator

    with (
        patch("gateway.api.routes.responses.aresponses", new=fake_aresponses),
        patch("gateway.api.routes.responses.responses_tool_loop_stream", new=fake_loop_stream),
    ):
        resp = client.post(
            "/v1/responses",
            json={"model": _MODEL, "input": "hi", "stream": True},
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
    ``responses_tool_loop_stream`` rather than calling ``aresponses`` directly.
    Catches regressions where the streaming dispatch silently falls through.
    """
    seen: dict[str, Any] = {}

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int
    ) -> AsyncIterator[ResponseStreamEvent]:
        seen["pool"] = pool
        seen["max_iterations"] = max_iterations
        yield _stream_completed_event()

    plain_aresponses_called = False

    async def fake_aresponses(**_kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        nonlocal plain_aresponses_called
        plain_aresponses_called = True
        return _stream_iter()

    with (
        patch("gateway.api.routes.responses.responses_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes.responses.aresponses", new=fake_aresponses),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "hi",
                "stream": True,
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:9999/mcp"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert seen.get("pool") is not None, "responses_tool_loop_stream was not invoked"
    assert plain_aresponses_called is False


def test_stream_code_execution_dispatches_through_sandbox(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stream: true`` with ``code_execution_*`` opens the sandbox backend and
    feeds it to ``responses_tool_loop_stream``.

    Note: the streaming path uses the ``sandbox_backend`` instance itself
    (not the value ``__aenter__`` returns) for ``purpose_hints()`` and as
    the ``pool`` argument — matching how real ``SandboxBackend.__aenter__``
    returns ``self``. The fake here mirrors that by having ``__aenter__``
    return the same outer mock and exposing ``purpose_hints`` on it.
    """
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    pool_seen: list[Any] = []

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int
    ) -> AsyncIterator[ResponseStreamEvent]:
        pool_seen.append(pool)
        yield _stream_completed_event()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []
    fake_backend.__aenter__ = AsyncMock(return_value=fake_backend)
    fake_backend.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("gateway.api.routes.responses.responses_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes.responses.SandboxBackend", return_value=fake_backend),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "compute",
                "stream": True,
                "tools": [{"type": "code_execution_20250825"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert pool_seen == [fake_backend], "tool loop didn't receive the SandboxBackend"


def test_stream_sandbox_unreachable_returns_502(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the eager-open error mapping bug: when the
    streaming sandbox eager-open fails, the route must return a 502 with the
    documented detail rather than a 500 (which is what would happen if the
    streaming dispatch wasn't wrapped in the error-mapping try/except).
    """
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.sandbox_backend import SandboxNotReachableError

    with patch(
        "gateway.api.routes.responses.SandboxBackend",
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=SandboxNotReachableError("boom"))),
    ):
        resp = client.post(
            "/v1/responses",
            json={
                "model": _MODEL,
                "input": "go",
                "stream": True,
                "tools": [{"type": "code_execution_20250825"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 502
    assert "sandbox unreachable" in resp.json()["detail"]


# ---------- provider-support guard (pre-existing behavior) ----------


def test_provider_without_responses_support_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """The pre-existing SUPPORTS_RESPONSES check still works after the
    wiring changes. Catches regressions where the provider-support guard
    might be bypassed by the tool dispatch path.
    """
    resp = client.post(
        "/v1/responses",
        json={"model": "anthropic:claude-3-5-sonnet-20241022", "input": "hi"},
        headers=api_key_header,
    )
    assert resp.status_code == 400
    assert "does not support the Responses API" in resp.json()["detail"]
