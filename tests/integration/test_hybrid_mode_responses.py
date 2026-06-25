"""Hybrid-mode integration tests for /v1/responses.

Mirror of :mod:`tests.integration.test_hybrid_mode_messages` for the OpenAI
Responses endpoint. Tool-loop platform requests are tested only in the
single-attempt collapsed form; pre-lock-in fallback for tool-loop is gated on
``on_first_response`` landing across the codebase.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import httpx
import pytest
from any_llm.types.responses import Response
from fastapi.testclient import TestClient
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_PLATFORM_TOKEN", "gw_test_token")
    app = create_app(
        GatewayConfig(
            mode="hybrid",
            platform={"base_url": "http://platform.test/api/v1"},
        )
    )

    with TestClient(app) as client:
        yield client

    reset_config()
    reset_db()


def _resolve_payload(
    attempts: list[dict[str, Any]],
    request_id: str = "req-1",
    fallback_enabled: bool = True,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "fallback_enabled": fallback_enabled,
        "attempts": attempts,
    }


def _attempt(position: int, attempt_id: str, model: str, api_key: str, provider: str = "openai") -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "position": position,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_base": None,
        "managed": True,
    }


def _response_object() -> Response:
    return Response(
        id="resp_platform",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, "completed"),
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(cached_tokens=5),
            output_tokens=7,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=17,
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


def test_hybrid_mode_requires_authorization_header(platform_client: TestClient) -> None:
    response = platform_client.post(
        "/v1/responses",
        json={"model": "openai:gpt-4o-mini", "input": "hi"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}


def test_hybrid_mode_sets_correlation_id_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, "att-1", "gpt-4o-mini", "sk-platform-key")]),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_aresponses(**kwargs: Any) -> Response:
        assert kwargs["api_key"] == "sk-platform-key"
        return _response_object()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.responses.aresponses", fake_aresponses)

    response = platform_client.post(
        "/v1/responses",
        json={"model": "gpt-4o-mini", "input": "hi"},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Correlation-ID"] == "att-1"
    assert response.headers["X-Otari-Request-ID"] == "req-1"
    assert usage_reports == [
        {
            "correlation_id": "att-1",
            "status": "success",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
                "cache_read_tokens": 5,
                "cache_write_tokens": 0,
            },
        }
    ]


def test_hybrid_mode_falls_through_on_first_attempt_failure(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload(
                    [
                        _attempt(0, "att-primary", "gpt-4o-mini", "sk-bad-key"),
                        _attempt(1, "att-fallback", "gpt-4o-mini", "sk-good-key"),
                    ]
                ),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[dict[str, Any]] = []

    async def fake_aresponses(**kwargs: Any) -> Response:
        calls.append(kwargs)
        if kwargs["api_key"] == "sk-bad-key":
            raise httpx.HTTPStatusError(
                "401",
                request=httpx.Request("POST", "http://upstream"),
                response=httpx.Response(401, request=httpx.Request("POST", "http://upstream")),
            )
        return _response_object()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.responses.aresponses", fake_aresponses)

    response = platform_client.post(
        "/v1/responses",
        json={"model": "gpt-4o-mini", "input": "hi"},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Correlation-ID"] == "att-fallback"
    assert len(calls) == 2
    outcomes = [report["status"] for report in usage_reports]
    assert "error" in outcomes
    assert "success" in outcomes


def test_hybrid_mode_returns_502_when_all_attempts_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload(
                    [
                        _attempt(0, "att-1", "gpt-4o-mini", "sk-1"),
                        _attempt(1, "att-2", "gpt-4o-mini", "sk-2"),
                    ]
                ),
            )
        return httpx.Response(204)

    async def fake_aresponses(**kwargs: Any) -> Response:
        raise httpx.HTTPStatusError(
            "500",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(500, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.responses.aresponses", fake_aresponses)

    response = platform_client.post(
        "/v1/responses",
        json={"model": "gpt-4o-mini", "input": "hi"},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "All upstream providers failed"}


def test_hybrid_mode_provider_without_responses_support_returns_400(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``SUPPORTS_RESPONSES`` provider guard must still fire in hybrid
    mode — once credentials are resolved, the guard checks the primary
    attempt's provider before any upstream call.
    """

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload(
                    [_attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1", provider="anthropic")]
                ),
            )
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/responses",
        json={"model": "claude-3-5-sonnet-20241022", "input": "hi"},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 400
    assert "does not support the Responses API" in response.json()["detail"]


# ---------- tool-loop fallback (pre-lock-in) ----------


class _FakeMcpPool:
    """Minimal MCPClientPool duck-type for the fallback-flow tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeMcpPool":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": "remote_search", "description": "", "parameters": {}},
            }
        ]

    def owns_tool(self, name: str) -> bool:
        return name == "remote_search"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return "tool ran"


def _two_attempt_resolve_response_openai_first(*, request_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": request_id,
            "fallback_enabled": True,
            "attempts": [
                _attempt(0, "tool-att-primary", "gpt-4o-mini", "sk-openai-broken"),
                _attempt(1, "tool-att-fallback", "gpt-4o-mini", "sk-openai-real"),
            ],
        },
    )


def test_hybrid_mode_tool_loop_falls_through_pre_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming MCP request on /v1/responses: first attempt errors before
    any tool round completes → fallback to the second attempt.
    """
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response_openai_first(request_id="tool-req-1")
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeAuthError(Exception):
        status_code = 401

    async def fake_loop_aresponses(**kwargs: Any) -> Response:
        calls.append(kwargs.get("api_key", ""))
        if kwargs.get("api_key") == "sk-openai-broken":
            raise _FakeAuthError("simulated upstream 401 on primary")
        return _response_object()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop_responses.aresponses", fake_loop_aresponses)

    response = platform_client.post(
        "/v1/responses",
        json={
            "model": "gpt-4o-mini",
            "input": "hi",
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Correlation-ID"] == "tool-att-fallback"
    assert len(calls) == 2
    error_reports = [r for r in usage_reports if r.get("status") == "error"]
    assert len(error_reports) == 1
    assert error_reports[0]["correlation_id"] == "tool-att-primary"


def test_hybrid_mode_tool_loop_no_fallback_after_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First attempt returns a function_call (lock-in fires), then upstream
    dies on round 2. The gateway must NOT try the second attempt — the
    transcript carries a provider-specific call_id that can't be replayed.
    """

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response_openai_first(request_id="tool-req-2")
        return httpx.Response(204)

    calls: list[str] = []
    state = {"round": 0}

    async def fake_loop_aresponses(**kwargs: Any) -> Response:
        calls.append(kwargs.get("api_key", ""))
        state["round"] += 1
        if state["round"] == 1:
            from openai.types.responses import ResponseFunctionToolCall

            return Response(
                id="resp_round1",
                created_at=0.0,
                model="gpt-4o-mini",
                object="response",
                status=cast(Any, "completed"),
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_1",
                        name="remote_search",
                        arguments="{}",
                    )
                ],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                usage=ResponseUsage(
                    input_tokens=3,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens=2,
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    total_tokens=5,
                ),
                error=None,
                incomplete_details=None,
                instructions=None,
                metadata=None,
                temperature=None,
                top_p=None,
            )
        raise RuntimeError("simulated upstream 5xx on round 2")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop_responses.aresponses", fake_loop_aresponses)

    response = platform_client.post(
        "/v1/responses",
        json={
            "model": "gpt-4o-mini",
            "input": "hi",
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    # Both calls were to attempt 1 (rounds 1 and 2). Lock-in fired on round 1
    # so the second attempt is never tried.
    assert calls == ["sk-openai-broken", "sk-openai-broken"]


# ---------- hybrid-mode tool-loop streaming contract ----------


def test_hybrid_mode_tool_loop_streaming_sets_correlation_id_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-attempt tool-loop streaming in hybrid mode must honor the
    platform contract: X-Correlation-ID + X-Otari-Request-ID headers + usage
    report via _report_platform_usage on complete. Regression test for the
    issue where ``_stream_responses`` was the standalone helper and silently
    dropped the platform metadata when invoked for platform + tool_loop.
    """
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, "stream-att-1", "gpt-4o-mini", "sk-platform")]),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    from unittest.mock import AsyncMock, patch

    from openai.types.responses import ResponseCompletedEvent

    async def fake_loop_stream(**_kwargs: Any) -> Any:
        # Emit a single response.completed event with usage so _on_complete
        # fires.
        yield ResponseCompletedEvent(
            type="response.completed",
            response=_response_object(),
            sequence_number=0,
        )

    with (
        patch("gateway.api.routes.responses.responses_tool_loop_stream", new=fake_loop_stream),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        with platform_client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "gpt-4o-mini",
                "input": "hi",
                "stream": True,
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
            },
            headers={"Authorization": "Bearer user_test_token"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            assert response.headers["X-Correlation-ID"] == "stream-att-1"
            assert response.headers["X-Otari-Request-ID"] == "req-1"
            for _ in response.iter_bytes():
                pass

    success_reports = [r for r in usage_reports if r.get("status") == "success"]
    assert success_reports, "expected a success usage report for the hybrid-mode tool-loop stream"
    assert success_reports[0]["correlation_id"] == "stream-att-1"


def test_hybrid_mode_supports_responses_guard_checks_every_attempt(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the SUPPORTS_RESPONSES guard. Previously only the
    primary attempt was checked; a fallback to an unsupported provider would
    crash the runner mid-fallback instead of failing fast.
    """

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload(
                    [
                        _attempt(0, "att-1", "gpt-4o-mini", "sk-openai", provider="openai"),
                        _attempt(1, "att-2", "claude-3-5-sonnet-20241022", "sk-ant", provider="anthropic"),
                    ]
                ),
            )
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/responses",
        json={"model": "gpt-4o-mini", "input": "hi"},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 400, response.text
    # Anthropic is the unsupported attempt; the guard must surface it
    # before any upstream call is made.
    assert "anthropic" in response.json()["detail"]


def test_hybrid_mode_streaming_single_attempt_classifies_provider_error(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-attempt streaming request that fails before its first chunk
    surfaces the classified status (404), not a generic 502."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, "att-1", "gpt-4o-mini", "sk-1")]),
            )
        return httpx.Response(204)

    async def fake_aresponses(**kwargs: Any) -> Response:
        raise httpx.HTTPStatusError(
            "404",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(404, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.responses.aresponses", fake_aresponses)

    response = platform_client.post(
        "/v1/responses",
        json={"model": "gpt-4o-mini", "input": "hi", "stream": True},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "The requested model was not found on the provider"}
