"""Hybrid-mode integration tests for /v1/messages.

Mirrors :mod:`tests.integration.test_hybrid_mode_chat`: exercises the
multi-attempt platform resolve / fallback / usage-reporting flow against the
Anthropic Messages endpoint.

Tool-loop platform requests are tested only in the single-attempt collapsed
form here. Once the ``on_first_response`` lock-in plumbing lands across the
codebase, a follow-up will add pre-lock-in fallback tests symmetric with
the chat suite.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import httpx
import pytest
from any_llm.types.messages import (
    MessageResponse,
    MessageUsage,
    TextBlock,
)
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
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


def _attempt(position: int, attempt_id: str, model: str, api_key: str, provider: str = "anthropic") -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "position": position,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_base": None,
        "managed": True,
    }


def _message_response(text: str = "hello") -> MessageResponse:
    return MessageResponse(
        id="msg_platform",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=10,
            output_tokens=7,
            cache_creation_input_tokens=4,
            cache_read_input_tokens=3,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


def _message_response_with_1h_cache_write() -> MessageResponse:
    """A response whose cache creation splits across the 5m and 1h TTL buckets."""
    from anthropic.types.cache_creation import CacheCreation

    return MessageResponse(
        id="msg_platform",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text="hello", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=10,
            output_tokens=7,
            cache_creation_input_tokens=30,
            cache_read_input_tokens=3,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=20, ephemeral_1h_input_tokens=10),
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


def test_hybrid_mode_requires_authorization_header(platform_client: TestClient) -> None:
    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )

    assert response.status_code == 401
    # /v1/messages errors, including auth, are delivered in the Anthropic envelope.
    assert response.json() == {
        "detail": {
            "type": "error",
            "error": {"type": "authentication_error", "message": "Missing authentication token"},
        }
    }


def test_hybrid_mode_maps_resolve_unauthorized(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid user token"})

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": {"type": "error", "error": {"type": "authentication_error", "message": "Invalid user token"}}
    }


def test_hybrid_mode_sets_correlation_id_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_reports: list[dict[str, Any]] = []
    attempt_id = "3f1b6a1e-0000-4000-8000-000000000002"

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, attempt_id, "claude-3-5-sonnet-20241022", "sk-platform-key")]),
            )
        usage_reports.append(body)
        return httpx.Response(
            200,
            json={
                "correlation_id": body["correlation_id"],
                "status": "completed",
                "outcome": "success",
                "cost_usd": "0.012345",
                "currency": "USD",
                "usage_status": "reported",
                "pricing": {"source": "managed"},
            },
        )

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        assert kwargs["model"] == "anthropic:claude-3-5-sonnet-20241022"
        assert kwargs["api_key"] == "sk-platform-key"
        return _message_response()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Correlation-ID"] == attempt_id
    assert response.headers["X-Otari-Request-ID"] == "req-1"
    assert response.json()["usage"]["cost_usd"] == "0.012345"
    assert response.json()["usage"]["pricing_source"] == "managed"
    assert usage_reports == [
        {
            "correlation_id": attempt_id,
            "status": "success",
            "is_final_attempt": True,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
                "cache_read_tokens": 3,
                "cache_write_tokens": 4,
                "cache_write_1h_tokens": 0,
            },
        }
    ]


def test_hybrid_mode_reports_one_hour_cache_write_subset(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1h cache write is reported as a subset of the cache-write total (#856)."""
    usage_reports: list[dict[str, Any]] = []
    attempt_id = "3f1b6a1e-0000-4000-8000-000000000009"

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, attempt_id, "claude-3-5-sonnet-20241022", "sk-platform-key")]),
            )
        usage_reports.append(body)
        return httpx.Response(200, json={"correlation_id": body["correlation_id"], "status": "accepted"})

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _message_response_with_1h_cache_write()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    reported = usage_reports[0]["usage"]
    assert reported["cache_write_tokens"] == 30
    assert reported["cache_write_1h_tokens"] == 10
    # The 1h bucket is a subset, never an addition, so the 5m portion is the remainder.
    assert reported["cache_write_tokens"] - reported["cache_write_1h_tokens"] == 20


def test_hybrid_mode_falls_through_on_first_attempt_failure(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the primary attempt fails with a retryable error, the runner moves
    to the next attempt and the client sees the second attempt's response.
    """
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
                        _attempt(0, "att-primary", "claude-3-5-sonnet-20241022", "sk-bad-key"),
                        _attempt(1, "att-fallback", "claude-3-5-sonnet-20241022", "sk-good-key"),
                    ]
                ),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs)
        if kwargs["api_key"] == "sk-bad-key":
            raise httpx.HTTPStatusError(
                "401",
                request=httpx.Request("POST", "http://upstream"),
                response=httpx.Response(401, request=httpx.Request("POST", "http://upstream")),
            )
        return _message_response("from-fallback")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "from-fallback"
    assert response.headers["X-Correlation-ID"] == "att-fallback"
    # Both attempts hit upstream — the first one failed, the second succeeded.
    assert len(calls) == 2
    # Both attempts reported usage: error for the primary, success for the fallback.
    outcomes = [report["status"] for report in usage_reports]
    assert "error" in outcomes
    assert "success" in outcomes


def test_hybrid_mode_falls_through_on_404_model_unavailable(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 from the primary (deprecated/renamed/retired model) is retryable;
    the runner falls through to the next attempt instead of failing the whole
    request. Recovering from a retired model is a primary reason users configure
    fallback, so 404 must not be treated as terminal.
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
                    [
                        _attempt(0, "att-retired", "claude-3-5-sonnet-20241022", "sk-retired"),
                        _attempt(1, "att-fallback", "claude-3-5-sonnet-20241022", "sk-good-key"),
                    ]
                ),
            )
        return httpx.Response(204)

    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs)
        if kwargs["api_key"] == "sk-retired":
            request = httpx.Request("POST", "http://upstream")
            raise httpx.HTTPStatusError(
                "404",
                request=request,
                response=httpx.Response(404, request=request),
            )
        return _message_response("from-fallback")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "from-fallback"
    assert response.headers["X-Correlation-ID"] == "att-fallback"
    assert len(calls) == 2, "404 on the primary must fall through to the next attempt"


def test_hybrid_mode_returns_502_and_reports_every_attempt_when_all_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All attempts failing with retryable errors → 502 with the all-failed
    detail wording, and each attempt's error outcome is still reported back to
    the platform. The terminal 502 drops the queued BackgroundTasks, so the
    reports must be sent inline; otherwise a total outage leaves no per-attempt
    record and the platform can't account for a fully-exhausted fallback chain.
    """
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
                        _attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1"),
                        _attempt(1, "att-2", "claude-3-5-sonnet-20241022", "sk-2"),
                    ]
                ),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        raise httpx.HTTPStatusError(
            "500",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(500, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    # The aggregate failure is delivered in the Anthropic error envelope rather
    # than a bare {"detail": <str>}.
    assert response.json() == {
        "detail": {"type": "error", "error": {"type": "api_error", "message": "All upstream providers failed"}}
    }
    # Each failed attempt is reported exactly once, despite the terminal 502. A
    # set would mask a double-report (the dropped-then-also-flushed bug), so pin
    # the exact count and contents: the inline flush and the dropped background
    # copies must not both fire.
    assert len(usage_reports) == 2
    reported = sorted((r["correlation_id"], r["status"], r.get("error_class")) for r in usage_reports)
    assert reported == [
        ("att-1", "error", "http_500"),
        ("att-2", "error", "http_500"),
    ]


def test_hybrid_mode_falls_through_on_provider_400(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider 400 advances to the next candidate like every pre-lock-in failure."""
    calls: list[dict[str, Any]] = []
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
                        _attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1"),
                        _attempt(1, "att-2", "claude-3-5-sonnet-20241022", "sk-2"),
                    ]
                ),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs)
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(400, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": {"type": "error", "error": {"type": "api_error", "message": "All upstream providers failed"}}
    }
    assert len(calls) == 2
    assert sorted((r["correlation_id"], r["is_final_attempt"], r.get("error_class")) for r in usage_reports) == [
        ("att-1", False, "http_400"),
        ("att-2", True, "http_400"),
    ]


def test_hybrid_mode_resolves_workspace_mcp_server_ids(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mcp_server_ids`` triggers a second platform call to swap workspace
    ids for inline MCP configs. The resolved configs are merged into the
    request before the tool loop runs.
    """
    server_id = "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68"
    resolve_calls: list[str] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        resolve_calls.append(url)
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1")]),
            )
        if url.endswith("/gateway/mcp-servers/resolve"):
            return httpx.Response(
                200,
                json={
                    "servers": [
                        {
                            "name": "workspace-server",
                            "url": "http://127.0.0.1:9000/mcp",
                            "authorization_token": None,
                            "purpose_hint": None,
                            "allowed_tools": None,
                        }
                    ]
                },
            )
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    # The tool loop is patched so we don't need a real MCP server. We just
    # need to confirm the resolve-mcp-servers call happened (the request
    # successfully reached the tool-loop dispatch).
    async def fake_loop(
        *,
        completion_kwargs: Any,
        pool: Any,
        max_iterations: int,
        on_first_response: Any = None,
        emit_native_web_search: bool = False,
    ) -> MessageResponse:
        return _message_response()

    from unittest.mock import AsyncMock, patch

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        response = platform_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "mcp_server_ids": [server_id],
            },
            headers={"Authorization": "Bearer user_test_token"},
        )

    assert response.status_code == 200, response.text
    # Both platform endpoints were hit: resolve credentials, then resolve
    # mcp-servers.
    assert any(url.endswith("/gateway/provider-keys/resolve") for url in resolve_calls)
    assert any(url.endswith("/gateway/mcp-servers/resolve") for url in resolve_calls)


# ---------- tool-loop fallback (pre-lock-in) ----------


class _FakeMcpPool:
    """Minimal MCPClientPool duck-type — same shape as test_hybrid_mode_chat's
    helper, but exposes the Anthropic-flavored hooks the messages.py runner
    threads through.
    """

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


def _two_attempt_resolve_response_anthropic_first(*, request_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": request_id,
            "fallback_enabled": True,
            "attempts": [
                _attempt(0, "tool-att-primary", "claude-haiku-4-5", "sk-ant-broken"),
                _attempt(1, "tool-att-fallback", "claude-3-5-sonnet-20241022", "sk-ant-real"),
            ],
        },
    )


def test_hybrid_mode_tool_loop_falls_through_pre_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming MCP request: first attempt errors before any tool round
    completes → the gateway falls through to the second attempt and returns
    its successful completion.

    Verifies the ``[:1]`` collapse is gone for tool-loop requests on
    ``/v1/messages`` now that ``on_first_response`` lock-in is wired into
    ``anthropic_tool_loop``.
    """
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response_anthropic_first(request_id="tool-req-1")
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeAuthError(Exception):
        status_code = 401

    async def fake_loop_amessages(**kwargs: Any) -> MessageResponse:
        model = kwargs.get("model", "")
        calls.append(model)
        if "sk-ant-broken" == kwargs.get("api_key"):
            raise _FakeAuthError("simulated upstream 401 on primary")
        return _message_response("from-fallback")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop_messages.amessages", fake_loop_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Correlation-ID"] == "tool-att-fallback"
    body = response.json()
    assert body["content"][0]["text"] == "from-fallback"
    # Both attempts were tried in order — confirms the [:1] collapse is gone.
    assert len(calls) == 2
    # The first attempt's failure was reported to the platform.
    error_reports = [r for r in usage_reports if r.get("status") == "error"]
    assert len(error_reports) == 1
    assert error_reports[0]["correlation_id"] == "tool-att-primary"


def test_hybrid_mode_tool_loop_no_fallback_after_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming MCP request: first attempt returns a tool_use (lock-in
    fires via ``on_first_response``), then upstream dies on round 2. The
    gateway must NOT try the second attempt — that would replay a
    provider-specific transcript on a different provider.
    """
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response_anthropic_first(request_id="tool-req-2")
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []
    state = {"round": 0}

    async def fake_loop_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs.get("api_key", ""))
        state["round"] += 1
        if state["round"] == 1:
            # Round 1 returns a tool_use — fires on_first_response in the loop,
            # locking the attempt in.
            from any_llm.types.messages import ToolUseBlock

            return MessageResponse(
                id="msg_round1",
                type="message",
                role="assistant",
                model="claude-haiku-4-5",
                content=[ToolUseBlock(type="tool_use", id="tu_1", name="remote_search", input={})],
                stop_reason=cast(Any, "tool_use"),
                stop_sequence=None,
                usage=MessageUsage(
                    input_tokens=3,
                    output_tokens=2,
                    cache_creation_input_tokens=None,
                    cache_read_input_tokens=None,
                    cache_creation=None,
                    server_tool_use=None,
                    service_tier=None,
                ),
                container=None,
            )
        # Round 2 (still on attempt 1 — lock-in is in effect) — upstream dies.
        raise RuntimeError("simulated upstream 5xx on round 2")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop_messages.amessages", fake_loop_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    # Both calls were to attempt 1 (rounds 1 and 2 of the tool loop). The
    # second attempt is never tried because lock-in fired on round 1.
    assert calls == ["sk-ant-broken", "sk-ant-broken"]
    assert len(usage_reports) == 1
    assert usage_reports[0]["is_final_attempt"] is True


# ---------- hybrid-mode tool-loop streaming contract ----------


def test_hybrid_mode_tool_loop_streaming_sets_correlation_id_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-loop streaming returns settled cost on the terminal usage event."""
    usage_reports: list[dict[str, Any]] = []
    attempt_id = "3f1b6a1e-0000-4000-8000-000000000003"

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, attempt_id, "claude-3-5-sonnet-20241022", "sk-platform")]),
            )
        usage_reports.append(body)
        return httpx.Response(
            200,
            json={
                "correlation_id": body["correlation_id"],
                "status": "completed",
                "outcome": "success",
                "cost_usd": "0.012345",
                "currency": "USD",
                "usage_status": "reported",
                "pricing": {"source": "managed"},
            },
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    # Patch the streaming tool loop to a no-op generator so we don't need a
    # real MCP server. Return a single message_delta-like event so the
    # streaming_generator triggers _on_complete.
    from unittest.mock import AsyncMock, patch

    from any_llm.types.messages import MessageDelta, MessageDeltaEvent, MessageDeltaUsage

    async def fake_loop_stream(**_kwargs: Any) -> Any:
        yield MessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
            usage=MessageDeltaUsage(
                input_tokens=3,
                output_tokens=5,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                server_tool_use=None,
            ),
        )

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        with platform_client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
            },
            headers={"Authorization": "Bearer user_test_token"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            assert response.headers["X-Correlation-ID"] == attempt_id
            assert response.headers["X-Otari-Request-ID"] == "req-1"
            wire = response.read().decode()

    assert '"cost_usd":"0.012345"' in wire
    assert '"pricing_source":"managed"' in wire
    success_reports = [r for r in usage_reports if r.get("status") == "success"]
    assert success_reports, "expected a success usage report for the hybrid-mode tool-loop stream"
    assert success_reports[0]["correlation_id"] == attempt_id


def test_hybrid_mode_tool_loop_streaming_forwards_session_label(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming success closure (``build_streaming_response._on_complete``)
    on the hybrid tool-loop streaming path (now ``run_streaming_with_fallback``,
    same as chat) must carry the caller's session_label onto the usage report."""
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, "stream-att-1", "claude-3-5-sonnet-20241022", "sk-platform")]),
            )
        usage_reports.append(body)
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    from unittest.mock import AsyncMock, patch

    from any_llm.types.messages import MessageDelta, MessageDeltaEvent, MessageDeltaUsage

    async def fake_loop_stream(**_kwargs: Any) -> Any:
        yield MessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
            usage=MessageDeltaUsage(
                input_tokens=3,
                output_tokens=5,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                server_tool_use=None,
            ),
        )

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        with platform_client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
                "session_label": "my-run-personas",
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
            },
            headers={"Authorization": "Bearer user_test_token"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            # Consume the stream so on_complete fires.
            for _ in response.iter_bytes():
                pass

    success_reports = [r for r in usage_reports if r.get("status") == "success"]
    assert success_reports, "expected a success usage report for the hybrid-mode tool-loop stream"
    assert success_reports[0]["session_label"] == "my-run-personas"


def test_hybrid_mode_count_tokens_requires_authorization_header(
    platform_client: TestClient,
) -> None:
    response = platform_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "type": "error",
            "error": {"type": "authentication_error", "message": "Missing authentication token"},
        }
    }


def test_hybrid_mode_count_tokens_validates_token(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-invalid bearer token is rejected: hybrid mode resolves
    the token rather than just checking the header exists.
    """

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid user token"})

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer not_a_real_token"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": {"type": "error", "error": {"type": "authentication_error", "message": "Invalid user token"}}
    }


def test_hybrid_mode_count_tokens_succeeds_without_provider_call(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid token resolves, and the count is returned without any upstream
    provider call (counting is local).
    """

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        assert url.endswith("/gateway/provider-keys/resolve")
        return httpx.Response(
            200,
            json=_resolve_payload([_attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-platform-key")]),
        )

    async def fail_amessages(**kwargs: Any) -> MessageResponse:
        raise AssertionError("count_tokens must not call the provider")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fail_amessages)

    response = platform_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_hybrid_mode_streaming_single_attempt_classifies_provider_error(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-attempt streaming request that fails before its first chunk
    surfaces the classified status (404) in the Anthropic envelope, not a
    generic 502."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([_attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1")]),
            )
        return httpx.Response(204)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        raise httpx.HTTPStatusError(
            "404",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(404, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "not_found_error"


def test_hybrid_mode_preamble_rejection_uses_anthropic_envelope_and_keeps_retry_after(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform-resolve rejection in the hybrid preamble (here a 429) is
    delivered in the Anthropic error envelope with the matching error.type, and
    its Retry-After header is preserved (the preamble runs before the execution
    runners, so it must be enveloped too)."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(429, json={"detail": "rate limited"}, headers={"Retry-After": "30"})
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "30"
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "rate_limit_error"


def test_hybrid_mode_tool_loop_streaming_falls_through_pre_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming MCP request on /v1/messages: the first attempt errors before
    yielding any event, so the gateway falls through to the second attempt and
    streams its response (same pre-lock-in semantics as chat, which this
    format previously collapsed to a single attempt)."""
    from any_llm.types.messages import MessageDelta, MessageDeltaEvent, MessageDeltaUsage

    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response_anthropic_first(request_id="tool-stream-req-1")
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeAuthError(Exception):
        status_code = 401

    async def fake_loop_stream(**kwargs: Any) -> Any:
        api_key = str(kwargs["completion_kwargs"].get("api_key"))
        calls.append(api_key)
        if api_key == "sk-ant-broken":
            raise _FakeAuthError("simulated upstream 401 on primary")
        yield MessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
            usage=MessageDeltaUsage(
                input_tokens=3,
                output_tokens=5,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                server_tool_use=None,
            ),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.api.routes.messages.anthropic_tool_loop_stream", fake_loop_stream)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Correlation-ID"] == "tool-att-fallback"
    assert response.headers["X-Otari-Request-ID"] == "tool-stream-req-1"
    assert "message_delta" in response.text
    # Both attempts were tried in order: the tool-loop gate is gone.
    assert calls == ["sk-ant-broken", "sk-ant-real"]
    # The first attempt's failure was reported to the platform.
    error_reports = [r for r in usage_reports if r.get("status") == "error"]
    assert len(error_reports) == 1
    assert error_reports[0]["correlation_id"] == "tool-att-primary"


# ---------- container continuity on a shared upstream account ----------


def _container_route(monkeypatch: pytest.MonkeyPatch, *, managed: bool, calls: list[str]) -> None:
    """Wire a single-attempt route whose credential is managed or BYO."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            attempt = _attempt(0, "3f1b6a1e-0000-4000-8000-0000000000c1", "claude-3-5-sonnet-20241022", "sk-upstream")
            attempt["managed"] = managed
            return httpx.Response(200, json=_resolve_payload([attempt]))
        return httpx.Response(
            200,
            json={
                "correlation_id": body["correlation_id"],
                "status": "completed",
                "outcome": "success",
                "cost_usd": "0.010000",
                "currency": "USD",
                "usage_status": "reported",
                "pricing": {"source": "managed"},
            },
        )

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(str(kwargs.get("container")))
        return _message_response()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)


def test_container_is_refused_on_a_managed_credential(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed attempt runs on an account many workspaces share, so a
    caller-chosen container id would address another tenant's state."""
    calls: list[str] = []
    _container_route(monkeypatch, managed=True, calls=calls)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "container": "container_01ABC",
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 400, response.text
    body = response.json()["detail"]
    assert body["error"]["type"] == "invalid_request_error"
    assert "container cannot be used on this route" in body["error"]["message"]
    assert calls == [], "the provider must not be called at all"


def test_container_reaches_a_byo_credential(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the workspace's own key the container is already the caller's."""
    calls: list[str] = []
    _container_route(monkeypatch, managed=False, calls=calls)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "container": "container_01ABC",
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert calls == ["container_01ABC"]


def test_a_managed_attempt_anywhere_on_the_route_refuses_the_container(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which attempt serves the request is decided during fallback, past the
    gate, so a managed fallback behind a BYO primary still refuses."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        assert url.endswith("/gateway/provider-keys/resolve")
        byo = _attempt(0, "3f1b6a1e-0000-4000-8000-0000000000d1", "claude-3-5-sonnet-20241022", "sk-byo")
        byo["managed"] = False
        managed = _attempt(1, "3f1b6a1e-0000-4000-8000-0000000000d2", "claude-3-5-sonnet-20241022", "sk-managed")
        managed["managed"] = True
        return httpx.Response(200, json=_resolve_payload([byo, managed]))

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "container": "container_01ABC",
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 400, response.text
    assert "container cannot be used on this route" in response.json()["detail"]["error"]["message"]


def test_a_request_without_a_container_is_untouched_by_the_gate(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate reads a field the overwhelming majority of requests never set."""
    calls: list[str] = []
    _container_route(monkeypatch, managed=True, calls=calls)

    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    assert calls == ["None"]
