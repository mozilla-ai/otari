from collections.abc import Generator
from typing import Any

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    PromptTokensDetails,
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


def test_hybrid_mode_requires_authorization_header(platform_client: TestClient) -> None:
    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}


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
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid user token"}


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
                json={
                    "request_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )

        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        assert kwargs["model"] == "openai:gpt-4o-mini"
        assert kwargs["api_key"] == "sk-platform-key"
        return ChatCompletion(
            id="chatcmpl-platform",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=7,
                total_tokens=17,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=6),
            ),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68"
    assert usage_reports == [
        {
            "correlation_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
            "status": "success",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
                "cache_read_tokens": 6,
                "cache_write_tokens": 0,
            },
        }
    ]


def test_hybrid_mode_accepts_legacy_resolve_shape(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older otari (pre-fallback) returns a flat resolve payload.

    Gateway must accept it and treat it as a single-attempt route so deployments
    where the platform side hasn't been upgraded yet still work.
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
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-platform-key",
                    "api_base": "https://api.openai.com/v1",
                    "managed": True,
                    "correlation_id": "9b2cce4a-5e91-4c19-9ad5-17a83f72b001",
                },
            )

        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-legacy",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    # Gateway maps the legacy correlation_id onto attempt_id, so X-Correlation-ID
    # still carries the same value as before.
    assert response.headers["X-Correlation-ID"] == "9b2cce4a-5e91-4c19-9ad5-17a83f72b001"
    assert usage_reports[0]["correlation_id"] == "9b2cce4a-5e91-4c19-9ad5-17a83f72b001"
    assert usage_reports[0]["status"] == "success"


def test_hybrid_mode_maps_provider_timeout(
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
                json={
                    "request_id": "41a9667f-0af7-4ddf-8468-65c5f5c2af57",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "41a9667f-0af7-4ddf-8468-65c5f5c2af57",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )

        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        raise TimeoutError("provider timeout")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "LLM provider timeout"}


def test_hybrid_mode_propagates_resolve_rate_limit_retry_after(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Rate limited"}, headers={"Retry-After": "11"})

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "11"
    assert response.json() == {"detail": "Rate limited"}


def test_hybrid_mode_usage_retries_only_transient_failures(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_calls: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "e655dc9a-6d90-4207-b371-f58d521a7a81",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "e655dc9a-6d90-4207-b371-f58d521a7a81",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )

        usage_calls.append(body)
        if len(usage_calls) == 1:
            return httpx.Response(500)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-platform",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )
    assert response.status_code == 200
    assert len(usage_calls) == 2


def test_hybrid_mode_maps_resolve_validation_error_to_bad_gateway(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(422, json={"detail": "missing headers"})

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Authorization service unavailable"}


# ---------------------------------------------------------------------------
# Streaming fallback (v1.1)
# ---------------------------------------------------------------------------


def test_hybrid_mode_streaming_falls_through_on_first_attempt_failure(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming request whose first attempt errors before any chunk → falls
    through to the second attempt; client sees a clean 200 SSE stream from
    the second provider."""
    from collections.abc import AsyncIterator

    from any_llm.types.completion import ChatCompletionChunk

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
                json={
                    "request_id": "stream-req-1",
                    "fallback_enabled": True,
                    "attempts": [
                        {
                            "attempt_id": "stream-att-anthropic",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-ant-broken",
                            "api_base": None,
                            "managed": False,
                        },
                        {
                            "attempt_id": "stream-att-openai",
                            "position": 1,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-openai-real",
                            "api_base": "https://api.openai.com/v1",
                            "managed": False,
                        },
                    ],
                },
            )
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeApiStatusError(Exception):
        # status_code on the exception is what _classify_upstream_error reads;
        # 401 is in _FALLBACK_RETRYABLE_STATUS_CODES so the gateway will move
        # on to the next attempt.
        status_code = 401

    async def fake_acompletion(**kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        calls.append(model)
        if "anthropic" in model:
            raise _FakeApiStatusError("simulated upstream 401")

        async def _success_stream() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
                }
            )
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chunk-2",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            )

        return _success_stream()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "stream-att-openai"
    # StreamingResponse builds its own response object, so X-Otari-Request-ID
    # has to be set in the StreamingResponse headers directly — assigning to
    # the dependency-injected Response object doesn't propagate.
    assert response.headers["X-Otari-Request-ID"] == "stream-req-1"
    # Both attempts were tried in order — anthropic first, then openai succeeded.
    assert [m for m in calls if "anthropic" in m or "openai" in m] == [
        "anthropic:claude-haiku-4-5",
        "openai:gpt-4o-mini",
    ]
    # The body should be a valid SSE stream from openai.
    body = response.text
    assert "data:" in body
    assert "hi" in body

    # The failed anthropic attempt should have reported an error to the platform.
    error_reports = [r for r in usage_reports if r.get("status") == "error"]
    assert len(error_reports) == 1
    assert error_reports[0]["correlation_id"] == "stream-att-anthropic"


def test_hybrid_mode_streaming_returns_502_when_all_attempts_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every attempt fails before yielding, the gateway returns 502 with
    the multi-attempt error wording instead of starting an SSE stream."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "stream-req-fail",
                    "fallback_enabled": True,
                    "attempts": [
                        {
                            "attempt_id": "att-a",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-ant-broken",
                            "api_base": None,
                            "managed": False,
                        },
                        {
                            "attempt_id": "att-b",
                            "position": 1,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-openai-broken",
                            "api_base": None,
                            "managed": False,
                        },
                    ],
                },
            )
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> Any:
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "All upstream providers failed"}


def test_hybrid_mode_streaming_reports_every_attempt_when_all_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every streaming attempt fails before its first chunk, each attempt's
    error outcome is still reported back to the platform. The terminal 502 drops
    the queued BackgroundTasks, so the reports must be sent inline; otherwise a
    total streaming outage leaves no per-attempt record.
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
                json={
                    "request_id": "stream-req-fail",
                    "fallback_enabled": True,
                    "attempts": [
                        {
                            "attempt_id": "att-a",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-ant-broken",
                            "api_base": None,
                            "managed": False,
                        },
                        {
                            "attempt_id": "att-b",
                            "position": 1,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-openai-broken",
                            "api_base": None,
                            "managed": False,
                        },
                    ],
                },
            )
        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> Any:
        raise httpx.HTTPStatusError(
            "500",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(500, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    # Each failed attempt is reported exactly once, despite the terminal 502. A
    # set would mask a double-report (the dropped-then-also-flushed bug), so pin
    # the exact count and contents: the inline flush and the dropped background
    # copies must not both fire.
    assert len(usage_reports) == 2
    reported = sorted((r["correlation_id"], r["status"], r.get("error_class")) for r in usage_reports)
    assert reported == [
        ("att-a", "error", "http_500"),
        ("att-b", "error", "http_500"),
    ]


# ---------------------------------------------------------------------------
# Tool-loop fallback (pre-lock-in)
# ---------------------------------------------------------------------------
#
# Contract: when a request uses an inline tool backend (mcp_servers /
# sandbox / web_search), per-attempt fallback still applies as long as the
# chosen attempt has not yet returned its first assistant message. Once it
# has — i.e. the tool loop has "locked in" — subsequent upstream failures
# terminate the request; we never silently swap providers between tool-use
# rounds.


class _FakeMcpPool:
    """Minimal MCPClientPool duck-type for the fallback-flow tests. We don't
    actually want to dial out to an MCP server here — only to exercise the
    chat route's per-attempt iteration around the tool loop."""

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


def _two_attempt_resolve_response(*, request_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": request_id,
            "fallback_enabled": True,
            "attempts": [
                {
                    "attempt_id": "tool-att-anthropic",
                    "position": 0,
                    "provider": "anthropic",
                    "model": "claude-haiku-4-5",
                    "api_key": "sk-ant-broken",
                    "api_base": None,
                    "managed": False,
                },
                {
                    "attempt_id": "tool-att-openai",
                    "position": 1,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-openai-real",
                    "api_base": "https://api.openai.com/v1",
                    "managed": False,
                },
            ],
        },
    )


def test_hybrid_mode_tool_loop_falls_through_pre_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming MCP request: first attempt errors before any tool round
    completes → the gateway falls through to the second attempt and returns
    its successful completion."""

    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response(request_id="tool-req-1")
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeAuthError(Exception):
        status_code = 401

    async def fake_loop_acompletion(**kwargs: Any) -> ChatCompletion:
        model = kwargs.get("model", "")
        calls.append(model)
        if "anthropic" in model:
            raise _FakeAuthError("simulated upstream 401 on anthropic")
        return ChatCompletion(
            id="cmpl-1",
            object="chat.completion",
            created=0,
            model="openai:gpt-4o-mini",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hello from openai"),
                )
            ],
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop.acompletion", fake_loop_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "tool-att-openai"
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello from openai"
    # Both attempts were tried in order — confirms the [:1] collapse is gone.
    assert calls == ["anthropic:claude-haiku-4-5", "openai:gpt-4o-mini"]
    # The first attempt's failure was reported to the platform as `error`.
    error_reports = [r for r in usage_reports if r.get("status") == "error"]
    assert len(error_reports) == 1
    assert error_reports[0]["correlation_id"] == "tool-att-anthropic"


def test_hybrid_mode_tool_loop_no_fallback_after_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming MCP request: first attempt returns a tool_call (lock-in
    fires), then upstream dies on round 2. The gateway must NOT try the
    second attempt — that would replay a provider-specific transcript on a
    different provider."""

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response(request_id="tool-req-2")
        return httpx.Response(204)

    calls: list[str] = []
    state = {"round": 0}

    async def fake_loop_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs.get("model", ""))
        state["round"] += 1
        if state["round"] == 1:
            return ChatCompletion(
                id="cmpl-round1",
                object="chat.completion",
                created=0,
                model="anthropic:claude-haiku-4-5",
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                {  # type: ignore[list-item]
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "remote_search", "arguments": "{}"},
                                }
                            ],
                        ),
                    )
                ],
                usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        # Round 2 (still on attempt 1 — lock-in is in effect) — upstream dies.
        raise RuntimeError("simulated upstream 5xx on round 2")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop.acompletion", fake_loop_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    # Both calls were to attempt 1 (rounds 1 and 2 of the tool loop). The
    # second attempt (openai) is never tried because lock-in fired on round 1.
    assert calls == ["anthropic:claude-haiku-4-5", "anthropic:claude-haiku-4-5"]


def test_hybrid_mode_tool_loop_streaming_falls_through_pre_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming MCP request: first attempt errors before yielding any
    chunk → gateway falls through to the second attempt and streams its
    response. Same pre-lock-in semantics as the non-streaming case."""
    from collections.abc import AsyncIterator

    from any_llm.types.completion import ChatCompletionChunk

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response(request_id="tool-stream-req-1")
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeAuthError(Exception):
        status_code = 401

    async def fake_loop_acompletion(**kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        calls.append(model)
        if "anthropic" in model:
            raise _FakeAuthError("simulated upstream 401")

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "openai:gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
                }
            )
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chunk-2",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "openai:gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            )

        return _stream()

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.MCPClientPool", _FakeMcpPool)
    monkeypatch.setattr("gateway.services.mcp_loop.acompletion", fake_loop_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "tool-att-openai"
    assert calls == ["anthropic:claude-haiku-4-5", "openai:gpt-4o-mini"]
    assert "hello" in response.text


# ---------------------------------------------------------------------------
# Web-search platform policy resolution
# ---------------------------------------------------------------------------
#
# In hybrid mode an `otari_web_search` request consults the platform's
# `/gateway/web-search/resolve` endpoint: if web search is disabled for the
# workspace the gateway 403s; otherwise the resolved workspace config is
# merged into the tool entry (per-request values win) before the backend runs.


class _FakeWebSearchBackend:
    """Minimal WebSearchBackend duck-type that records the tool_entry it was
    built from and resolves the tool loop in a single round (no real search)."""

    last_tool_entry: dict[str, Any] | None = None
    last_auth_token: str | None = None

    def __init__(self, *, base_url: str, tool_entry: dict[str, Any], auth_token: str | None = None) -> None:
        type(self).last_tool_entry = dict(tool_entry)
        type(self).last_auth_token = auth_token

    async def __aenter__(self) -> "_FakeWebSearchBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "web_search", "description": "", "parameters": {}}}]

    def owns_tool(self, name: str) -> bool:
        return name == "web_search"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return "results"


def _single_attempt_resolve_response(*, request_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": request_id,
            "fallback_enabled": False,
            "attempts": [
                {
                    "attempt_id": request_id,
                    "position": 0,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-platform-key",
                    "api_base": "https://api.openai.com/v1",
                    "managed": True,
                }
            ],
        },
    )


def test_hybrid_mode_web_search_403_when_disabled(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An `otari_web_search` request whose workspace has web search disabled
    is rejected with 403 before any provider call."""
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://searxng:8080")

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _single_attempt_resolve_response(request_id="ws-req-disabled")
        if url.endswith("/gateway/web-search/resolve"):
            return httpx.Response(200, json={"enabled": False})
        return httpx.Response(204)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "otari_web_search"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "web search is not enabled for this workspace"}


def test_hybrid_mode_web_search_merges_workspace_config(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When enabled, the resolved workspace config is merged into the tool
    entry with per-request values winning over workspace defaults."""
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://searxng:8080")
    _FakeWebSearchBackend.last_tool_entry = None
    _FakeWebSearchBackend.last_auth_token = None

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _single_attempt_resolve_response(request_id="ws-req-enabled")
        if url.endswith("/gateway/web-search/resolve"):
            return httpx.Response(
                200,
                json={
                    "enabled": True,
                    "max_results": 9,
                    "allowed_domains": ["docs.python.org"],
                    "purpose_hint": "workspace hint",
                    "provider_options": {"search_depth": "advanced"},
                },
            )
        return httpx.Response(204)

    async def fake_loop_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="cmpl-ws",
            object="chat.completion",
            created=0,
            model="openai:gpt-4o-mini",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="answer"),
                )
            ],
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline._build_web_search_backend", _FakeWebSearchBackend)
    monkeypatch.setattr("gateway.services.mcp_loop.acompletion", fake_loop_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            # Per-request max_results=3 must win over the workspace default 9.
            "tools": [{"type": "otari_web_search", "max_results": 3}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    merged = _FakeWebSearchBackend.last_tool_entry
    assert merged is not None
    # Per-request value wins.
    assert merged["max_results"] == 3
    # Workspace defaults fill in the unset keys.
    assert merged["allowed_domains"] == ["docs.python.org"]
    assert merged["purpose_hint"] == "workspace hint"
    assert merged["provider_options"] == {"search_depth": "advanced"}
    # The backend here is searxng (NOT the platform base URL), so the gateway
    # must NOT leak the platform token to it.
    assert _FakeWebSearchBackend.last_auth_token is None


def test_hybrid_mode_web_search_forwards_token_to_platform_backend(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When GATEWAY_WEB_SEARCH_URL points at the platform itself, the gateway
    forwards its platform token as X-Gateway-Token so the platform-hosted
    backend can authenticate it. (Non-platform backends get no token.)"""
    # platform_client's base_url is http://platform.test/api/v1.
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://platform.test/api/v1/gateway/web-search")
    _FakeWebSearchBackend.last_auth_token = None

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _single_attempt_resolve_response(request_id="ws-req-platform")
        if url.endswith("/gateway/web-search/resolve"):
            return httpx.Response(200, json={"enabled": True})
        return httpx.Response(204)

    async def fake_loop_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="cmpl-ws-plat",
            object="chat.completion",
            created=0,
            model="openai:gpt-4o-mini",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="answer"),
                )
            ],
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline._build_web_search_backend", _FakeWebSearchBackend)
    monkeypatch.setattr("gateway.services.mcp_loop.acompletion", fake_loop_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "otari_web_search"}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert _FakeWebSearchBackend.last_auth_token == "gw_test_token"


def test_hybrid_mode_web_search_empty_request_list_keeps_workspace_policy(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request `allowed_domains: []` reads as "no preference" and must NOT clear
    the workspace allow-list — empty/falsy per-request values fall back to the
    workspace value instead of overriding it."""
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://searxng:8080")
    _FakeWebSearchBackend.last_tool_entry = None

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _single_attempt_resolve_response(request_id="ws-req-empty")
        if url.endswith("/gateway/web-search/resolve"):
            return httpx.Response(200, json={"enabled": True, "allowed_domains": ["docs.python.org"]})
        return httpx.Response(204)

    async def fake_loop_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="cmpl-ws",
            object="chat.completion",
            created=0,
            model="openai:gpt-4o-mini",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="answer"),
                )
            ],
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline._build_web_search_backend", _FakeWebSearchBackend)
    monkeypatch.setattr("gateway.services.mcp_loop.acompletion", fake_loop_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "otari_web_search", "allowed_domains": []}],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    merged = _FakeWebSearchBackend.last_tool_entry
    assert merged is not None
    # Empty per-request list did not wipe the workspace allow-list.
    assert merged["allowed_domains"] == ["docs.python.org"]


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
                json={
                    "request_id": "stream-req-single",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "att-a",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-broken",
                            "api_base": None,
                            "managed": False,
                        }
                    ],
                },
            )
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> Any:
        raise httpx.HTTPStatusError(
            "404",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(404, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "The requested model was not found on the provider"}


def test_hybrid_mode_streaming_multi_attempt_classifies_non_retryable_invalid_request(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-retryable invalid request (400) short-circuits the streaming
    fallback and is surfaced as a classified 400 even with multiple attempts,
    matching the non-streaming path rather than the aggregate 502."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "stream-req-400",
                    "fallback_enabled": True,
                    "attempts": [
                        {
                            "attempt_id": "att-a",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-1",
                            "api_base": None,
                            "managed": False,
                        },
                        {
                            "attempt_id": "att-b",
                            "position": 1,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-2",
                            "api_base": None,
                            "managed": False,
                        },
                    ],
                },
            )
        return httpx.Response(204)

    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("POST", "http://upstream"),
            response=httpx.Response(400, request=httpx.Request("POST", "http://upstream")),
        )

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "anything", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The provider rejected the request as invalid (check the model name and parameters)"
    }
    # Non-retryable: the fallback must short-circuit after the first attempt.
    assert len(calls) == 1
