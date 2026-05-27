"""Platform-mode integration tests for /v1/messages.

Mirrors :mod:`tests.integration.test_platform_mode_chat`: exercises the
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
    monkeypatch.setenv("OTARI_PLATFORM_TOKEN", "gw_test_token")
    app = create_app(
        GatewayConfig(
            mode="platform",
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
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


def test_platform_mode_requires_authorization_header(platform_client: TestClient) -> None:
    response = platform_client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}


def test_platform_mode_maps_resolve_unauthorized(
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
    assert response.json() == {"detail": "Invalid user token"}


def test_platform_mode_sets_correlation_id_and_reports_usage(
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
                json=_resolve_payload([_attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-platform-key")]),
            )
        usage_reports.append(body)
        return httpx.Response(204)

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
    assert response.headers["X-Correlation-ID"] == "att-1"
    assert response.headers["X-Otari-Request-ID"] == "req-1"
    assert usage_reports == [
        {
            "correlation_id": "att-1",
            "status": "success",
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }
    ]


def test_platform_mode_falls_through_on_first_attempt_failure(
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
                json=_resolve_payload([
                    _attempt(0, "att-primary", "claude-3-5-sonnet-20241022", "sk-bad-key"),
                    _attempt(1, "att-fallback", "claude-3-5-sonnet-20241022", "sk-good-key"),
                ]),
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


def test_platform_mode_returns_502_when_all_attempts_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All attempts failing with retryable errors → 502 with the all-failed
    detail wording.
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
                json=_resolve_payload([
                    _attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1"),
                    _attempt(1, "att-2", "claude-3-5-sonnet-20241022", "sk-2"),
                ]),
            )
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
    assert response.json() == {"detail": "All upstream providers failed"}


def test_platform_mode_non_retryable_error_raises_immediately(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 from the upstream is non-retryable — runner stops the iteration
    at the first attempt and returns 502 right away rather than walking the
    rest of the route.
    """
    calls: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json=_resolve_payload([
                    _attempt(0, "att-1", "claude-3-5-sonnet-20241022", "sk-1"),
                    _attempt(1, "att-2", "claude-3-5-sonnet-20241022", "sk-2"),
                ]),
            )
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
    assert len(calls) == 1, "Non-retryable error must short-circuit the attempts loop"


def test_platform_mode_resolves_workspace_mcp_server_ids(
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
    """Minimal MCPClientPool duck-type — same shape as test_platform_mode_chat's
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


def test_platform_mode_tool_loop_falls_through_pre_lock_in(
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
    monkeypatch.setattr("gateway.api.routes.messages.MCPClientPool", _FakeMcpPool)
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


def test_platform_mode_tool_loop_no_fallback_after_lock_in(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming MCP request: first attempt returns a tool_use (lock-in
    fires via ``on_first_response``), then upstream dies on round 2. The
    gateway must NOT try the second attempt — that would replay a
    provider-specific transcript on a different provider.
    """

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return _two_attempt_resolve_response_anthropic_first(request_id="tool-req-2")
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
    monkeypatch.setattr("gateway.api.routes.messages.MCPClientPool", _FakeMcpPool)
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


# ---------- platform-mode tool-loop streaming contract ----------


def test_platform_mode_tool_loop_streaming_sets_correlation_id_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-attempt tool-loop streaming in platform mode must still honor
    the platform contract: X-Correlation-ID + X-Otari-Request-ID headers,
    and usage reported back via _report_platform_usage on stream complete.

    Regression test for the issue where ``_stream_messages`` was the
    standalone helper and silently dropped the platform metadata when
    invoked for platform + tool_loop streaming.
    """
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
            assert response.headers["X-Correlation-ID"] == "stream-att-1"
            assert response.headers["X-Otari-Request-ID"] == "req-1"
            # Consume the stream so on_complete fires.
            for _ in response.iter_bytes():
                pass

    # _report_platform_usage scheduled via asyncio.create_task; give the loop
    # a moment to drain. TestClient's lifespan generally runs to completion
    # by the time the with-block exits.
    success_reports = [r for r in usage_reports if r.get("status") == "success"]
    assert success_reports, "expected a success usage report for the platform-mode tool-loop stream"
    assert success_reports[0]["correlation_id"] == "stream-att-1"
