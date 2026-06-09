"""End-to-end tests for the standalone-vs-platform mode strategy seam.

These drive real HTTP requests through the FastAPI app (TestClient) with mocked
upstream providers, and assert the observable contracts the
``RequestModeStrategy`` / ``RequestSettlement`` seam is responsible for:

- standalone settlement: success reconciles spend and releases the budget hold;
  a provider error refunds the hold and logs an error row;
- the deliberately-preserved asymmetry where a pre-commit streaming provider
  error refunds for chat but logs-without-refund for messages (see
  ``on_provider_error_precommit``);
- mode-resolution: ``mcp_server_ids`` is rejected in standalone mode, and
  platform mode sets the ``X-Otari-Request-ID`` / ``X-Correlation-ID`` headers
  and reports usage upstream.

The existing standalone and platform integration suites prove the refactor did
not regress; this file adds seam-targeted coverage that asserts budget /
UsageLog state, not just status codes.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from any_llm.types.messages import MessageResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.entities import UsageLog, User

from .conftest import MODEL_NAME

_PRICING = {"input_price_per_million": 2.5, "output_price_per_million": 10.0}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _seed_budgeted_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    budget = client.post("/v1/budgets", json={"max_budget": 100.0}, headers=headers)
    assert budget.status_code == 200
    budget_id = budget.json()["budget_id"]
    created = client.post(
        "/v1/users",
        json={"user_id": user_id, "budget_id": budget_id},
        headers=headers,
    )
    assert created.status_code == 200


def _configure_pricing(client: TestClient, headers: dict[str, str], model_key: str = MODEL_NAME) -> None:
    res = client.post(
        "/v1/pricing",
        json={"model_key": model_key, **_PRICING},
        headers=headers,
    )
    assert res.status_code == 200


def _user_state(test_config: GatewayConfig, user_id: str) -> tuple[float, float]:
    """Return ``(spend, reserved)`` for a user, read on a fresh sync session."""
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        assert user is not None
        return float(user.spend), float(user.reserved)
    finally:
        db.close()
        engine.dispose()


def _poll_usage_logs(
    test_config: GatewayConfig, user_id: str, *, timeout: float = 3.0
) -> list[tuple[str, str | None]]:
    """Poll the async log writer until at least one row exists for the user."""
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(bind=engine)
    deadline = time.time() + timeout
    try:
        while True:
            db = session_local()
            try:
                rows = db.query(UsageLog).filter(UsageLog.user_id == user_id).all()
                if rows or time.time() > deadline:
                    return [(r.status, r.error_message) for r in rows]
            finally:
                db.close()
            time.sleep(0.1)
    finally:
        engine.dispose()


def _chat_completion(usage: CompletionUsage) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-e2e",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


# --------------------------------------------------------------------------
# Standalone settlement
# --------------------------------------------------------------------------


def test_standalone_chat_success_reconciles_and_releases_hold(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """settlement.on_success: a successful non-stream chat completion logs the
    usage row, charges the actual cost to spend, and frees the reservation."""
    user_id = "e2e-chat-success"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header)

    # 1,000,000 prompt + 500,000 completion tokens -> 2.5 + 5.0 = 7.5
    usage = CompletionUsage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000)

    async def _fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return _chat_completion(usage)

    with patch("gateway.api.routes.chat.acompletion", side_effect=_fake_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "user": user_id},
            headers=master_key_header,
        )

    assert response.status_code == 200
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(7.5)
    assert reserved == pytest.approx(0.0)
    logs = _poll_usage_logs(test_config, user_id)
    assert [s for s, _ in logs] == ["success"]


def test_standalone_chat_provider_error_refunds_hold(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """settlement.on_error: a non-stream provider error logs an error row,
    refunds the reservation, and leaves spend untouched."""
    user_id = "e2e-chat-error"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header)

    async def _boom(**kwargs: Any) -> ChatCompletion:
        raise RuntimeError("simulated upstream failure")

    with patch("gateway.api.routes.chat.acompletion", side_effect=_boom):
        response = client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "user": user_id},
            headers=master_key_header,
        )

    assert response.status_code == 502
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)  # hold refunded
    logs = _poll_usage_logs(test_config, user_id)
    assert [s for s, _ in logs] == ["error"]


def test_standalone_chat_streaming_success_reconciles(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """settlement.on_success via the streaming generator: a streamed completion
    aggregates usage, charges spend, and releases the hold."""
    user_id = "e2e-chat-stream"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header)

    async def _fake_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
                }
            )
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "c2",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000, "total_tokens": 1_500_000},
                }
            )

        return _stream()

    with patch("gateway.api.routes.chat.acompletion", side_effect=_fake_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": user_id,
                "stream": True,
            },
            headers=master_key_header,
        )
        assert response.status_code == 200
        body = response.text  # fully consume the SSE stream so settlement runs

    assert "[DONE]" in body
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(7.5)
    assert reserved == pytest.approx(0.0)
    logs = _poll_usage_logs(test_config, user_id)
    assert [s for s, _ in logs] == ["success"]


def test_standalone_messages_streaming_precommit_error_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """settlement.on_error: a messages streaming request whose provider call
    fails before the first event logs an error row AND refunds the reservation,
    matching chat and the non-streaming handlers. (Regression guard for the
    pre-commit reservation leak that messages/responses streaming used to have.)"""
    user_id = "e2e-messages-precommit"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header)

    async def _boom(**kwargs: Any) -> MessageResponse:
        raise RuntimeError("simulated upstream failure before first event")

    with patch("gateway.api.routes.messages.amessages", side_effect=_boom):
        response = client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "stream": True,
                "metadata": {"user_id": user_id},
            },
            headers=master_key_header,
        )

    assert response.status_code == 500
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)  # hold refunded (no leak)
    logs = _poll_usage_logs(test_config, user_id)
    assert [s for s, _ in logs] == ["error"]


def test_standalone_responses_streaming_precommit_error_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """settlement.on_error on /v1/responses streaming pre-commit: logs an error
    row and refunds the reservation, same uniform contract as the other two."""
    user_id = "e2e-responses-precommit"
    _seed_budgeted_user(client, master_key_header, user_id)
    # openai supports the Responses API; price it so the reservation hold is
    # genuinely non-zero before the failure refunds it.
    _configure_pricing(client, master_key_header, model_key="openai:gpt-4o-mini")

    async def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("simulated upstream failure before first event")

    with patch("gateway.api.routes.responses.aresponses", side_effect=_boom):
        response = client.post(
            "/v1/responses",
            json={"model": "openai:gpt-4o-mini", "input": "hi", "stream": True, "user": user_id},
            headers=master_key_header,
        )

    assert response.status_code == 502
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)  # hold refunded (no leak)
    logs = _poll_usage_logs(test_config, user_id)
    assert [s for s, _ in logs] == ["error"]


def test_standalone_chat_streaming_precommit_error_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """A chat streaming request whose provider call fails before the first
    chunk logs an error row and refunds the reservation (settlement.on_error on
    the pre-commit path) — the same uniform contract as messages and responses."""
    user_id = "e2e-chat-precommit"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header)

    async def _boom(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        raise RuntimeError("simulated upstream failure before first chunk")

    with patch("gateway.api.routes.chat.acompletion", side_effect=_boom):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": user_id,
                "stream": True,
            },
            headers=master_key_header,
        )

    assert response.status_code == 502
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)  # hold refunded (contrast with messages)
    logs = _poll_usage_logs(test_config, user_id)
    assert [s for s, _ in logs] == ["error"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
        ("/v1/responses", {"input": "hi"}),
    ],
)
def test_standalone_rejects_mcp_server_ids(
    client: TestClient,
    master_key_header: dict[str, str],
    path: str,
    body: dict[str, Any],
) -> None:
    """strategy.resolve_mcp_servers (standalone) rejects workspace-scoped MCP
    server ids with a 400 on every LLM endpoint."""
    response = client.post(
        path,
        json={"model": "openai:gpt-4o-mini", "mcp_server_ids": ["11111111-1111-1111-1111-111111111111"], **body},
        headers=master_key_header,
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# Platform resolution + settlement
# --------------------------------------------------------------------------


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = create_app(GatewayConfig(mode="platform", platform={"base_url": "http://platform.test/api/v1"}))
    with TestClient(app) as client:
        yield client
    reset_config()
    reset_db()


def test_platform_resolve_sets_request_id_header_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PlatformStrategy.resolve sets X-Otari-Request-ID from the resolve call;
    the platform settlement sets X-Correlation-ID and reports usage upstream."""
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "req-e2e-1",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "att-e2e-1",
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
        return _chat_completion(CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17))

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Otari-Request-ID"] == "req-e2e-1"
    assert response.headers["X-Correlation-ID"] == "att-e2e-1"
    assert usage_reports == [
        {
            "correlation_id": "att-e2e-1",
            "status": "success",
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }
    ]


def test_platform_resolves_mcp_server_ids_via_platform(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """strategy.resolve_mcp_servers (platform) swaps workspace-scoped ids for
    inline configs by calling the platform's mcp-servers/resolve endpoint."""
    called_urls: list[str] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        called_urls.append(url)
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "req-mcp-1",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "att-mcp-1",
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
        if url.endswith("/gateway/mcp-servers/resolve"):
            return httpx.Response(200, json={"servers": []})
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return _chat_completion(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_server_ids": ["22222222-2222-2222-2222-222222222222"],
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    # The platform mcp-servers/resolve endpoint was consulted by the strategy.
    assert any(u.endswith("/gateway/mcp-servers/resolve") for u in called_urls)
