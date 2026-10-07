"""A non-streaming provider call carries ``provider_request_timeout_seconds``.

Without an explicit timeout the anthropic SDK refuses a non-streaming request
whose ``max_tokens`` could outlast its default time limit, before anything is
sent (#988).
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import any_llm
import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig

from .conftest import build_test_client


class _ShortCircuit(Exception):
    """Raised by a fake provider call once it has captured its kwargs."""


_ANTHROPIC_MESSAGE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


def test_large_max_tokens_non_streaming_message_reaches_anthropic(
    client: TestClient,
    api_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """The real any-llm and anthropic SDK path, over a mock transport."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_ANTHROPIC_MESSAGE)

    async def real_amessages(**kwargs: Any) -> Any:
        kwargs.setdefault("api_key", "sk-ant-test")
        kwargs["client_args"] = {"http_client": httpx.AsyncClient(transport=httpx.MockTransport(handler))}
        return await any_llm.amessages(**kwargs)

    with patch("gateway.api.routes.messages.amessages", new=real_amessages):
        resp = client.post(
            f"{API_ROOT}/messages",
            json={
                "model": "anthropic:claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32000,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert len(sent) == 1
    assert sent[0].extensions["timeout"]["read"] == test_config.provider_request_timeout_seconds


def _capture_chat(client: TestClient, headers: dict[str, str], model: str, **body: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _ShortCircuit

    with patch("gateway.api.routes.chat.acompletion", new=fake_acompletion):
        client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], **body},
            headers=headers,
        )
    return captured


def test_chat_completion_carries_the_default_timeout(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    captured = _capture_chat(client, api_key_header, "openai:gpt-4o")
    assert captured["timeout"] == 600.0


def test_responses_call_carries_the_timeout(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _ShortCircuit

    with patch("gateway.api.routes.responses.aresponses", new=fake_aresponses):
        client.post(
            f"{API_ROOT}/responses",
            json={"model": "openai:gpt-4o", "input": "hi"},
            headers=api_key_header,
        )

    assert captured["timeout"] == 600.0


def test_provider_without_a_per_request_timeout_gets_none(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """any-llm rejects any timeout for such a provider, so none is sent."""
    captured = _capture_chat(client, api_key_header, "ollama:llama3")
    assert captured["model"] == "ollama:llama3"
    assert "timeout" not in captured


def test_streaming_call_is_left_to_the_provider_default(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    captured = _capture_chat(client, api_key_header, "openai:gpt-4o", stream=True)
    assert captured["stream"] is True
    assert "timeout" not in captured


@pytest.fixture
def client_with_short_timeout(postgres_url: str) -> Generator[TestClient]:
    yield from build_test_client(
        GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            auto_migrate=False,
            require_pricing=False,
            provider_request_timeout_seconds=42,
        )
    )


def test_configured_timeout_is_honored(client_with_short_timeout: TestClient) -> None:
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    created = client_with_short_timeout.post(
        f"{API_ROOT}/users", json={"user_id": "timeout-user", "alias": "Timeout"}, headers=headers
    )
    assert created.status_code == 200, created.text

    captured = _capture_chat(client_with_short_timeout, headers, "openai:gpt-4o", user="timeout-user")
    assert captured["timeout"] == 42.0

