"""The fetch tool on a provider fetch instance, through the completion routes.

The instance is Tabstack's, named in ``web_fetch_default_tool``. Its calls go
through the app's web provider client, which the routes hand down to the tool
backends through ``ToolPorts``; the test answers them on that client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from fastapi.testclient import TestClient

from gateway.api.deps import get_optional_web_provider_client
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.tools import OTARI_TOOL_CALLS_FIELD

from .conftest import MODEL_NAME

_URL = "https://example.com/notes"
_ENDPOINT = "https://api.tabstack.ai/v1/extract/markdown"


def completion(*, fetch: bool) -> ChatCompletion:
    """A model round: one fetch of ``_URL``, or the final answer."""
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id="call_0", type="function", function=Function(name="web_fetch", arguments=json.dumps({"url": _URL}))
        )
    ]
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None if fetch else "done", tool_calls=cast(Any, calls if fetch else None)
                ),
                finish_reason="tool_calls" if fetch else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@contextmanager
def scripted_model() -> Iterator[list[dict[str, Any]]]:
    """The model asks for one fetch, then answers; what it was sent is kept."""
    sent: list[dict[str, Any]] = []
    rounds = iter([completion(fetch=True), completion(fetch=False)])

    async def fake(**kwargs: Any) -> ChatCompletion:
        sent.append(kwargs)
        return next(rounds)

    with patch("gateway.services.mcp_loop.acompletion", new=fake):
        yield sent


def serve_fetch_on_tabstack(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    config: GatewayConfig,
    answer: Callable[[httpx.Request], httpx.Response],
) -> None:
    """A ``tabstack`` instance as the fetch tool's, and the app's web provider client answering with ``answer``."""
    monkeypatch.setattr(config, "web_fetch_enabled", True)
    monkeypatch.setattr(config, "fetch_tools", {"tabstack": {"provider": "tabstack", "api_key": "tabstack-key"}})
    monkeypatch.setattr(config, "web_fetch_default_tool", "tabstack")

    async def web_provider_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as answering:
            yield answering

    # The client the tool ports carry, which every completion route takes.
    client.app.dependency_overrides[get_optional_web_provider_client] = web_provider_client  # type: ignore[attr-defined]


def tool_result(sent: list[dict[str, Any]]) -> str:
    """What the model was handed for its fetch, in the round after it asked."""
    [message] = [message for message in sent[1]["messages"] if message.get("role") == "tool"]
    return str(message["content"])


def test_a_chat_request_fetches_through_the_instance_on_the_apps_client(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig
) -> None:
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"content": "The notes, as markdown.", "url": _URL})

    serve_fetch_on_tabstack(client, monkeypatch, test_config, answer)
    with scripted_model() as sent:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert [(str(request.url), json.loads(request.content)) for request in requests] == [
        (_ENDPOINT, {"url": _URL, "metadata": True})
    ]
    assert requests[0].headers["Authorization"] == "Bearer tabstack-key"
    assert tool_result(sent).startswith(f"Source: {_URL}\n")
    assert tool_result(sent).endswith("---\nThe notes, as markdown.")
    [ended] = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert (ended["type"], ended["status"], ended["url"]) == ("web_fetch", "completed", _URL)


def test_a_failed_instance_fetch_shows_the_chat_client_its_safe_fields(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig
) -> None:
    serve_fetch_on_tabstack(client, monkeypatch, test_config, lambda _: httpx.Response(401, text="bad key"))
    with scripted_model() as sent:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert tool_result(sent) == "[tool error] tabstack fetch failed (http_error, HTTP 401)"
    [ended] = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert ended["status"] == "failed"
    assert ended["error"] == {"provider": "tabstack", "status": 401, "tag": "http_error", "retryable": False}
    assert "bad key" not in response.text
