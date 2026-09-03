"""End-to-end web-search interception over /v1/chat/completions.

The `/v1/messages` side (including the native server-tool blocks, which only
Anthropic has a vocabulary for) is covered in
``test_messages_route_dispatch.py``. This file covers the other half of the
contract: a provider-named declaration on the OpenAI-shaped endpoint reaches the
gateway's backend and the search actually runs, rather than being forwarded to a
provider that may not serve it.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

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

from .conftest import MODEL_NAME


def _completion(*, tool_call: bool) -> ChatCompletion:
    """A provider response, optionally asking for the gateway's web_search tool."""
    tool_calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                type="function",
                function=Function(name="web_search", arguments='{"query": "otari"}'),
            )
        ]
        if tool_call
        else None
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None if tool_call else "done",
        tool_calls=cast(Any, tool_calls),
    )
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=message, finish_reason="tool_calls" if tool_call else "stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.mark.parametrize(
    "tool_entry",
    [
        {"type": "web_search"},
        {"type": "web_search_20250305"},
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
    ],
)
def test_intercepted_declaration_runs_the_gateway_search(
    client: TestClient,
    api_key_header: dict[str, str],
    tool_entry: dict[str, Any],
) -> None:
    """The search runs and its results reach the model, so the answer comes back."""
    search = AsyncMock(return_value="search results for otari")
    with (
        patch(
            "gateway.services.mcp_loop.acompletion",
            new=AsyncMock(side_effect=[_completion(tool_call=True), _completion(tool_call=False)]),
        ),
        patch("gateway.services.web_search_backend.WebSearchBackend._search_tool", new=search),
        patch.dict(
            "os.environ",
            {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid", "OTARI_WEB_SEARCH_INTERCEPT": "true"},
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [tool_entry],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert search.await_count == 1, "the gateway's search backend never ran"


def test_declaration_is_forwarded_when_interception_is_off(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Default behavior: the keyword reaches the provider and no gateway search runs."""
    search = AsyncMock(return_value="never called")
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return _completion(tool_call=False)

    with (
        patch("gateway.api.routes.chat.acompletion", new=fake_acompletion),
        patch("gateway.services.web_search_backend.WebSearchBackend._search_tool", new=search),
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert [tool["type"] for tool in captured.get("tools") or []] == ["web_search_20250305"]
    assert search.await_count == 0


def test_a_caller_function_named_web_search_is_never_intercepted(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """It stays in ``tools[]`` so the caller can dispatch it themselves."""
    search = AsyncMock(return_value="never called")
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return _completion(tool_call=False)

    own_tool = {
        "type": "function",
        "function": {"name": "web_search", "parameters": {"type": "object", "properties": {}}},
    }
    with (
        patch("gateway.api.routes.chat.acompletion", new=fake_acompletion),
        patch("gateway.services.web_search_backend.WebSearchBackend._search_tool", new=search),
        patch.dict(
            "os.environ",
            {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid", "OTARI_WEB_SEARCH_INTERCEPT": "true"},
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [own_tool],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert captured.get("tools") == [own_tool]
    assert search.await_count == 0


def test_max_uses_bounds_the_searches_on_the_openai_shaped_endpoint(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """The cap is billing, not formatting, so it holds where there is no native error shape.

    The model asks twice with ``max_uses: 1``; the second call never reaches the
    backend and comes back as a tool error the model can answer around.
    """
    search = AsyncMock(return_value="search results for otari")
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.append(kwargs)
        return [_completion(tool_call=True), _completion(tool_call=True), _completion(tool_call=False)][
            len(captured) - 1
        ]

    with (
        patch("gateway.services.mcp_loop.acompletion", new=fake_acompletion),
        patch("gateway.services.web_search_backend.WebSearchBackend._search_tool", new=search),
        patch.dict(
            "os.environ",
            {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid", "OTARI_WEB_SEARCH_INTERCEPT": "true"},
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert search.await_count == 1, "the cap did not stop the second search"
    assert captured[2]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "[tool error] max_uses_exceeded",
    }


def test_max_uses_zero_refuses_the_first_search_rather_than_uncapping_it(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A cap of zero is a cap, so the backend is never reached."""
    search = AsyncMock(return_value="never called")
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.append(kwargs)
        return [_completion(tool_call=True), _completion(tool_call=False)][len(captured) - 1]

    with (
        patch("gateway.services.mcp_loop.acompletion", new=fake_acompletion),
        patch("gateway.services.web_search_backend.WebSearchBackend._search_tool", new=search),
        patch.dict(
            "os.environ",
            {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid", "OTARI_WEB_SEARCH_INTERCEPT": "true"},
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 0}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert search.await_count == 0, "a cap of zero was read as no cap"
    assert captured[1]["messages"][-1]["content"] == "[tool error] max_uses_exceeded"


@pytest.mark.parametrize("invalid_max_uses", [-1, True, False, "2", 1.5])
def test_invalid_max_uses_is_rejected_instead_of_becoming_uncapped(
    client: TestClient,
    api_key_header: dict[str, str],
    invalid_max_uses: Any,
) -> None:
    search = AsyncMock(return_value="never called")

    with (
        patch("gateway.services.web_search_backend.WebSearchBackend._search_tool", new=search),
        patch.dict(
            "os.environ",
            {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid", "OTARI_WEB_SEARCH_INTERCEPT": "true"},
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": invalid_max_uses,
                    }
                ],
            },
            headers=api_key_header,
        )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "web_search max_uses must be a non-negative integer"}
    assert search.await_count == 0
