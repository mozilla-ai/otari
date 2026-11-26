"""End-to-end web-search interception over /api/v1/chat/completions.

The `/api/v1/messages` side (including the native server-tool blocks, which only
Anthropic has a vocabulary for) is covered in
``test_messages_route_dispatch.py``. This file covers the other half of the
contract: a provider-named declaration on the OpenAI-shaped endpoint reaches the
gateway's backend and the search actually runs, rather than being forwarded to a
provider that may not serve it.
"""

from __future__ import annotations

from collections.abc import Callable
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

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services._tool_loop import FINAL_ROUND_NOTE
from gateway.services.tools import WEB_FETCH_HEADER, WEB_SEARCH_HEADER
from gateway.services.web_fetch_service import WebFetchResult, WebFetchService
from gateway.services.web_retrieval_policy import canonicalize_web_url

from .conftest import MODEL_NAME, admitted_search_spy


def _page() -> WebFetchResult:
    """What the built-in fetcher reads off a page, so no test reaches the network."""
    url = canonicalize_web_url("https://example.com")
    return WebFetchResult(
        text="page text",
        content_type="text/plain",
        content_kind="text",
        requested_url=url,
        final_url=url,
        redirect_count=0,
    )


def _completion(*, tool_call: bool, tool: str = "web_search", arguments: str = '{"query": "otari"}') -> ChatCompletion:
    """A provider response, optionally asking for one of the gateway's web tools."""
    tool_calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                type="function",
                function=Function(name=tool, arguments=arguments),
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
    serve_search_on_fake: Callable[..., None],
) -> None:
    """The search runs and its results reach the model, so the answer comes back."""
    serve_search_on_fake(extract=False)
    with (
        patch(
            "gateway.services.mcp_loop.acompletion",
            new=AsyncMock(side_effect=[_completion(tool_call=True), _completion(tool_call=False)]),
        ),
        admitted_search_spy() as search,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_INTERCEPT": "true"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
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


def test_an_intercepted_declaration_searches_on_the_deployments_one_instance(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """A deployment whose only search configuration is one instance intercepts to it."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")
    only = {"only": {"provider": "fake"}}
    monkeypatch.setattr(test_config, "search_tools", only)
    # As the overlay's baseline too, for the reason ``serve_search_on_fake`` gives.
    monkeypatch.setattr(test_config, "_search_tool_baseline", only)
    monkeypatch.setattr(test_config, "web_search_extract", False)
    model = AsyncMock(side_effect=[_completion(tool_call=True), _completion(tool_call=False)])
    with patch("gateway.services.mcp_loop.acompletion", new=model):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    [tool_message] = [message for message in model.await_args_list[1].kwargs["messages"] if message["role"] == "tool"]
    assert tool_message["content"].startswith("[1] Fake result one (2026-01-15T09:30:00+00:00)")


@pytest.mark.parametrize("tool_type", ["web_search_20250305", "web_fetch_20250910", "web_fetch_20260209"])
def test_declaration_is_forwarded_when_interception_is_off(
    client: TestClient,
    api_key_header: dict[str, str],
    tool_type: str,
    serve_search_on_fake: Callable[..., None],
) -> None:
    """Default behavior: the keyword reaches the provider and no gateway search runs."""
    serve_search_on_fake(extract=False)
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return _completion(tool_call=False)

    with (
        patch("gateway.api.routes.chat.acompletion", new=fake_acompletion),
        admitted_search_spy() as search,
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": tool_type}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert [tool["type"] for tool in captured.get("tools") or []] == [tool_type]
    assert search.await_count == 0


def test_a_caller_function_named_web_search_is_never_intercepted(
    client: TestClient,
    api_key_header: dict[str, str],
    serve_search_on_fake: Callable[..., None],
) -> None:
    """It stays in ``tools[]`` so the caller can dispatch it themselves."""
    serve_search_on_fake(extract=False)
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
        admitted_search_spy() as search,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_INTERCEPT": "true"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
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


def _offers_search(kwargs: dict[str, Any]) -> bool:
    return any((tool.get("function") or {}).get("name") == "web_search" for tool in kwargs.get("tools") or [])


def test_a_model_that_keeps_searching_past_max_uses_answers_one_round_later(
    client: TestClient,
    api_key_header: dict[str, str],
    serve_search_on_fake: Callable[..., None],
) -> None:
    """The cap is billing, not formatting, so it holds where there is no native error shape.

    The model asks for a search on every round, with ``max_uses: 1`` and ten rounds
    allowed. The round after the one search is the last: it no longer offers the
    search, and the search the model asks for anyway is dropped from the answer (#3183).
    """
    serve_search_on_fake(extract=False)
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.append(kwargs)
        return _completion(tool_call=True)

    with (
        patch("gateway.services.mcp_loop.acompletion", new=fake_acompletion),
        admitted_search_spy() as search,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_INTERCEPT": "true"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
                "max_tool_iterations": 10,
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert search.await_count == 1, "the cap did not stop the second search"
    assert [_offers_search(kwargs) for kwargs in captured] == [True, False]
    assert captured[1]["messages"][-1] == {"role": "user", "content": FINAL_ROUND_NOTE}
    [choice] = response.json()["choices"]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")


def test_a_model_that_keeps_searching_past_web_search_max_calls_answers_one_round_later(
    client: TestClient,
    api_key_header: dict[str, str],
    serve_search_on_fake: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment's shared call limit withholds the search as ``max_uses`` does.

    With ``web_search_max_calls: 1`` and no ``max_uses``, the round after the one
    search no longer offers it, and is the last since no other tool is left.
    """
    serve_search_on_fake(extract=False)
    monkeypatch.setattr(cast(Any, client.app).state.config, "web_search_max_calls", 1)
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.append(kwargs)
        return _completion(tool_call=True)

    with (
        patch("gateway.services.mcp_loop.acompletion", new=fake_acompletion),
        admitted_search_spy() as search,
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "otari_web_search"}],
                "max_tool_iterations": 10,
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert search.await_count == 1, "the call limit did not stop the second search"
    assert [_offers_search(kwargs) for kwargs in captured] == [True, False]
    assert captured[1]["messages"][-1] == {"role": "user", "content": FINAL_ROUND_NOTE}
    [choice] = response.json()["choices"]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")


def test_max_uses_zero_offers_no_search_rather_than_uncapping_it(
    client: TestClient,
    api_key_header: dict[str, str],
    serve_search_on_fake: Callable[..., None],
) -> None:
    """A cap of zero is a cap: the only round offers no search, and the backend is never reached."""
    serve_search_on_fake(extract=False)
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.append(kwargs)
        return _completion(tool_call=False)

    with (
        patch("gateway.services.mcp_loop.acompletion", new=fake_acompletion),
        admitted_search_spy() as search,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_INTERCEPT": "true"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 0}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert search.await_count == 0, "a cap of zero was read as no cap"
    assert [_offers_search(kwargs) for kwargs in captured] == [False]


@pytest.mark.parametrize("invalid_max_uses", [-1, True, False, "2", 1.5])
def test_invalid_max_uses_is_rejected_instead_of_becoming_uncapped(
    client: TestClient,
    api_key_header: dict[str, str],
    invalid_max_uses: Any,
    serve_search_on_fake: Callable[..., None],
) -> None:
    serve_search_on_fake(extract=False)

    with (
        admitted_search_spy() as search,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_INTERCEPT": "true"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
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


def test_auto_claims_a_search_keyword_no_provider_runs_without_interception(
    client: TestClient,
    api_key_header: dict[str, str],
    serve_search_on_fake: Callable[..., None],
) -> None:
    """Chat Completions has no native search, so `Otari-Web-Search: auto` runs the
    keyword on the backend rather than sending it to a provider that cannot serve it."""
    serve_search_on_fake(extract=False)
    with (
        patch(
            "gateway.services.mcp_loop.acompletion",
            new=AsyncMock(side_effect=[_completion(tool_call=True), _completion(tool_call=False)]),
        ),
        admitted_search_spy() as search,
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305"}],
            },
            headers={**api_key_header, WEB_SEARCH_HEADER: "auto"},
        )

    assert response.status_code == 200, response.text
    assert search.await_count == 1, "the keyword was forwarded instead of claimed"


def test_auto_claims_anthropics_fetch_keyword_on_chat_completions(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """Chat Completions has no native fetch, so `Otari-Web-Fetch: auto` runs the keyword on Otari."""
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)
    fetch = AsyncMock(return_value=_page())
    with (
        patch(
            "gateway.services.mcp_loop.acompletion",
            new=AsyncMock(
                side_effect=[
                    _completion(tool_call=True, tool="web_fetch", arguments='{"url": "https://example.com"}'),
                    _completion(tool_call=False),
                ]
            ),
        ),
        patch.object(WebFetchService, "fetch", new=fetch),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read example.com"}],
                "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
            },
            headers={**api_key_header, WEB_FETCH_HEADER: "auto"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert fetch.await_count == 1, "the keyword was forwarded instead of claimed"
