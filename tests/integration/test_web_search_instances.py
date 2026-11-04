"""In-loop search on a search instance, through the completion routes.

The instance is any-search's ``fake``, named in ``web_search_default_tool``,
with no enrichment, so no case reaches the network. The model is scripted.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock
from fastapi.testclient import TestClient

from gateway.api.deps import get_workspace_web_search_policies
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tools import ResolvedWebSearchConfig, WebSearchCredential, WebTool, WebToolCredentialSource
from gateway.services.tools import OTARI_TOOL_CALLS_FIELD, WEB_SEARCH_HEADER

from .conftest import MODEL_NAME

_INSTANCE = "fake-search"
_HITS = [
    {"url": "https://example.com/a", "title": "A", "snippet": "Snippet A.", "published": "2026-01-15T09:30:00Z"},
    {"url": "https://example.org/b", "title": "B", "snippet": "Snippet B.", "published": "2026-02-01"},
]
_SOURCES = [
    {"url": "https://example.com/a", "title": "A", "published_date": "2026-01-15T09:30:00+00:00"},
    {"url": "https://example.org/b", "title": "B", "published_date": "2026-02-01"},
]


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig) -> Iterator[dict[str, Any]]:
    """The deployment searches on a ``fake`` instance only; a case sets its options in the dict yielded."""
    options: dict[str, Any] = {"hits": _HITS}
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    monkeypatch.setattr(test_config, "search_tools", {_INSTANCE: {"provider": "fake", "options": options}})
    monkeypatch.setattr(test_config, "web_search_default_tool", _INSTANCE)
    monkeypatch.setattr(test_config, "web_search_extract", False)
    yield options


@contextmanager
def _scripted(target: str, responses: list[Any]) -> Iterator[list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return next(remaining)

    with patch(target, new=fake):
        yield sent


def _completion(searches: int) -> ChatCompletion:
    """A model round asking for ``searches`` searches, or the answer when it asks for none."""
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{index}", type="function", function=Function(name="web_search", arguments='{"query": "otari"}')
        )
        for index in range(searches)
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
                    role="assistant", content=None if searches else "done", tool_calls=cast(Any, calls or None)
                ),
                finish_reason="tool_calls" if searches else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


async def _events(*events: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for event in events:
        yield event


def _chat(client: TestClient, headers: dict[str, str], **extra: Any) -> Any:
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "what is otari"}],
        "tools": [{"type": "otari_web_search"}],
        **extra,
    }
    return client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)


def _tool_messages(sent: list[dict[str, Any]]) -> list[str]:
    """What the model was handed for its searches, in the last round it was sent."""
    return [str(message["content"]) for message in sent[-1]["messages"] if message.get("role") == "tool"]


def test_a_chat_search_runs_on_the_instance_and_reports_its_sources(
    client: TestClient, api_key_header: dict[str, str], instance: dict[str, Any]
) -> None:
    with _scripted("gateway.services.mcp_loop.acompletion", [_completion(1), _completion(0)]) as sent:
        response = _chat(client, api_key_header)

    assert response.status_code == 200, response.text
    assert _tool_messages(sent) == [
        "[1] A (2026-01-15T09:30:00+00:00)\nhttps://example.com/a\nSnippet A.\n\n"
        "[2] B (2026-02-01)\nhttps://example.org/b\nSnippet B."
    ]
    [entry] = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert (entry["status"], entry["query"], entry["sources"]) == ("completed", "otari", _SOURCES)


def test_a_streamed_chat_search_is_reported_as_it_starts_and_ends(
    client: TestClient, api_key_header: dict[str, str], instance: dict[str, Any]
) -> None:
    function = {"name": "web_search", "arguments": json.dumps({"query": "otari"})}
    call = {"index": 0, "id": "call_0", "type": "function", "function": function}
    rounds = [
        _events(_chunk({"tool_calls": [call]}), _chunk({}, "tool_calls")),
        _events(_chunk({"content": "done"}), _chunk({}, "stop")),
    ]
    with _scripted("gateway.services.mcp_loop.acompletion", rounds):
        response = _chat(client, api_key_header, stream=True)

    assert response.status_code == 200, response.text
    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    reported = [entry for payload in payloads for entry in payload.get(OTARI_TOOL_CALLS_FIELD, [])]
    assert [entry["status"] for entry in reported] == ["in_progress", "completed"]
    assert reported[1]["sources"] == _SOURCES


def test_a_search_past_the_call_limit_is_refused_and_the_answer_still_comes(
    client: TestClient,
    api_key_header: dict[str, str],
    instance: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """Past ``web_search_max_calls`` the model reads a tool error, where the old backend answered a 422."""
    monkeypatch.setattr(test_config, "web_search_max_calls", 1)
    with _scripted("gateway.services.mcp_loop.acompletion", [_completion(2), _completion(0)]) as sent:
        response = _chat(client, api_key_header)

    assert response.status_code == 200, response.text
    assert _tool_messages(sent)[1] == "[tool error] web search call limit reached for this request"
    first, second = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert first["status"] == "completed"
    assert second["status"] == "failed"
    assert second["error"] == {"provider": None, "status": None, "tag": "call_limit", "retryable": False}


def test_a_providers_failure_is_shown_to_the_model_and_to_the_caller(
    client: TestClient, api_key_header: dict[str, str], instance: dict[str, Any]
) -> None:
    instance.update(error="rate_limited", error_status=429)
    with _scripted("gateway.services.mcp_loop.acompletion", [_completion(1), _completion(0)]) as sent:
        response = _chat(client, api_key_header)

    assert response.status_code == 200, response.text
    assert _tool_messages(sent) == ["[tool error] fake search failed (rate_limited, HTTP 429)"]
    [entry] = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert entry["status"] == "failed"
    assert entry["error"] == {"provider": "fake", "status": 429, "tag": "rate_limited", "retryable": True}


def _message(content: list[Any], stop_reason: str) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=content,
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def test_a_messages_search_on_the_instance_is_cited_with_its_dates(
    client: TestClient, api_key_header: dict[str, str], instance: dict[str, Any]
) -> None:
    rounds = [
        _message(
            [ToolUseBlock(type="tool_use", id="toolu_1", name="web_search", input={"query": "otari"})], "tool_use"
        ),
        _message([TextBlock(type="text", text="done", citations=None)], "end_turn"),
    ]
    with _scripted("gateway.services.mcp_loop_messages.amessages", rounds):
        response = client.post(
            f"{API_ROOT}/messages",
            json={
                "model": "anthropic:claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
            # Anthropic runs its own keyword unless the request hands it to Otari.
            headers={**api_key_header, WEB_SEARCH_HEADER: "otari"},
        )

    assert response.status_code == 200, response.text
    [result] = [block for block in response.json()["content"] if block["type"] == "web_search_tool_result"]
    assert [(hit["url"], hit["page_age"]) for hit in result["content"]] == [
        ("https://example.com/a", "2026-01-15T09:30:00+00:00"),
        ("https://example.org/b", "2026-02-01"),
    ]


def test_a_request_option_the_provider_does_not_take_is_refused_before_the_model_is_called(
    client: TestClient, api_key_header: dict[str, str], instance: dict[str, Any]
) -> None:
    """As on direct search: the request's ``provider_options`` are checked against the instance's provider."""
    tools = [{"type": "otari_web_search", "provider_options": {"no_such_option": 1}}]
    with _scripted("gateway.services.mcp_loop.acompletion", [_completion(0)]) as sent:
        response = _chat(client, api_key_header, tools=tools)

    assert response.status_code == 400, response.text
    assert "no_such_option" in response.text
    assert sent == []


class _PolicyWithKey:
    """A workspace policy that hands the search a key its provider's client cannot use."""

    async def resolve(self, workspace_id: Any, requested_tools: Any) -> ResolvedWebSearchConfig:
        del workspace_id, requested_tools
        credential = WebSearchCredential(
            provider="tavily",
            source=WebToolCredentialSource.ORG,
            api_key="tvly-key",
            api_base="https://tavily-proxy.example",
        )
        return ResolvedWebSearchConfig(
            enabled=True,
            max_results=None,
            purpose_hint=None,
            allowed_domains=None,
            blocked_domains=None,
            provider_options=None,
            authorized_tools=frozenset({WebTool.SEARCH.value}),
            credential=credential,
        )


def test_a_key_the_resolver_refuses_is_refused_at_admission_before_the_model_is_called(
    client: TestClient, api_key_header: dict[str, str], instance: dict[str, Any]
) -> None:
    client.app.dependency_overrides[get_workspace_web_search_policies] = _PolicyWithKey  # type: ignore[attr-defined]
    try:
        with _scripted("gateway.services.mcp_loop.acompletion", [_completion(0)]) as sent:
            response = _chat(client, api_key_header)
    finally:
        client.app.dependency_overrides.pop(get_workspace_web_search_policies, None)  # type: ignore[attr-defined]

    assert response.status_code == 502, response.text
    assert sent == [], "the request was refused before any model call"
