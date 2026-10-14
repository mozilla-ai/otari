"""The gateway's web searches and fetches, reported to a Chat Completions client in ``otari_tool_calls``.

The web retrieval backend runs with only its search and its page fetch patched, so the
real admission, backend and chat tool loop run. The provider is scripted to call the
declared tools in one round and answer in the next.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, patch

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
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.tools import OTARI_TOOL_CALLS_FIELD
from gateway.services.web_fetch_service import WebFetchResult, WebFetchService
from gateway.services.web_retrieval_backend import WebRetrievalBackend
from gateway.services.web_retrieval_policy import canonicalize_web_url

from .conftest import MODEL_NAME

_TOOLS = [{"type": "otari_web_search"}, {"type": "otari_web_fetch"}]
_ARGUMENTS = {
    "web_search": {"query": "python release"},
    "web_fetch": {"url": "https://example.com/notes?token=secret#part"},
}
_PAGE_TEXT = "page text only the model reads"


def _page() -> WebFetchResult:
    url = canonicalize_web_url("https://example.com/notes")
    return WebFetchResult(
        text=_PAGE_TEXT,
        content_type="text/html",
        content_kind="html",
        requested_url=url,
        final_url=url,
        redirect_count=0,
    )


@pytest.fixture
def web_tools(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig) -> Iterator[None]:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://web-search.invalid")
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)
    hits = [{"url": "https://python.org", "title": "Python", "content": _PAGE_TEXT, "published_date": "2026-10-07"}]
    with (
        patch.object(WebRetrievalBackend, "_search", new=AsyncMock(return_value=hits)),
        patch.object(WebFetchService, "fetch", new=AsyncMock(return_value=_page())),
        # A search reads each hit's page through the same service; these hits keep their snippets.
        patch.object(WebFetchService, "fetch_for_search", new=AsyncMock(side_effect=OSError)),
    ):
        yield


@contextmanager
def _scripted(target: str, responses: list[Any]) -> Iterator[list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return next(remaining)

    with patch(target, new=fake):
        yield sent


def _completion(names: list[str] | None) -> ChatCompletion:
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{index}", type="function", function=Function(name=name, arguments=json.dumps(_ARGUMENTS[name]))
        )
        for index, name in enumerate(names or [])
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
                    role="assistant", content=None if names else "done", tool_calls=cast(Any, calls or None)
                ),
                finish_reason="tool_calls" if names else "stop",
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


def _streamed_round(names: list[str]) -> AsyncIterator[ChatCompletionChunk]:
    calls = [
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(_ARGUMENTS[name])},
                    }
                ]
            }
        )
        for index, name in enumerate(names)
    ]
    return _events(*calls, _chunk({}, "tool_calls"))


def _streamed_answer() -> AsyncIterator[ChatCompletionChunk]:
    return _events(_chunk({"content": "done"}), _chunk({}, "stop"))


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: {")]


def _body(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL_NAME, "messages": [{"role": "user", "content": "research"}], "tools": _TOOLS, **extra}


@pytest.mark.usefixtures("web_tools")
def test_an_answer_reports_the_search_and_the_fetch_it_ran(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _scripted(
        "gateway.services.mcp_loop.acompletion", [_completion(["web_search", "web_fetch"]), _completion(None)]
    ):
        response = client.post(f"{API_ROOT}/chat/completions", json=_body(), headers=api_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    search, fetch = body[OTARI_TOOL_CALLS_FIELD]
    assert search == {
        "id": search["id"],
        "type": "web_search",
        "status": "completed",
        "query": "python release",
        "sources": [{"url": "https://python.org", "title": "Python", "published_date": "2026-10-07"}],
    }
    assert search["id"].startswith("otari_ws_")
    assert fetch == {"id": fetch["id"], "type": "web_fetch", "status": "completed", "url": "https://example.com/notes"}
    assert fetch["id"].startswith("otari_wf_")
    message = body["choices"][0]["message"]
    assert message["content"] == "done"
    assert OTARI_TOOL_CALLS_FIELD not in message
    assert OTARI_TOOL_CALLS_FIELD not in (message.get("provider_specific_fields") or {})
    assert _PAGE_TEXT not in json.dumps(body[OTARI_TOOL_CALLS_FIELD])


@pytest.mark.usefixtures("web_tools")
def test_a_stream_reports_each_call_as_it_starts_and_ends(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _scripted(
        "gateway.services.mcp_loop.acompletion", [_streamed_round(["web_search", "web_fetch"]), _streamed_answer()]
    ):
        response = client.post(f"{API_ROOT}/chat/completions", json=_body(stream=True), headers=api_key_header)

    assert response.status_code == 200, response.text
    payloads = _sse_payloads(response.text)
    reported = [
        (index, entry) for index, payload in enumerate(payloads) for entry in payload.get(OTARI_TOOL_CALLS_FIELD, [])
    ]
    assert [(entry["type"], entry["status"]) for _, entry in reported] == [
        ("web_search", "in_progress"),
        ("web_search", "completed"),
        ("web_fetch", "in_progress"),
        ("web_fetch", "completed"),
    ]
    assert reported[0][1]["id"] == reported[1][1]["id"]
    assert reported[2][1]["id"] == reported[3][1]["id"]
    first_text = next(index for index, payload in enumerate(payloads) if payload["choices"][0]["delta"].get("content"))
    assert reported[-1][0] < first_text
    for _, entry in reported:
        assert _PAGE_TEXT not in json.dumps(entry)
    for payload in payloads:
        assert not payload["choices"][0]["delta"].get("tool_calls"), "a gateway-run call leaked to the client"


def test_an_echoed_assistant_message_reaches_the_provider_without_the_report(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    echoed = {
        "role": "assistant",
        "content": "Python 3.14 is out.",
        OTARI_TOOL_CALLS_FIELD: [{"id": "otari_ws_1", "type": "web_search", "status": "completed", "query": "q"}],
    }
    with _scripted("gateway.api.routes.chat.acompletion", [_completion(None)]) as sent:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "news?"}, echoed, {"role": "user", "content": "more?"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert sent[0]["messages"][1] == {"role": "assistant", "content": "Python 3.14 is out."}
