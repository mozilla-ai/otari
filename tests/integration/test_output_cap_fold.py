"""The chat endpoint dispatches one output cap, whichever field the caller sent.

mozilla-ai/otari-ai#1062: OpenAI renamed the cap to ``max_completion_tokens`` and
deprecated ``max_tokens``, so an unmodified OpenAI-compatible client sends the
current name. Both are real any-llm ``CompletionParams`` fields, so both used to
reach the provider call verbatim, and a provider whose param conversion predates
the rename either raised (Anthropic's SDK takes no such keyword, which surfaced
as ``502 {"detail": "LLM provider error"}`` for every upstream) or dropped the
cap without saying so (Google's honors ``max_tokens`` alone).

These assert on the kwargs the gateway hands ``acompletion``, because that is
where the bug lived: the request was accepted, validated, and then forwarded in
a shape the provider could not take. The last test covers the other half of the
fix, a provider that rejects a param the gateway does forward.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from fastapi.testclient import TestClient

from .conftest import MODEL_NAME

_MESSAGES = [{"role": "user", "content": "hi"}]


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-cap",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _dispatch(client: TestClient, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    """POST a chat completion and return the kwargs the gateway dispatched."""
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": _MESSAGES, **body},
            headers=headers,
        )
    assert response.status_code == 200, response.text
    return captured


@pytest.mark.asyncio
async def test_current_name_is_dispatched_as_max_tokens(client: TestClient, api_key_header: dict[str, str]) -> None:
    """The reported bug: the cap now reaches the provider under the name every
    provider understands, and the name that broke them is gone from the call."""
    kwargs = _dispatch(client, api_key_header, {"max_completion_tokens": 20})
    assert kwargs["max_tokens"] == 20
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_legacy_name_is_dispatched_unchanged(client: TestClient, api_key_header: dict[str, str]) -> None:
    """The deprecated spelling still works exactly as it did."""
    kwargs = _dispatch(client, api_key_header, {"max_tokens": 20})
    assert kwargs["max_tokens"] == 20
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_current_name_wins_when_a_request_sends_both(client: TestClient, api_key_header: dict[str, str]) -> None:
    """One cap is dispatched, not two, and it is the current field's value."""
    kwargs = _dispatch(client, api_key_header, {"max_tokens": 300, "max_completion_tokens": 20})
    assert kwargs["max_tokens"] == 20
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_no_cap_is_dispatched_when_the_caller_sends_none(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The fold adds no cap of its own: a request with neither field leaves the
    provider's own default in charge."""
    kwargs = _dispatch(client, api_key_header, {})
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_explicit_null_current_name_is_not_forwarded(client: TestClient, api_key_header: dict[str, str]) -> None:
    """An explicit null is a value pydantic keeps in ``exclude_unset`` dumps, so
    it too must not travel under the name that breaks providers."""
    kwargs = _dispatch(client, api_key_header, {"max_completion_tokens": None})
    assert "max_completion_tokens" not in kwargs
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_streaming_dispatches_the_folded_cap(client: TestClient, api_key_header: dict[str, str]) -> None:
    """The streaming path builds its call kwargs from the same request fields, so
    it gets the same fold; it dispatches separately enough to be worth pinning."""
    captured: dict[str, Any] = {}

    async def chunk_stream() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="chatcmpl-cap",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    async def mock_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        captured.update(kwargs)
        return chunk_stream()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": _MESSAGES,
                "max_completion_tokens": 20,
                "stream": True,
            },
            headers=api_key_header,
        )
        assert response.status_code == 200, response.text
        response.read()

    assert captured["max_tokens"] == 20
    assert "max_completion_tokens" not in captured


@pytest.mark.asyncio
async def test_param_the_provider_cannot_take_is_a_400_naming_it(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The rest of the class the reported bug belonged to: a param the gateway
    does forward, against a provider whose SDK has no such keyword. It is
    permanent and the caller's to fix, so it is a 400 that names the param
    rather than a 502 that reads as an upstream outage."""

    async def mock_acompletion(**_kwargs: Any) -> ChatCompletion:
        raise TypeError("AsyncMessages.create() got an unexpected keyword argument 'seed'")

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": _MESSAGES, "seed": 7},
            headers=api_key_header,
        )

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "'seed'" in detail
    assert "AsyncMessages" not in detail
