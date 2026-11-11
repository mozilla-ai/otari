"""Guardrails that read the model's answer, through the completion routes.

A ``block`` guardrail on ``output`` refuses an answer it flags, and the answer
is still billed, streamed or not. A streamed answer is held until the guardrail
has read it, so nothing it refuses reaches the client. The guardrails service is
stood in for by patching ``run_input_guardrails``, so the routes, the targets
and the answer handling run for real.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.messages import MessageStreamEvent
from any_llm.types.responses import ResponseStreamEvent
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.guardrails import GuardrailConfig
from gateway.models.usage import UsageLog
from gateway.services.guardrails import GuardrailResult, GuardrailVerdict

from .conftest import MODEL_NAME

_RUNNER = "gateway.api.routes._guardrails.run_input_guardrails"
_ON_OUTPUT = {"profile": "mod", "mode": "block", "on": ["output"]}


async def _guardrails(guardrails: list[GuardrailConfig], text: str, **_: object) -> GuardrailVerdict:
    """Flags a text that says "forbidden"."""
    return GuardrailVerdict(
        results=[GuardrailResult(profile=g.profile, mode=g.mode, valid="forbidden" not in text) for g in guardrails]
    )


def _provider(answer: Any) -> Callable[..., Any]:
    async def provider(**_: Any) -> Any:
        return answer() if callable(answer) else answer

    return provider


def _completion(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )


def _chunks(*texts: str) -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    def chunk(delta: dict[str, Any], finish: str | None = None, usage: bool = False) -> ChatCompletionChunk:
        return ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": MODEL_NAME,
                "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18} if usage else None,
            }
        )

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield chunk({"role": "assistant", "content": ""})
        for text in texts:
            yield chunk({"content": text})
        yield chunk({}, "stop")
        yield chunk({}, usage=True)

    return stream


def _message_events(*texts: str) -> Callable[[], AsyncIterator[MessageStreamEvent]]:
    adapter: TypeAdapter[MessageStreamEvent] = TypeAdapter(MessageStreamEvent)
    raw: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 11, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}} for t in texts),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]

    async def stream() -> AsyncIterator[MessageStreamEvent]:
        for event in raw:
            yield adapter.validate_python(event)

    return stream


def _response_events(*texts: str) -> Callable[[], AsyncIterator[ResponseStreamEvent]]:
    adapter: TypeAdapter[ResponseStreamEvent] = TypeAdapter(ResponseStreamEvent)
    response = {
        "id": "resp_1",
        "created_at": 0.0,
        "model": "gpt-4o-mini",
        "object": "response",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    raw: list[dict[str, Any]] = [
        *(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": t,
                "sequence_number": i,
                "logprobs": [],
            }
            for i, t in enumerate(texts)
        ),
        {"type": "response.completed", "response": response, "sequence_number": len(texts)},
    ]

    async def stream() -> AsyncIterator[ResponseStreamEvent]:
        for event in raw:
            yield adapter.validate_python(event)

    return stream


# Each completion endpoint: its path, the provider call it makes, its stream, and a body.
_STREAMS: dict[str, tuple[str, str, Callable[..., Callable[[], AsyncIterator[Any]]], dict[str, Any]]] = {
    "chat": (
        "/chat/completions",
        "gateway.api.routes.chat.acompletion",
        _chunks,
        {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
    ),
    "messages": (
        "/messages",
        "gateway.api.routes.messages.amessages",
        _message_events,
        {
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    "responses": (
        "/responses",
        "gateway.api.routes.responses.aresponses",
        _response_events,
        {"model": "openai:gpt-4o-mini", "input": "hi"},
    ),
}


def _sse_text(body: str) -> str:
    """Every streamed text delta, whichever wire format carried it."""
    text = ""
    for line in body.splitlines():
        if not line.startswith("data: {"):
            continue
        event = json.loads(line[6:])
        for choice in event.get("choices") or []:
            text += (choice.get("delta") or {}).get("content") or ""
        if event.get("type") == "content_block_delta":
            text += event["delta"].get("text") or ""
        if event.get("type") == "response.output_text.delta":
            text += event.get("delta") or ""
    return text


def test_an_answer_a_guardrail_refuses_is_still_billed(
    client: TestClient, api_key_header: dict[str, str], db_session: Session
) -> None:
    with (
        patch(_RUNNER, new=_guardrails),
        patch("gateway.api.routes.chat.acompletion", new=_provider(_completion("a forbidden answer"))),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "guardrails": [_ON_OUTPUT]},
            headers=api_key_header,
        )

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "guardrail_violation"
    assert "forbidden answer" not in response.text
    [row] = db_session.scalars(select(UsageLog)).all()
    assert (row.prompt_tokens, row.completion_tokens, row.status, row.status_code) == (11, 7, "error", 403)


@pytest.mark.parametrize("endpoint", list(_STREAMS))
def test_a_streamed_answer_a_guardrail_refuses_ends_in_an_error_and_is_still_billed(
    client: TestClient, api_key_header: dict[str, str], db_session: Session, endpoint: str
) -> None:
    path, provider, events, body = _STREAMS[endpoint]

    with patch(_RUNNER, new=_guardrails), patch(provider, new=_provider(events("a forb", "idden answer"))):
        response = client.post(
            f"{API_ROOT}{path}", json={**body, "stream": True, "guardrails": [_ON_OUTPUT]}, headers=api_key_header
        )
        response.read()

    assert response.status_code == 200
    assert "forb" not in _sse_text(response.text)
    assert "guardrail_violation" in response.text
    [row] = db_session.scalars(select(UsageLog)).all()
    assert (row.prompt_tokens, row.completion_tokens, row.status_code) == (11, 7, 403)


@pytest.mark.parametrize("endpoint", list(_STREAMS))
def test_a_streamed_answer_a_guardrail_lets_through_arrives_whole(
    client: TestClient, api_key_header: dict[str, str], endpoint: str
) -> None:
    path, provider, events, body = _STREAMS[endpoint]

    with patch(_RUNNER, new=_guardrails), patch(provider, new=_provider(events("a fine ", "answer"))):
        response = client.post(
            f"{API_ROOT}{path}", json={**body, "stream": True, "guardrails": [_ON_OUTPUT]}, headers=api_key_header
        )
        response.read()

    assert response.status_code == 200, response.text
    assert _sse_text(response.text) == "a fine answer"
    assert "guardrail_violation" not in response.text


@pytest.mark.parametrize("mode", ["monitor", "block"])
def test_a_guardrail_that_reads_only_the_answer_does_not_read_the_input(
    client: TestClient, api_key_header: dict[str, str], mode: str
) -> None:
    seen: list[str] = []

    async def _recording(guardrails: list[GuardrailConfig], text: str, **kwargs: object) -> GuardrailVerdict:
        seen.append(text)
        return await _guardrails(guardrails, text, **kwargs)

    with (
        patch(_RUNNER, new=_recording),
        patch("gateway.api.routes.chat.acompletion", new=_provider(_completion("fine"))),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "forbidden question"}],
                "guardrails": [{"profile": "mod", "mode": mode, "on": ["output"]}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert seen == ["fine"]
