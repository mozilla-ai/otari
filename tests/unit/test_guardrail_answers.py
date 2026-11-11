"""The model's answer as a request's guardrails read it: its parts, and a stream held until they are checked."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage
from any_llm.types.messages import MessageResponse
from fastapi import Response

from gateway.api.routes._guardrail_answers import answer_parts, finish_result, guarded_stream
from gateway.api.routes._guardrails import AnswerRefused, RequestGuardrails
from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrails import GuardrailResult, GuardrailVerdict


def _completion(content: str | None, arguments: str | None = None) -> ChatCompletion:
    calls = (
        [{"id": "call_1", "type": "function", "function": {"name": "send_mail", "arguments": arguments}}]
        if arguments is not None
        else None
    )
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "tool_calls": calls},
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
        }
    )


def _chunk(content: str | None = None, *, finish: str | None = None, usage: bool = False) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [] if usage else [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7} if usage else None,
        }
    )


async def _events(*events: Any) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
    return chunk.usage


async def _flag(guardrails: list[GuardrailConfig], text: str, **_: object) -> GuardrailVerdict:
    return GuardrailVerdict(
        results=[GuardrailResult(profile=g.profile, mode=g.mode, valid="forbidden" not in text) for g in guardrails]
    )


def _on_output() -> RequestGuardrails:
    return RequestGuardrails([GuardrailConfig(profile="mod", mode="block", on=["output"])], response=Response())


def test_answer_parts_of_a_chat_answer_include_its_tool_calls() -> None:
    parts = answer_parts("chat", _completion("Sending.", arguments='{"to": "a@b"}'))

    assert [(p.kind, p.text, p.tool) for p in parts] == [
        ("text", "Sending.", None),
        ("tool_call", '{"to": "a@b"}', "send_mail"),
    ]


def test_answer_parts_of_a_messages_answer_include_its_tool_calls_as_json() -> None:
    result = MessageResponse.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [
                {"type": "text", "text": "Sending."},
                {"type": "tool_use", "id": "toolu_1", "name": "send_mail", "input": {"to": "a@b"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    assert [(p.kind, p.text, p.tool) for p in answer_parts("messages", result)] == [
        ("text", "Sending.", None),
        ("tool_call", '{"to": "a@b"}', "send_mail"),
    ]


@pytest.mark.asyncio
async def test_a_held_stream_refused_by_a_guardrail_on_the_answer_carries_its_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gateway.api.routes._guardrails.run_input_guardrails", _flag)
    chunks = [_chunk("a forb"), _chunk("idden answer", finish="stop"), _chunk(usage=True)]
    out: list[Any] = []

    with pytest.raises(AnswerRefused) as exc:
        out.extend([event async for event in guarded_stream(_events(*chunks), _on_output(), "chat", _usage)])

    assert out == []
    assert exc.value.usage is not None
    assert exc.value.usage.completion_tokens == 4


@pytest.mark.asyncio
async def test_a_held_stream_the_guardrails_let_through_is_sent_on_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def _record(guardrails: list[GuardrailConfig], text: str, **kwargs: object) -> GuardrailVerdict:
        seen.append(text)
        return await _flag(guardrails, text, **kwargs)

    monkeypatch.setattr("gateway.api.routes._guardrails.run_input_guardrails", _record)
    chunks = [_chunk("the ans"), _chunk("wer is 42", finish="stop")]

    out = [event async for event in guarded_stream(_events(*chunks), _on_output(), "chat", _usage)]

    assert out == chunks
    assert seen == ["the answer is 42"]


@pytest.mark.asyncio
async def test_a_held_messages_stream_does_not_read_a_call_the_provider_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider's own web search streams its query as a client call would, and is left out as it is unstreamed."""
    seen: list[str] = []

    async def _record(guardrails: list[GuardrailConfig], text: str, **kwargs: object) -> GuardrailVerdict:
        seen.append(text)
        return await _flag(guardrails, text, **kwargs)

    def _start(index: int, kind: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_start", index=index, content_block=SimpleNamespace(type=kind, name=name)
        )

    def _json(index: int, partial: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(type="input_json_delta", partial_json=partial),
        )

    monkeypatch.setattr("gateway.api.routes._guardrails.run_input_guardrails", _record)
    guardrails = RequestGuardrails(
        [GuardrailConfig(profile="mod", mode="block", on=["tool_calls"])], response=Response()
    )
    events = [
        _start(0, "server_tool_use", "web_search"),
        _json(0, '{"query": "forbidden"}'),
        _start(1, "tool_use", "send_mail"),
        _json(1, '{"to": "a@b"}'),
    ]

    out = [event async for event in guarded_stream(_events(*events), guardrails, "messages", lambda _: None)]

    assert out == events
    assert seen == ['{"to": "a@b"}']


@pytest.mark.asyncio
async def test_finish_result_runs_the_guardrails_on_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.api.routes._guardrails.run_input_guardrails", _flag)

    await finish_result(_on_output(), "chat", _completion("fine"))
    with pytest.raises(AnswerRefused):
        await finish_result(_on_output(), "chat", _completion("a forbidden answer"))
