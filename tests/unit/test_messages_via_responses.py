"""Unit tests for serving a refused Messages request through the Responses API.

OpenAI's Chat Completions refuses function tools combined with a reasoning
effort on some models (otari#1630). The fallback must retry exactly that refusal
through Responses, translate both directions faithfully, and leave every other
failure as it was.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from any_llm.exceptions import InvalidRequestError
from any_llm.types.messages import MessageResponse
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)

from gateway.services.mcp_loop_messages import _MessagesToolLoopStrategy
from gateway.services.providers import messages_via_responses
from gateway.services.providers.messages_via_responses import (
    ResponsesStreamFailedError,
    amessages_with_responses_fallback,
)

MODEL = "openai:gpt-6-luna"
REFUSAL = (
    "Function tools with reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions. "
    "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
)
TOOL = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _bad_request(message: str = REFUSAL, param: str | None = "reasoning_effort") -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=request),
        body={"message": message, "type": "invalid_request_error", "param": param, "code": None},
    )


def _refusing(exc: BaseException) -> Any:
    async def call(**_kwargs: Any) -> Any:
        raise exc

    return call


def _kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "api_key": "sk-test",
        "max_tokens": 256,
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "look it up", "signature": "sig"},
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}]},
        ],
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        **overrides,
    }


def _response(output: list[dict[str, Any]], **extra: Any) -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-6-luna",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 20},
                "total_tokens": 130,
            },
            **extra,
        }
    )


_FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_2",
    "name": "get_weather",
    "arguments": '{"city": "Lyon"}',
    "status": "completed",
}
_MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Checking Lyon too.", "annotations": []}],
}


@pytest.mark.asyncio
async def test_a_refused_request_is_answered_through_responses() -> None:
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return _response([_MESSAGE, _FUNCTION_CALL])

    with patch.object(messages_via_responses, "aresponses", new=fake_aresponses):
        result = await amessages_with_responses_fallback(_refusing(_bad_request()), _kwargs())

    assert isinstance(result, MessageResponse)
    assert [block.type for block in result.content] == ["text", "tool_use"]
    assert result.content[1].input == {"city": "Lyon"}  # type: ignore[union-attr]
    assert result.content[1].id == "call_2"  # type: ignore[union-attr]
    assert result.stop_reason == "tool_use"
    # Cached tokens are split out of the input total, as the Chat bridge reports them.
    assert result.usage.input_tokens == 60
    assert result.usage.cache_read_input_tokens == 40
    assert result.usage.output_tokens == 30

    # The effort is kept, not dropped, and credentials pass through untouched.
    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["api_key"] == "sk-test"
    assert captured["model"] == MODEL
    assert captured["max_output_tokens"] == 256
    assert captured["store"] is False
    assert captured["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": TOOL["input_schema"],
            "strict": False,
        }
    ]
    assert captured["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert captured["input_data"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]


@pytest.mark.asyncio
async def test_the_unified_exception_wrapper_is_recognized_too() -> None:
    wrapped = InvalidRequestError(REFUSAL, original_exception=_bad_request(), provider_name="openai")

    async def fake_aresponses(**_kwargs: Any) -> Response:
        return _response([_MESSAGE])

    with patch.object(messages_via_responses, "aresponses", new=fake_aresponses):
        result = await amessages_with_responses_fallback(_refusing(wrapped), _kwargs())

    assert result.stop_reason == "end_turn"
    assert result.content[0].text == "Checking Lyon too."


@pytest.mark.asyncio
async def test_an_output_cap_reads_as_max_tokens() -> None:
    async def fake_aresponses(**_kwargs: Any) -> Response:
        return _response([], status="incomplete", incomplete_details={"reason": "max_output_tokens"})

    with patch.object(messages_via_responses, "aresponses", new=fake_aresponses):
        result = await amessages_with_responses_fallback(_refusing(_bad_request()), _kwargs())

    assert result.stop_reason == "max_tokens"


@pytest.mark.parametrize(
    ("exc", "kwargs"),
    [
        pytest.param(_bad_request(param="tools"), _kwargs(), id="another-param"),
        pytest.param(_bad_request(message="reasoning_effort 'ultra' is invalid"), _kwargs(), id="another-message"),
        pytest.param(_bad_request(), _kwargs(model="mistral:mistral-large-latest"), id="no-responses-api"),
        pytest.param(_bad_request(), _kwargs(stop_sequences=["END"]), id="untranslatable-field"),
        pytest.param(
            _bad_request(),
            _kwargs(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}}
                        ],
                    }
                ]
            ),
            id="unmapped-content-part",
        ),
    ],
)
@pytest.mark.asyncio
async def test_other_failures_keep_their_error(exc: BaseException, kwargs: dict[str, Any]) -> None:
    async def fake_aresponses(**_kwargs: Any) -> Response:
        raise AssertionError("must not retry")

    with patch.object(messages_via_responses, "aresponses", new=fake_aresponses), pytest.raises(type(exc)) as raised:
        await amessages_with_responses_fallback(_refusing(exc), kwargs)

    assert raised.value is exc


@pytest.mark.asyncio
async def test_a_successful_call_is_returned_unchanged() -> None:
    sentinel = object()

    async def call(**_kwargs: Any) -> Any:
        return sentinel

    assert await amessages_with_responses_fallback(call, _kwargs()) is sentinel


def _stream_events() -> list[Any]:
    created = _response([], status="in_progress", usage=None)
    return [
        ResponseCreatedEvent(type="response.created", response=created, sequence_number=0),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=0,
            sequence_number=1,
            item={**_MESSAGE, "status": "in_progress", "content": []},  # type: ignore[arg-type]
        ),
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            item_id="msg_1",
            delta="Checking.",
            logprobs=[],
            sequence_number=2,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            output_index=0,
            sequence_number=3,
            item=_MESSAGE,  # type: ignore[arg-type]
        ),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=1,
            sequence_number=4,
            item={**_FUNCTION_CALL, "arguments": "", "status": "in_progress"},  # type: ignore[arg-type]
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            output_index=1,
            item_id="fc_1",
            delta='{"city": "Lyon"}',
            sequence_number=5,
        ),
        ResponseCompletedEvent(
            type="response.completed",
            response=_response([_MESSAGE, _FUNCTION_CALL]),
            sequence_number=6,
        ),
    ]


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_a_refused_stream_is_translated_from_the_responses_stream() -> None:
    upstream = _FakeStream(_stream_events())
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> _FakeStream:
        captured.update(kwargs)
        return upstream

    with patch.object(messages_via_responses, "aresponses", new=fake_aresponses):
        stream = await amessages_with_responses_fallback(_refusing(_bad_request()), _kwargs(stream=True))
        events = [event async for event in stream]

    assert captured["stream"] is True
    assert [event.type for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1].content_block.type == "text"
    assert events[2].delta.text == "Checking."
    assert events[4].content_block.type == "tool_use"
    assert events[4].content_block.id == "call_2"
    assert events[4].index == 1
    assert events[5].delta.partial_json == '{"city": "Lyon"}'
    closing = events[7]
    assert closing.delta.stop_reason == "tool_use"
    assert (closing.usage.input_tokens, closing.usage.cache_read_input_tokens, closing.usage.output_tokens) == (
        60,
        40,
        30,
    )
    assert upstream.closed


@pytest.mark.asyncio
async def test_a_failed_responses_stream_raises_without_the_upstream_text() -> None:
    failed = {"type": "response.failed", "response": _response([], status="failed")}
    upstream = _FakeStream([_stream_events()[0], type("Event", (), failed)()])

    async def fake_aresponses(**_kwargs: Any) -> _FakeStream:
        return upstream

    with patch.object(messages_via_responses, "aresponses", new=fake_aresponses):
        stream = await amessages_with_responses_fallback(_refusing(_bad_request()), _kwargs(stream=True))
        seen: list[str] = []
        # Collected one at a time: the events before the failure are the point.
        with pytest.raises(ResponsesStreamFailedError):
            while True:
                seen.append((await anext(stream)).type)

    # The usage the failed attempt reported is still flushed so it can be billed.
    assert seen == ["message_start", "message_delta"]
    assert upstream.closed


@pytest.mark.asyncio
async def test_the_gateway_tool_loop_falls_back_too() -> None:
    async def fake_aresponses(**_kwargs: Any) -> Response:
        return _response([_FUNCTION_CALL])

    with (
        patch("gateway.services.mcp_loop_messages.amessages", new=_refusing(_bad_request())),
        patch.object(messages_via_responses, "aresponses", new=fake_aresponses),
    ):
        result = await _MessagesToolLoopStrategy().call(_kwargs())

    assert result.stop_reason == "tool_use"
