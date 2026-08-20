"""Focused tests for inline platform-cost wire placement."""

from typing import Any, cast

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from any_llm.types.messages import (
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageUsage,
    TextBlock,
)
from any_llm.types.responses import Response
from openai.types.responses import ResponseCompletedEvent, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.api.routes import chat, messages, responses
from gateway.api.routes._platform import SettledCost

_SETTLEMENT = SettledCost(cost_usd="0.012345", pricing_source="managed")


def _chat_response(*, usage: CompletionUsage | None = None) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
            )
        ],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=usage,
    )


def _chat_usage_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk-1",
        choices=[],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
    )


def _message_usage() -> MessageUsage:
    return MessageUsage(
        input_tokens=10,
        output_tokens=7,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        cache_creation=None,
        server_tool_use=None,
        service_tier=None,
    )


def _message_response() -> MessageResponse:
    return MessageResponse(
        id="msg-1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-0",
        content=[TextBlock(type="text", text="hi", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=_message_usage(),
        container=None,
    )


def _message_delta() -> MessageDeltaEvent:
    return MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
        usage=MessageDeltaUsage.model_validate(
            {
                "input_tokens": None,
                "output_tokens": 7,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
            }
        ),
    )


def _message_start() -> MessageStartEvent:
    return MessageStartEvent(type="message_start", message=_message_response())


def _response(*, usage: ResponseUsage | None = None) -> Response:
    return Response(
        id="resp-1",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        status="completed",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


def _response_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=7,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=17,
    )


def _response_completed() -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        type="response.completed",
        response=_response(usage=_response_usage()),
        sequence_number=0,
    )


@pytest.mark.parametrize(
    ("adapter", "value", "usage_getter"),
    [
        (
            chat._ADAPTER,
            _chat_response(usage=CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17)),
            lambda value: value.usage,
        ),
        (chat._ADAPTER, _chat_usage_chunk(), lambda value: value.usage),
        (messages._ADAPTER, _message_response(), lambda value: value.usage),
        (messages._ADAPTER, _message_delta(), lambda value: value.usage),
        (responses._ADAPTER, _response(usage=_response_usage()), lambda value: value.usage),
        (responses._ADAPTER, _response_completed(), lambda value: value.response.usage),
    ],
)
def test_adapter_attaches_inline_cost(adapter: Any, value: Any, usage_getter: Any) -> None:
    assert adapter.attach_cost(value, _SETTLEMENT) is True

    usage = usage_getter(value)
    assert usage.cost_usd == "0.012345"
    assert usage.pricing_source == "managed"


@pytest.mark.parametrize(
    ("adapter", "event"),
    [
        (chat._ADAPTER, _chat_usage_chunk()),
        (messages._ADAPTER, _message_delta()),
        (responses._ADAPTER, _response_completed()),
    ],
)
def test_terminal_events_serialize_inline_cost(adapter: Any, event: Any) -> None:
    adapter.attach_cost(event, _SETTLEMENT)

    chunk = adapter.format_chunk(event)

    assert '"cost_usd":"0.012345"' in chunk
    assert '"pricing_source":"managed"' in chunk


def test_format_chunk_unchanged_without_inline_cost() -> None:
    event = _message_delta()
    assert messages._ADAPTER.format_chunk(event) == (
        f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"
    )


def test_chat_carrier_is_only_terminal_usage_chunk() -> None:
    assert chat._ADAPTER.is_stream_cost_carrier(_chat_usage_chunk()) is True
    content_chunk = _chat_usage_chunk().model_copy(
        update={
            "choices": [
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                )
            ]
        }
    )
    assert chat._ADAPTER.is_stream_cost_carrier(content_chunk) is False


def test_message_start_is_not_carrier() -> None:
    assert messages._ADAPTER.is_stream_cost_carrier(_message_start()) is False
    assert messages._ADAPTER.is_stream_cost_carrier(_message_delta()) is True


def test_response_completed_with_usage_is_carrier() -> None:
    assert responses._ADAPTER.is_stream_cost_carrier(_response_completed()) is True


@pytest.mark.parametrize(
    ("adapter", "value"),
    [
        (chat._ADAPTER, _chat_response()),
        (responses._ADAPTER, _response()),
    ],
)
def test_adapter_does_not_synthesize_usage(adapter: Any, value: Any) -> None:
    assert adapter.attach_cost(value, _SETTLEMENT) is False
    assert value.usage is None
