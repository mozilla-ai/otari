"""The region a provider reports running a call in survives every path its usage takes to settlement."""

from any_llm.types.completion import ChatCompletionChunk, CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageUsage,
)

from gateway.api.routes.chat import _ChatAdapter
from gateway.api.routes.messages import _messages_stream_usage
from gateway.core.usage import GatewayUsage
from gateway.streaming import merge_stream_usage


def test_a_completion_keeps_the_region_its_usage_reports() -> None:
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12, inference_geo="us")

    assert GatewayUsage.from_completion_usage(usage).inference_geo == "us"


def test_a_completion_stream_keeps_the_region_its_usage_chunk_reports() -> None:
    chunk = ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=0,
        model="claude",
        choices=[],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12, inference_geo="us"),
    )

    usage = _ChatAdapter().extract_stream_usage(chunk)

    assert usage is not None
    assert usage.inference_geo == "us"


def test_a_messages_stream_keeps_the_region_its_first_usage_reports() -> None:
    start = MessageStartEvent.model_construct(
        type="message_start",
        message=MessageResponse.model_construct(
            usage=MessageUsage(input_tokens=10, output_tokens=0, inference_geo="us")
        ),
    )
    delta = MessageDeltaEvent.model_construct(usage=MessageDeltaUsage.model_validate({"output_tokens": 2}))
    first, last = _messages_stream_usage(start), _messages_stream_usage(delta)
    assert first is not None
    assert last is not None

    assert merge_stream_usage(first, last).inference_geo == "us"
