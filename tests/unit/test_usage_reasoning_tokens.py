"""Unit tests for reasoning-token capture in the usage carrier, parse sites and stream merge."""

from any_llm.types.completion import ChatCompletionChunk, CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails

from gateway.api.routes.chat import _ChatAdapter
from gateway.api.routes.messages import _messages_stream_usage, _MessagesAdapter
from gateway.api.routes.responses import _usage_to_completion_usage
from gateway.core.usage import GatewayUsage, reasoning_tokens_of
from gateway.streaming import _merge_usage


def _openai_usage(reasoning: int) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning),
    )


def test_reasoning_tokens_default_to_zero() -> None:
    assert GatewayUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2).reasoning_tokens == 0
    assert reasoning_tokens_of(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)) == 0


def test_from_completion_usage_reads_completion_details() -> None:
    assert GatewayUsage.from_completion_usage(_openai_usage(12)).reasoning_tokens == 12


def test_from_completion_usage_honors_explicit_value() -> None:
    assert GatewayUsage.from_completion_usage(_openai_usage(12), reasoning_tokens=0).reasoning_tokens == 0


def test_chat_stream_chunk_captures_reasoning_tokens() -> None:
    chunk = ChatCompletionChunk.model_construct(usage=_openai_usage(15))
    usage = _ChatAdapter().extract_stream_usage(chunk)
    assert isinstance(usage, GatewayUsage)
    assert usage.reasoning_tokens == 15


def test_responses_captures_reasoning_tokens() -> None:
    from openai.types.responses import ResponseUsage
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    usage = _usage_to_completion_usage(
        ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=14),
        )
    )
    assert isinstance(usage, GatewayUsage)
    assert usage.reasoning_tokens == 14


def test_messages_reads_thinking_tokens_from_output_details() -> None:
    from any_llm.types.messages import MessageResponse, MessageUsage

    result = MessageResponse.model_construct(
        usage=MessageUsage.model_validate(
            {"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"thinking_tokens": 30}}
        )
    )
    usage = _MessagesAdapter().extract_usage(result)
    assert isinstance(usage, GatewayUsage)
    assert usage.completion_tokens == 50
    assert usage.reasoning_tokens == 30


def test_messages_stream_delta_reads_thinking_tokens() -> None:
    from anthropic.types.message_delta_usage import MessageDeltaUsage
    from any_llm.types.messages import MessageDeltaEvent

    event = MessageDeltaEvent.model_construct(
        usage=MessageDeltaUsage.model_validate({"output_tokens": 30, "output_tokens_details": {"thinking_tokens": 12}})
    )
    usage = _messages_stream_usage(event)
    assert isinstance(usage, GatewayUsage)
    assert usage.reasoning_tokens == 12


def test_merge_usage_keeps_reasoning_tokens() -> None:
    """The stream loop rebuilds usage on every chunk; the count must survive that."""
    start = GatewayUsage(prompt_tokens=100, completion_tokens=0, total_tokens=100)
    delta = GatewayUsage(prompt_tokens=0, completion_tokens=30, total_tokens=30, reasoning_tokens=12)
    merged = _merge_usage(
        _merge_usage(start, delta), GatewayUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )
    assert reasoning_tokens_of(merged) == 12
