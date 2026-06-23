"""Unit tests for cache-token capture in the usage carrier and parse sites."""

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
    PromptTokensDetails,
)

from gateway.api.routes.chat import _ChatAdapter
from gateway.api.routes.messages import _messages_stream_usage, _MessagesAdapter
from gateway.api.routes.responses import _usage_to_completion_usage
from gateway.core.usage import GatewayUsage, cache_read_tokens_of, cache_write_tokens_of


def test_gateway_usage_defaults_to_zero() -> None:
    usage = GatewayUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0


def test_from_completion_usage_reads_cached_tokens_fallback() -> None:
    base = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=42),
    )
    usage = GatewayUsage.from_completion_usage(base)
    assert usage.cache_read_tokens == 42
    assert usage.cache_write_tokens == 0


def test_from_completion_usage_explicit_overrides_fallback() -> None:
    base = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=42),
    )
    usage = GatewayUsage.from_completion_usage(base, cache_read_tokens=7, cache_write_tokens=3)
    assert usage.cache_read_tokens == 7
    assert usage.cache_write_tokens == 3


def test_from_completion_usage_honors_explicit_zero() -> None:
    """An explicit cache_read_tokens=0 must not be overridden by the fallback."""
    base = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=42),
    )
    usage = GatewayUsage.from_completion_usage(base, cache_read_tokens=0)
    assert usage.cache_read_tokens == 0


def test_from_completion_usage_preserves_gateway_usage_cache_fields() -> None:
    """When the input is already a GatewayUsage, its explicit cache fields are
    carried over (not silently dropped) without re-supplying them."""
    source = GatewayUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cache_read_tokens=11,
        cache_write_tokens=9,
    )
    usage = GatewayUsage.from_completion_usage(source)
    assert usage.cache_read_tokens == 11
    assert usage.cache_write_tokens == 9


def test_cache_helpers_on_plain_completion_usage() -> None:
    plain = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=8),
    )
    assert cache_read_tokens_of(plain) == 8
    assert cache_write_tokens_of(plain) == 0


def test_cache_helpers_on_completion_usage_without_details() -> None:
    plain = CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    assert cache_read_tokens_of(plain) == 0
    assert cache_write_tokens_of(plain) == 0


def test_chat_non_stream_captures_cached_tokens() -> None:
    result = ChatCompletion.model_construct(
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=25),
        ),
    )
    usage = _ChatAdapter().extract_usage(result)
    assert isinstance(usage, GatewayUsage)
    assert usage.cache_read_tokens == 25
    assert usage.cache_write_tokens == 0


def test_chat_stream_captures_cached_tokens() -> None:
    chunk = ChatCompletionChunk.model_construct(
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=11),
        ),
    )
    usage = _ChatAdapter().extract_stream_usage(chunk)
    assert isinstance(usage, GatewayUsage)
    assert usage.cache_read_tokens == 11


def test_messages_non_stream_captures_read_and_write() -> None:
    from anthropic.types.usage import Usage
    from any_llm.types.messages import MessageResponse

    result = MessageResponse.model_construct(
        usage=Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=40,
            cache_creation_input_tokens=15,
        ),
    )
    usage = _MessagesAdapter().extract_usage(result)
    assert isinstance(usage, GatewayUsage)
    assert usage.prompt_tokens == 100
    assert usage.total_tokens == 120
    assert usage.cache_read_tokens == 40
    assert usage.cache_write_tokens == 15


def test_messages_stream_delta_captures_read_and_write() -> None:
    from anthropic.types.message_delta_usage import MessageDeltaUsage
    from any_llm.types.messages import MessageDeltaEvent

    event = MessageDeltaEvent.model_construct(
        usage=MessageDeltaUsage(
            input_tokens=10,
            output_tokens=30,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=2,
        ),
    )
    usage = _messages_stream_usage(event)
    assert isinstance(usage, GatewayUsage)
    assert usage.cache_read_tokens == 5
    assert usage.cache_write_tokens == 2


def test_responses_captures_cached_tokens() -> None:
    from openai.types.responses import ResponseUsage
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    usage_in = ResponseUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=InputTokensDetails(cached_tokens=33),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    usage = _usage_to_completion_usage(usage_in)
    assert isinstance(usage, GatewayUsage)
    assert usage.cache_read_tokens == 33
    assert usage.cache_write_tokens == 0
