"""Unit tests for reasoning/thinking token capture in the usage carrier, parse sites, and pricing."""

import pytest
from any_llm.types.completion import (
    ChatCompletionChunk,
    CompletionUsage,
)
from openai.types.completion_usage import CompletionTokensDetails

from gateway.api.routes.chat import _ChatAdapter
from gateway.api.routes.messages import _messages_stream_usage, _MessagesAdapter
from gateway.api.routes.responses import _usage_to_completion_usage
from gateway.core.usage import GatewayUsage, reasoning_tokens_of
from gateway.models.entities import ModelPricing
from gateway.services.metered_pricing import calculate_metered_cost


def test_gateway_usage_reasoning_tokens_defaults_to_zero() -> None:
    usage = GatewayUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.reasoning_tokens == 0


def test_from_completion_usage_reads_reasoning_tokens_fallback() -> None:
    base = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=12),
    )
    usage = GatewayUsage.from_completion_usage(base)
    assert usage.reasoning_tokens == 12


def test_from_completion_usage_explicit_overrides_fallback() -> None:
    base = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=12),
    )
    usage = GatewayUsage.from_completion_usage(base, reasoning_tokens=5)
    assert usage.reasoning_tokens == 5


def test_from_completion_usage_honors_explicit_zero() -> None:
    base = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=12),
    )
    usage = GatewayUsage.from_completion_usage(base, reasoning_tokens=0)
    assert usage.reasoning_tokens == 0


def test_reasoning_tokens_of_helper_on_plain_completion_usage() -> None:
    plain = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=8),
    )
    assert reasoning_tokens_of(plain) == 8


def test_reasoning_tokens_of_helper_without_details() -> None:
    plain = CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    assert reasoning_tokens_of(plain) == 0


def test_chat_stream_captures_reasoning_tokens() -> None:
    chunk = ChatCompletionChunk.model_construct(
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=15),
        ),
    )
    usage = _ChatAdapter().extract_stream_usage(chunk)
    assert isinstance(usage, GatewayUsage)
    assert usage.reasoning_tokens == 15


def test_responses_captures_reasoning_tokens() -> None:
    from openai.types.responses import ResponseUsage
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    usage_in = ResponseUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=InputTokensDetails(cached_tokens=33),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=14),
    )
    usage = _usage_to_completion_usage(usage_in)
    assert isinstance(usage, GatewayUsage)
    assert usage.reasoning_tokens == 14
    assert usage.cache_read_tokens == 33


def test_messages_non_stream_captures_thinking_tokens() -> None:
    from any_llm.types.messages import MessageResponse, MessageUsage

    # Simulate Anthropic response usage containing thinking_tokens/reasoning_tokens
    result = MessageResponse.model_construct(
        usage=MessageUsage.model_validate(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "thinking_tokens": 30,
            }
        )
    )
    usage = _MessagesAdapter().extract_usage(result)
    assert isinstance(usage, GatewayUsage)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.reasoning_tokens == 30


def test_messages_stream_delta_captures_thinking_tokens() -> None:
    from anthropic.types.message_delta_usage import MessageDeltaUsage
    from any_llm.types.messages import MessageDeltaEvent

    event = MessageDeltaEvent.model_construct(
        usage=MessageDeltaUsage.model_validate(
            {
                "input_tokens": 10,
                "output_tokens": 30,
                "thinking_tokens": 12,
            }
        )
    )
    usage = _messages_stream_usage(event)
    assert isinstance(usage, GatewayUsage)
    assert usage.completion_tokens == 30
    assert usage.reasoning_tokens == 12


def test_messages_stream_start_captures_thinking_tokens() -> None:
    from any_llm.types.messages import MessageResponse, MessageStartEvent, MessageUsage

    msg = MessageResponse.model_construct(
        usage=MessageUsage.model_validate(
            {
                "input_tokens": 100,
                "output_tokens": 0,
                "thinking_tokens": 25,
            }
        )
    )
    event = MessageStartEvent.model_construct(message=msg)
    usage = _messages_stream_usage(event)
    assert isinstance(usage, GatewayUsage)
    assert usage.prompt_tokens == 100
    assert usage.reasoning_tokens == 25


def test_metered_pricing_adds_reasoning_tokens_to_meters_but_does_not_double_bill() -> None:
    pricing = ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=5.0,
        output_price_per_million=15.0,
    )
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        reasoning_tokens=200,
    )
    cost, meters, breakdown = calculate_metered_cost(pricing, usage)

    # Expected cost should only bill input (1000) and completion (500) tokens.
    # Reasoning tokens (200) are a subset of completion tokens and do not have their own pricing rate.
    expected_cost = (1000 / 1_000_000) * 5.0 + (500 / 1_000_000) * 15.0
    assert cost == pytest.approx(expected_cost)

    assert meters["reasoning_tokens"] == 200
    assert meters["completion_tokens"] == 500
    assert meters["total_input_tokens"] == 1000
