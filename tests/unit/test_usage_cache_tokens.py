"""Unit tests for cache-token capture in the usage carrier and parse sites.

Also covers which cached-token convention a *stored* row is repriced under
(mozilla-ai/otari#690): the recorded column when it has one, the meters the
pricing wrote when it does not.
"""

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
    PromptTokensDetails,
)

from gateway.api.routes.chat import _ChatAdapter
from gateway.api.routes.messages import _messages_stream_usage, _MessagesAdapter, _requested_cache_write_ttl
from gateway.api.routes.responses import _usage_to_completion_usage
from gateway.core.usage import GatewayUsage, cache_read_tokens_of, cache_write_1h_tokens_of, cache_write_tokens_of
from gateway.models.entities import UsageLog
from gateway.services.usage_admin_service import _row_cache_tokens_included


def test_gateway_usage_defaults_to_zero() -> None:
    usage = GatewayUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0


def test_requested_cache_write_ttl_detects_nested_and_top_level_controls() -> None:
    assert _requested_cache_write_ttl({"type": "ephemeral"}) == "5m"
    assert _requested_cache_write_ttl([{"cache_control": {"type": "ephemeral", "ttl": "1h"}}]) == "1h"
    assert (
        _requested_cache_write_ttl(
            {"cache_control": {"type": "ephemeral"}},
            [{"cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        )
        == "1h"
    )
    assert _requested_cache_write_ttl({"type": "text", "text": "uncached"}) is None


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


def test_messages_non_stream_bills_compaction_iterations() -> None:
    from any_llm.types.messages import MessageResponse, MessageUsage

    result = MessageResponse.model_construct(
        usage=MessageUsage.model_validate(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 7,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 3,
                            "ephemeral_1h_input_tokens": 4,
                        },
                    },
                    {
                        "type": "message",
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 2,
                        "cache_creation_input_tokens": 3,
                    },
                ],
            }
        )
    )

    usage = _MessagesAdapter().extract_usage(result)

    assert isinstance(usage, GatewayUsage)
    assert usage.prompt_tokens == 110
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 135
    assert usage.cache_read_tokens == 32
    assert usage.cache_write_tokens == 10
    assert cache_write_1h_tokens_of(usage) == 4


def test_messages_non_stream_captures_1h_cache_write_breakdown() -> None:
    """Anthropic's optional TTL breakdown is retained for its distinct rate."""
    from anthropic.types.cache_creation import CacheCreation
    from anthropic.types.usage import Usage
    from any_llm.types.messages import MessageResponse

    result = MessageResponse.model_construct(
        usage=Usage(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=15,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=10),
        ),
    )

    usage = _MessagesAdapter().extract_usage(result)

    assert isinstance(usage, GatewayUsage)
    assert usage.cache_write_tokens == 15
    assert cache_write_1h_tokens_of(usage) == 10


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


def test_messages_stream_delta_bills_compaction_iterations() -> None:
    from any_llm.types.messages import MessageDeltaEvent, MessageDeltaUsage

    event = MessageDeltaEvent.model_construct(
        usage=MessageDeltaUsage.model_validate(
            {
                "input_tokens": None,
                "output_tokens": 5,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 7,
                    },
                    {
                        "type": "message",
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 2,
                        "cache_creation_input_tokens": 3,
                    },
                ],
            }
        )
    )

    usage = _messages_stream_usage(event)

    assert isinstance(usage, GatewayUsage)
    assert usage.prompt_tokens == 110
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 135
    assert usage.cache_read_tokens == 32
    assert usage.cache_write_tokens == 10


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


def _row(**overrides: object) -> UsageLog:
    """A stored usage row carrying 1000 prompt tokens with 500 of them cached."""
    fields: dict[str, object] = {
        "id": "row",
        "model": "gpt-4",
        "endpoint": "external",
        "status": "success",
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cache_read_tokens": 500,
    }
    fields.update(overrides)
    return UsageLog(**fields)


def test_recorded_convention_answers_without_the_meters() -> None:
    """The column is the direct record, so it decides even with no meters to recover from.

    This is the case the recovery cannot reach: a row imported without pricing
    (no rate row for its model) has no billing meters at all.
    """
    assert _row_cache_tokens_included(_row(cache_tokens_in_prompt=True)) is True
    assert _row_cache_tokens_included(_row(cache_tokens_in_prompt=False)) is False


def test_recorded_convention_outranks_the_meter_recovery() -> None:
    """A recorded convention is read, not re-derived from numbers that only imply it."""
    inclusive_meters = {"total_input_tokens": 1000, "fresh_input_tokens": 500}
    additive_meters = {"total_input_tokens": 1500, "fresh_input_tokens": 1000}
    assert _row_cache_tokens_included(_row(cache_tokens_in_prompt=True, billing_meters=additive_meters)) is True
    assert _row_cache_tokens_included(_row(cache_tokens_in_prompt=False, billing_meters=inclusive_meters)) is False


def test_unrecorded_convention_falls_back_to_the_meters() -> None:
    """NULL means "not recorded", which is what keeps rows written before the column working."""
    assert _row_cache_tokens_included(_row(billing_meters={"total_input_tokens": 1000})) is True
    assert _row_cache_tokens_included(_row(billing_meters={"total_input_tokens": 1500})) is False
    # Neither a flag nor meters: nothing recorded the convention, so the ingest
    # default (the additive Claude Code shape) is the only answer left.
    assert _row_cache_tokens_included(_row()) is False
