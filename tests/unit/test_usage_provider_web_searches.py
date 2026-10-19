"""Unit tests for the searches a provider ran itself: read off a Messages response, kept across a stream, tallied."""

from anthropic.types.message_delta_usage import MessageDeltaUsage
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import MessageDeltaEvent, MessageResponse, MessageUsage

from gateway.api.routes.messages import _messages_stream_usage, _MessagesAdapter
from gateway.core.usage import GatewayUsage, provider_web_searches_of
from gateway.services.pricing import tool_rate_lookup
from gateway.services.tool_usage import ToolUsageTally
from gateway.streaming import merge_stream_usage


def test_a_response_carries_the_searches_anthropic_ran() -> None:
    result = MessageResponse.model_construct(
        usage=MessageUsage.model_validate(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1},
            }
        )
    )

    usage = _MessagesAdapter().extract_usage(result)

    assert isinstance(usage, GatewayUsage)
    assert usage.provider_web_searches == 2


def test_a_response_with_no_server_tool_use_carries_none() -> None:
    result = MessageResponse.model_construct(usage=MessageUsage.model_validate({"input_tokens": 1, "output_tokens": 1}))

    usage = _MessagesAdapter().extract_usage(result)

    assert usage is not None
    assert provider_web_searches_of(usage) == 0
    assert provider_web_searches_of(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)) == 0


def test_a_stream_keeps_the_count_its_final_delta_reports() -> None:
    """Anthropic's message_delta usage is cumulative, so the last report is the count, not a sum."""
    delta = MessageDeltaEvent.model_construct(
        usage=MessageDeltaUsage.model_validate(
            {"output_tokens": 30, "server_tool_use": {"web_search_requests": 3, "web_fetch_requests": 0}}
        )
    )
    start = GatewayUsage(prompt_tokens=100, completion_tokens=0, total_tokens=100)

    reported = _messages_stream_usage(delta)
    assert reported is not None
    merged = merge_stream_usage(
        merge_stream_usage(start, reported), GatewayUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )

    assert provider_web_searches_of(merged) == 3


def test_the_tally_counts_provider_searches_under_their_own_label() -> None:
    tally = ToolUsageTally()
    tally.record_result("web_search", "results")
    tally.record_provider_searches("anthropic", 2, deployment_paid=False)
    tally.record_provider_searches("anthropic", 0, deployment_paid=False)

    assert tally.billable_calls() == {"web_search": 3}
    [own, provider] = sorted(tally.label_calls(), key=lambda entry: entry.label or "")
    assert (own.label, own.calls) == (None, 1)
    assert (provider.label, provider.rate_keys, provider.calls, provider.unreported_calls) == (
        "anthropic:web_search",
        ("anthropic:web_search",),
        2,
        2,
    )
    assert provider.deployment_paid is False


def test_whose_overrides_price_a_provider_search_follows_the_call_not_the_label() -> None:
    """A search on a tenant's own Anthropic key takes the tenant's override.

    One on the deployment's own key takes only the operator's, as the call's tokens do.
    """
    for deployment_paid in (False, True):
        tally = ToolUsageTally()
        tally.record_provider_searches("anthropic", 1, deployment_paid=deployment_paid)
        [entry] = tally.label_calls()

        lookup = tool_rate_lookup(entry)

        assert lookup is not None
        assert lookup.rate_keys == ("anthropic:web_search",)
        assert lookup.operator_set_only is deployment_paid
