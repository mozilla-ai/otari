"""Every dialect carries the tier the provider reports serving a call at, which settlement prices it at."""

from types import SimpleNamespace

from any_llm.types.completion import CompletionUsage

from gateway.api.routes.messages import _billable_messages_usage
from gateway.api.routes.responses import _served_tier, _usage_to_completion_usage
from gateway.core.usage import GatewayUsage, service_tier_of
from gateway.streaming import merge_stream_usage

_PLAIN = CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)


def test_a_plain_usage_reports_no_tier() -> None:
    assert service_tier_of(_PLAIN) is None


def test_a_gateway_usage_keeps_the_tier_it_was_built_with() -> None:
    usage = GatewayUsage.from_completion_usage(_PLAIN, service_tier="flex")

    assert service_tier_of(usage) == "flex"
    assert service_tier_of(GatewayUsage.from_completion_usage(usage)) == "flex"


def test_a_stream_keeps_the_last_tier_reported() -> None:
    first = GatewayUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10, service_tier="priority")
    last = GatewayUsage(prompt_tokens=0, completion_tokens=2, total_tokens=0)

    assert service_tier_of(merge_stream_usage(first, last)) == "priority"
    assert service_tier_of(merge_stream_usage(last, first)) == "priority"


def test_anthropic_reports_its_tier_in_usage() -> None:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=2,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        service_tier="priority",
    )

    assert service_tier_of(_billable_messages_usage(usage)) == "priority"


def test_a_responses_response_reports_its_tier_beside_its_usage() -> None:
    usage = SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12)

    converted = _usage_to_completion_usage(usage, service_tier=_served_tier(SimpleNamespace(service_tier="flex")))  # type: ignore[arg-type]

    assert converted is not None
    assert service_tier_of(converted) == "flex"
    assert _served_tier(SimpleNamespace(service_tier=None)) is None
