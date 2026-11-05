"""Every way a request is priced reads its rate at the rules the request meets."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from any_llm.types.completion import CompletionUsage

from gateway.core.metered_pricing import billable_usage
from gateway.models.pricing import ModelPricing
from gateway.services.budgets import count_estimate, estimate_cost, report_estimate
from gateway.services.pricing import (
    PriceFacts,
    estimate_video_cost,
    model_search_rate,
    price_completion,
    price_usage,
    price_video,
)
from gateway.types.price_rates import columns_from_rates

_BEFORE = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
_AFTER = datetime(2027, 1, 1, 1, 0, tzinfo=UTC)
_NEW_YEAR = {"date": {"gte": "2027-01-01"}}


def _row(rates: dict[str, Any]) -> ModelPricing:
    """A row as every writer stores it: the block, and the columns read out of it."""
    columns = columns_from_rates(rates)
    return ModelPricing(
        model_key="google:gemini-x",
        input_price_per_million=columns.input_price_per_million,
        output_price_per_million=columns.output_price_per_million,
        cache_read_price_per_million=columns.cache_read_price_per_million,
        cache_write_price_per_million=columns.cache_write_price_per_million,
        pricing_tiers=[dict(tier) for tier in columns.pricing_tiers],
        unit=columns.unit,
        rates=rates,
    )


_LONG_CONTEXT = _row(
    {
        "text": {
            "input": {"price": 1, "overrides": [{"when": {"context": {"gt": 1000}}, "price": 2}]},
            "output": {"price": 10, "overrides": [{"when": {"context": {"gt": 1000}}, "price": 20}]},
        },
        "web_search": {"price": 35},
        "multipliers": [{"when": _NEW_YEAR, "factor": 2}],
    }
)


def _completion(input_tokens: int, *, at: datetime = _BEFORE) -> Decimal:
    usage = CompletionUsage(prompt_tokens=input_tokens, completion_tokens=100, total_tokens=input_tokens + 100)
    priced = price_completion(_LONG_CONTEXT, usage, succeeded=True, facts=PriceFacts(at=at))
    assert priced is not None
    return priced[0]


def test_a_completion_reaches_a_context_override_only_past_its_bound() -> None:
    assert _completion(1000) == Decimal("0.002000")
    assert _completion(1001) == Decimal("0.004002")


def test_a_completion_pays_the_price_of_the_time_it_started() -> None:
    assert _completion(1000, at=_AFTER) == 2 * _completion(1000, at=_BEFORE)


def test_tokens_on_a_pass_through_endpoint_reach_their_context_override() -> None:
    charge = price_usage(
        _LONG_CONTEXT,
        tokens=billable_usage(input_tokens=2000, output_tokens=0, cache_tokens_included=True),
        units={},
        facts=PriceFacts(at=_BEFORE),
    )

    assert charge is not None
    assert charge.cost == Decimal("0.004000")


def test_the_estimate_reads_the_override_its_estimated_prompt_reaches() -> None:
    short = estimate_cost(_LONG_CONTEXT, prompt_chars=4000, max_output_tokens=0, default_output_tokens=0)
    long = estimate_cost(_LONG_CONTEXT, prompt_chars=4004, max_output_tokens=0, default_output_tokens=0)

    assert (short, long) == (Decimal("0.001000"), Decimal("0.002002"))


def test_the_estimate_holds_the_price_of_the_time_it_is_made() -> None:
    held = estimate_cost(
        _LONG_CONTEXT, prompt_chars=4000, max_output_tokens=0, default_output_tokens=0, facts=PriceFacts(at=_AFTER)
    )

    assert held == Decimal("0.002000")


def test_a_count_and_a_report_hold_their_price_at_the_rules_met() -> None:
    per_request = _row(
        {
            "text": {"input": {"price": 0.002, "unit": "request"}, "output": {"price": 0, "unit": "request"}},
            "multipliers": [{"when": _NEW_YEAR, "factor": 3}],
        }
    )

    counted = count_estimate(per_request, meter="requests", count=1, facts=PriceFacts(at=_AFTER))
    reported = report_estimate(
        per_request, slot=None, quantities={}, units={"requests": 1}, facts=PriceFacts(at=_AFTER)
    )

    assert counted == reported == Decimal("0.006000")


def test_a_video_is_priced_and_held_at_the_rules_met() -> None:
    video = _row(
        {
            "video": {"output": {"price": 0.1, "unit": "second"}},
            "multipliers": [{"when": _NEW_YEAR, "factor": {"video": 1.5}}],
        }
    )

    charge = price_video(video, 10, facts=PriceFacts(at=_AFTER))

    assert charge is not None
    assert charge.cost == Decimal("1.500000")
    assert estimate_video_cost(video, 10, facts=PriceFacts(at=_AFTER)) == Decimal("1.500000")
    assert estimate_video_cost(video, 10, facts=PriceFacts(at=_BEFORE)) == Decimal("1.000000")


def test_a_search_the_provider_ran_is_priced_at_the_rules_met() -> None:
    before = model_search_rate(_LONG_CONTEXT, "defaults", facts=PriceFacts(at=_BEFORE))
    after = model_search_rate(_LONG_CONTEXT, "defaults", facts=PriceFacts(at=_AFTER))

    assert before is not None and after is not None
    assert (before.unit_rate, after.unit_rate) == (Decimal("0.035"), Decimal("0.070"))
