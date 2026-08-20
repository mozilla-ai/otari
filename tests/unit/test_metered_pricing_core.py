"""The cost-math core: exact arithmetic, both cached-token conventions, one rounding.

``tests/unit/test_compute_cost.py`` pins the request path's behavior through
``_compute_cost``. This file tests the core underneath it directly, where the
cached-token convention is an argument rather than a property of a usage
carrier, and where the guards against a corrupt rate live.
"""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.core.metered_pricing import (
    COST_QUANTUM,
    billable_usage,
    calculate_token_cost,
    effective_rates,
    estimate_metered_cost,
    price_billable_usage,
    quantize_cost,
    to_decimal,
)


def _pricing(**overrides: Any) -> SimpleNamespace:
    """A rate row, read structurally the way the core reads any pricing object."""
    rates: dict[str, Any] = {
        "input_price_per_million": Decimal("30"),
        "output_price_per_million": Decimal("60"),
        "cache_read_price_per_million": None,
        "cache_write_price_per_million": None,
        "cache_write_1h_price_per_million": None,
        "pricing_tiers": [],
    }
    rates.update(overrides)
    return SimpleNamespace(**rates)


# ---------------------------------------------------------------------------
# Exactness: the reason the core is Decimal
# ---------------------------------------------------------------------------


def test_a_rate_float_cannot_represent_prices_exactly() -> None:
    """700k tokens at $0.10/M is seven cents, which binary floating point misses."""
    cost = calculate_token_cost(
        _pricing(input_price_per_million=Decimal("0.1")),
        input_tokens=700_000,
        output_tokens=0,
        cache_tokens_included=True,
    )

    assert cost == Decimal("0.070000")
    # What the float arithmetic this replaced produced for the same inputs.
    assert (700_000 / 1_000_000) * 0.1 != 0.07


def test_a_stored_float_rate_is_read_through_its_shortest_decimal() -> None:
    """A rate that still arrives as a float (a JSON tier, a transient row) is exact."""
    assert to_decimal(0.075) == Decimal("0.075")
    assert to_decimal("0.075") == Decimal("0.075")
    assert to_decimal(Decimal("0.075")) == Decimal("0.075")


# ---------------------------------------------------------------------------
# The rounding this change defines
# ---------------------------------------------------------------------------


def test_a_settled_cost_is_rounded_half_up_to_the_micro_dollar() -> None:
    assert COST_QUANTUM == Decimal("0.000001")
    assert quantize_cost(Decimal("0.0000005")) == Decimal("0.000001")
    assert quantize_cost(Decimal("0.0000004")) == Decimal("0.000000")
    # Half-up, not Python's default half-even, which would round this one down.
    assert quantize_cost(Decimal("0.0000015")) == Decimal("0.000002")


def test_a_sub_micro_dollar_request_settles_at_the_rounded_amount() -> None:
    """One token at $0.50/M is half a micro-dollar, and rounds up to one."""
    cost = calculate_token_cost(
        _pricing(input_price_per_million=Decimal("0.5")),
        input_tokens=1,
        output_tokens=0,
        cache_tokens_included=True,
    )

    assert cost == Decimal("0.000001")


# ---------------------------------------------------------------------------
# Both cached-token conventions, stated by the caller
# ---------------------------------------------------------------------------


def test_the_convention_has_no_default() -> None:
    """Which convention a caller speaks cannot be inferred, so it must be passed."""
    with pytest.raises(TypeError):
        billable_usage(input_tokens=1_000, output_tokens=0)  # type: ignore[call-arg]


def test_the_two_conventions_price_the_same_counts_differently() -> None:
    pricing = _pricing(cache_read_price_per_million=Decimal("5"))
    counts: dict[str, Any] = {
        "input_tokens": 1_000,
        "output_tokens": 0,
        "cache_read_tokens": 400,
    }

    inclusive = calculate_token_cost(pricing, cache_tokens_included=True, **counts)
    additive = calculate_token_cost(pricing, cache_tokens_included=False, **counts)

    # Inclusive: 600 fresh + 400 cached. Additive: 1000 fresh + 400 cached on top.
    assert inclusive == Decimal("0.020000")
    assert additive == Decimal("0.032000")


def test_the_additive_convention_grows_the_billable_input_total() -> None:
    usage = billable_usage(
        input_tokens=1_000,
        output_tokens=0,
        cache_read_tokens=400,
        cache_write_tokens=100,
        cache_tokens_included=False,
    )

    assert usage.total_input_tokens == 1_500


def test_a_1h_write_is_a_subset_of_the_write_total() -> None:
    usage = billable_usage(
        input_tokens=1_000,
        output_tokens=0,
        cache_write_tokens=100,
        cache_write_1h_tokens=400,  # nonsense: larger than the write total
        cache_tokens_included=False,
    )

    assert usage.cache_write_1h_tokens == 100
    assert usage.cache_write_base_tokens == 0


def test_an_inclusive_payload_that_contradicts_itself_loses_its_cache_discount() -> None:
    usage = billable_usage(
        input_tokens=1_000,
        output_tokens=0,
        cache_read_tokens=5_000,
        cache_write_tokens=5_000,
        cache_tokens_included=True,
    )

    assert usage.total_input_tokens == 1_000
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (0, 0)


# ---------------------------------------------------------------------------
# Rates: tiers, fallbacks, and what a corrupt one does
# ---------------------------------------------------------------------------


def test_a_tier_reprices_the_whole_request_and_only_the_fields_it_names() -> None:
    pricing = _pricing(
        cache_read_price_per_million=Decimal("5"),
        pricing_tiers=[{"min_input_tokens": 200_000, "input_price_per_million": 60}],
    )

    rates = effective_rates(pricing, 250_000)

    assert rates.input_price_per_million == Decimal("60")
    assert rates.output_price_per_million == Decimal("60")
    assert rates.cache_read_price_per_million == Decimal("5")


def test_the_highest_reached_tier_wins() -> None:
    pricing = _pricing(
        pricing_tiers=[
            {"min_input_tokens": 200_000, "input_price_per_million": 60},
            {"min_input_tokens": 1_000_000, "input_price_per_million": 90},
        ]
    )

    assert effective_rates(pricing, 250_000).input_price_per_million == Decimal("60")
    assert effective_rates(pricing, 1_500_000).input_price_per_million == Decimal("90")


@pytest.mark.parametrize(
    "tier",
    [
        {"min_input_tokens": "not-a-number", "input_price_per_million": 1},
        {"min_input_tokens": -1, "input_price_per_million": 1},
        "not-a-tier",
    ],
)
def test_a_malformed_tier_is_ignored_rather_than_repricing_everything(tier: Any) -> None:
    """A tier whose bound cannot be read is not treated as a bound of zero."""
    pricing = _pricing(pricing_tiers=[tier])

    assert effective_rates(pricing, 1_000).input_price_per_million == Decimal("30")


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf"), "not-a-rate", None])
def test_an_unusable_tier_rate_falls_back_to_the_base_rate(rate: Any) -> None:
    pricing = _pricing(pricing_tiers=[{"min_input_tokens": 0, "input_price_per_million": rate}])

    assert effective_rates(pricing, 1_000).input_price_per_million == Decimal("30")


@pytest.mark.parametrize("rate", [None, -1, float("nan"), "not-a-rate"])
def test_an_unusable_base_rate_refuses_to_price(rate: Any) -> None:
    """Pricing at zero would bill the request at nothing and say so nowhere."""
    with pytest.raises(ValueError, match="no usable input or output rate"):
        effective_rates(_pricing(input_price_per_million=rate), 1_000)


def test_an_unpriced_1h_write_bills_at_the_ordinary_cache_write_rate() -> None:
    cost = calculate_token_cost(
        _pricing(cache_write_price_per_million=Decimal("3.75")),
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=100,
        cache_write_1h_tokens=100,
        cache_tokens_included=False,
    )

    assert cost == Decimal("0.000375")


def test_sub_amounts_of_one_row_can_be_summed_before_the_row_is_rounded() -> None:
    """Rounding per line would round the row once per line, not once.

    A batch is one usage row priced a request at a time, so a line is not a
    settled total. Ten thousand identical lines of 200 input and 2 output
    tokens at gpt-4o-mini rates come to $0.312; rounding each line to the
    micro-dollar first loses $0.002 of it, every time, in the same direction.
    """
    pricing = _pricing(input_price_per_million=Decimal("0.15"), output_price_per_million=Decimal("0.6"))
    counts: dict[str, Any] = {"input_tokens": 200, "output_tokens": 2, "cache_tokens_included": True}

    exact_line = calculate_token_cost(pricing, quantize=False, **counts)
    settled_line = calculate_token_cost(pricing, **counts)

    assert exact_line == Decimal("0.0000312")
    assert settled_line == Decimal("0.000031")
    assert quantize_cost(exact_line * 10_000) == Decimal("0.312000")
    assert settled_line * 10_000 == Decimal("0.310000")


# ---------------------------------------------------------------------------
# Meters and charge lines
# ---------------------------------------------------------------------------


def test_charge_lines_explain_the_settled_total() -> None:
    pricing = _pricing(cache_read_price_per_million=Decimal("5"))
    cost, meters, lines = price_billable_usage(
        pricing,
        billable_usage(
            input_tokens=1_000,
            output_tokens=500,
            cache_read_tokens=400,
            cache_tokens_included=True,
        ),
    )

    assert meters == {
        "total_input_tokens": 1_000,
        "fresh_input_tokens": 600,
        "cache_read_tokens": 400,
        "cache_write_tokens": 0,
        "cache_write_1h_tokens": 0,
        "completion_tokens": 500,
    }
    assert [line["meter"] for line in lines] == ["input", "output", "cache_read"]
    # The lines are JSON, so they carry floats; the settled amount is the Decimal.
    assert all(isinstance(line["cost"], float) for line in lines)
    assert cost == quantize_cost(Decimal(str(sum(line["cost"] for line in lines))))


def test_a_meter_with_no_units_gets_no_charge_line() -> None:
    _, _, lines = price_billable_usage(
        _pricing(),
        billable_usage(input_tokens=1_000, output_tokens=0, cache_tokens_included=True),
    )

    assert [line["meter"] for line in lines] == ["input"]


# ---------------------------------------------------------------------------
# The reserve-time upper bound
# ---------------------------------------------------------------------------


def test_an_estimate_reserves_the_dearest_rate_a_prompt_token_could_attract() -> None:
    pricing = _pricing(
        cache_write_price_per_million=Decimal("37.5"),
        cache_write_1h_price_per_million=Decimal("60"),
    )

    plain = estimate_metered_cost(pricing, estimated_input_tokens=1_000, estimated_output_tokens=0)
    five_minute = estimate_metered_cost(
        pricing, estimated_input_tokens=1_000, estimated_output_tokens=0, cache_write_ttl="5m"
    )
    one_hour = estimate_metered_cost(
        pricing, estimated_input_tokens=1_000, estimated_output_tokens=0, cache_write_ttl="1h"
    )

    assert plain == Decimal("0.030000")
    assert five_minute == Decimal("0.037500")
    assert one_hour == Decimal("0.060000")


def test_an_estimate_never_prices_below_the_input_rate() -> None:
    """A cache-write rate cheaper than input must not shrink the reservation."""
    pricing = _pricing(cache_write_price_per_million=Decimal("1"))

    estimate = estimate_metered_cost(
        pricing, estimated_input_tokens=1_000, estimated_output_tokens=0, cache_write_ttl="5m"
    )

    assert estimate == Decimal("0.030000")
