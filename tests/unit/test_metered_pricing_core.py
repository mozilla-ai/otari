"""The cost-math core: exact arithmetic, both cached-token conventions, one rounding.

``tests/unit/test_compute_cost.py`` pins the request path's total, calling
``calculate_metered_cost`` with a usage object. This file tests the core underneath it directly, where the
cached-token convention is an argument rather than a property of a usage
carrier, and where the guards against a corrupt rate live.
"""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.core.metered_pricing import (
    COST_QUANTUM,
    UnitCharge,
    UnitRate,
    billable_usage,
    calculate_token_cost,
    charge_unit,
    charge_unit_price,
    effective_rates,
    estimate_metered_cost,
    estimate_unit_cost,
    price_billable_usage,
    price_request,
    priced_per_request,
    quantize_cost,
    slot_price,
    to_decimal,
    unit_cost,
    unit_rate,
    web_search_rate,
)
from gateway.types.models_dev import (
    MODELS_DEV_UNITS,
    ModelsDevCachePrices,
    ModelsDevModalityCost,
    ModelsDevPricing,
    ModelsDevUnitPrice,
)
from gateway.types.price_unit import PRICE_UNITS


def _pricing(**overrides: Any) -> SimpleNamespace:
    """A rate row, read structurally the way the core reads any pricing object."""
    rates: dict[str, Any] = {
        "input_price_per_million": Decimal("30"),
        "output_price_per_million": Decimal("60"),
        "cache_read_price_per_million": None,
        "cache_write_price_per_million": None,
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
# 1h cache writes
# ---------------------------------------------------------------------------

# Haiku 5.5's catalog rates. Anthropic bills a 5m cache write at 1.25x input and
# a 1h write at 2x; the catalog carries only the first.
_HAIKU_5_5: dict[str, Any] = {
    "model_key": "anthropic:claude-haiku-5-5",
    "input_price_per_million": Decimal("0.10"),
    "output_price_per_million": Decimal("0.50"),
    "cache_write_price_per_million": Decimal("0.125"),
}


def _one_hour_writes(pricing: SimpleNamespace, tokens: int) -> Decimal:
    return calculate_token_cost(
        pricing,
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=tokens,
        cache_write_1h_tokens=tokens,
        cache_tokens_included=False,
    )


def test_a_claude_1h_write_bills_at_twice_the_input_rate() -> None:
    cost, _, lines = price_billable_usage(
        _pricing(**_HAIKU_5_5),
        billable_usage(
            input_tokens=0,
            output_tokens=0,
            cache_write_tokens=4_000,
            cache_write_1h_tokens=3_000,
            cache_tokens_included=False,
        ),
    )

    assert cost == Decimal("0.000725")
    assert [(line["meter"], line["units"], line["rate_per_million"]) for line in lines] == [
        ("cache_write", 1_000, 0.125),
        ("cache_write_1h", 3_000, 0.2),
    ]


@pytest.mark.parametrize(
    ("model_key", "cost"),
    [
        ("anthropic:claude-haiku-5-5", "0.000800"),
        ("bedrock:us.anthropic.claude-haiku-5-5", "0.000800"),
        ("vertexaianthropic:claude-haiku-5-5@default", "0.000800"),
        ("openai:gpt-5", "0.000500"),
        # An Anthropic-compatible endpoint serving another vendor's model.
        ("minimax:MiniMax-M2", "0.000500"),
        # An instance named for Claude serving another vendor's model is not Claude.
        ("claude-proxy:MiniMax-M2", "0.000500"),
        # No provider: the key is the bare model id, whose Bedrock ":<n>" suffix is a version.
        ("claude-haiku-5-5", "0.000800"),
        ("us.anthropic.claude-haiku-5-5-v1:0", "0.000800"),
        ("anthropic.claude-haiku-5-5-v1:0", "0.000800"),
        ("anthropic.claude-3-sonnet-20240229-v1:0:200k", "0.000800"),
        ("meta.llama3-70b-instruct-v1:0", "0.000500"),
        # A non-ASCII digit is not a version, so the instance name stays ignored.
        ("claude-proxy:\u00b2", "0.000500"),
        # Digits followed by more than a colon start a model id, not a version.
        ("claude-proxy:4o-mini", "0.000500"),
        # A provider and a Bedrock id that carries its own version suffix.
        ("bedrock:us.anthropic.claude-haiku-5-5-v1:0", "0.000800"),
    ],
)
def test_a_1h_write_bills_at_twice_input_only_for_a_claude_model(model_key: str, cost: str) -> None:
    assert _one_hour_writes(_pricing(**{**_HAIKU_5_5, "model_key": model_key}), 4_000) == Decimal(cost)


def test_a_1h_write_on_a_rate_that_names_no_model_bills_at_the_cache_write_rate() -> None:
    pricing = _pricing(cache_write_price_per_million=Decimal("3.75"))

    assert _one_hour_writes(pricing, 100) == Decimal("0.000375")


def test_a_claude_1h_write_takes_twice_the_input_rate_of_the_tier_it_reaches() -> None:
    tier = {
        "min_input_tokens": 100_000,
        "input_price_per_million": Decimal("0.5"),
        "cache_write_price_per_million": Decimal("0.625"),
    }

    assert _one_hour_writes(_pricing(**_HAIKU_5_5, pricing_tiers=[tier]), 200_000) == Decimal("0.200000")


def test_a_claude_1h_write_with_no_cache_write_rate_bills_at_the_input_rate() -> None:
    pricing = _pricing(**{**_HAIKU_5_5, "cache_write_price_per_million": None})

    assert _one_hour_writes(pricing, 4_000) == Decimal("0.000400")


def test_a_claude_1h_write_follows_a_free_cache_write_rate() -> None:
    pricing = _pricing(**{**_HAIKU_5_5, "cache_write_price_per_million": Decimal(0)})

    assert _one_hour_writes(pricing, 4_000) == Decimal(0)


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
    pricing = _pricing(cache_write_price_per_million=Decimal("37.5"))

    plain = estimate_metered_cost(pricing, estimated_input_tokens=1_000, estimated_output_tokens=0)
    five_minute = estimate_metered_cost(
        pricing, estimated_input_tokens=1_000, estimated_output_tokens=0, cache_write_ttl="5m"
    )
    one_hour = estimate_metered_cost(
        pricing, estimated_input_tokens=1_000, estimated_output_tokens=0, cache_write_ttl="1h"
    )

    assert plain == Decimal("0.030000")
    assert five_minute == Decimal("0.037500")
    assert one_hour == Decimal("0.037500")


def test_a_claude_estimate_reserves_a_1h_write_at_the_rate_it_settles_at() -> None:
    estimate = estimate_metered_cost(
        _pricing(**_HAIKU_5_5), estimated_input_tokens=4_000, estimated_output_tokens=0, cache_write_ttl="1h"
    )

    assert estimate == _one_hour_writes(_pricing(**_HAIKU_5_5), 4_000) == Decimal("0.000800")


def test_a_1h_estimate_still_covers_5m_writes_priced_above_twice_the_input_rate() -> None:
    """A request asking for any 1h write is estimated as 1h, though most of its writes may be 5m.

    The bundled snapshot's daoxe row prices a Claude cache write at 5x input.
    """
    pricing = _pricing(
        model_key="daoxe:claude-haiku-4-5-20251001",
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("5"),
        cache_write_price_per_million=Decimal("5"),
    )
    settled = calculate_token_cost(
        pricing,
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=10_000,
        cache_write_1h_tokens=100,
        cache_tokens_included=False,
    )

    estimate = estimate_metered_cost(
        pricing, estimated_input_tokens=10_000, estimated_output_tokens=0, cache_write_ttl="1h"
    )

    assert settled == Decimal("0.049700")
    assert estimate == Decimal("0.050000")


def test_an_estimate_never_prices_below_the_input_rate() -> None:
    """A cache-write rate cheaper than input must not shrink the reservation."""
    pricing = _pricing(cache_write_price_per_million=Decimal("1"))

    estimate = estimate_metered_cost(
        pricing, estimated_input_tokens=1_000, estimated_output_tokens=0, cache_write_ttl="5m"
    )

    assert estimate == Decimal("0.030000")


def test_a_rate_priced_per_request_settles_one_request_whatever_the_tokens() -> None:
    pricing = _pricing(input_price_per_million=Decimal("5000"), output_price_per_million=Decimal("0"), unit="request")

    cost, meters, lines = price_request(pricing)

    assert priced_per_request(pricing)
    assert cost == Decimal("0.005")
    assert meters == {"requests": 1}
    assert lines == [{"meter": "request", "units": 1, "unit_rate": 0.005, "cost": 0.005}]


def test_a_per_request_charge_line_agrees_with_the_settled_cost() -> None:
    """A rate below a micro-dollar per request rounds once, and the line shows the rounded amount."""
    cost, _, lines = price_request(_pricing(input_price_per_million=Decimal("1.5"), unit="request"))

    assert cost == Decimal("0.000002")
    assert lines == [{"meter": "request", "units": 1, "unit_rate": 0.000002, "cost": 0.000002}]

    cost, _, lines = price_request(_pricing(input_price_per_million=Decimal("0.4"), unit="request"))

    assert (cost, lines) == (Decimal("0"), [])


def test_a_free_per_request_rate_writes_no_charge_line() -> None:
    cost, meters, lines = price_request(_pricing(input_price_per_million=Decimal("0"), unit="request"))

    assert (cost, meters, lines) == (Decimal("0"), {"requests": 1}, [])


def test_a_per_request_rate_refuses_an_unusable_rate() -> None:
    with pytest.raises(ValueError, match="no usable per-request rate"):
        price_request(_pricing(input_price_per_million=None, unit="request"))


def test_a_per_request_rate_is_estimated_at_one_request() -> None:
    """The reservation holds what the request settles at, not a token estimate."""
    pricing = _pricing(input_price_per_million=Decimal("5000"), output_price_per_million=Decimal("0"), unit="request")

    estimate = estimate_metered_cost(pricing, estimated_input_tokens=1_000_000, estimated_output_tokens=4096)

    assert estimate == Decimal("0.005")


@pytest.mark.parametrize("unit", ["tokens", "images", None])
def test_a_rate_not_priced_per_request_keeps_token_pricing(unit: str | None) -> None:
    """``tokens`` (and a row with no unit) is priced exactly as before; ``images`` is not rerouted here."""
    pricing = _pricing(unit=unit) if unit is not None else _pricing()

    assert not priced_per_request(pricing)
    assert estimate_metered_cost(
        pricing, estimated_input_tokens=1_000_000, estimated_output_tokens=1_000_000
    ) == Decimal("90")


# ---------------------------------------------------------------------------
# A price in any unit, read from a cost block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "quantity", "price", "cost"),
    [
        ("token", 1_000, "30", "0.03"),
        ("character", 2_000_000, "15", "30"),
        ("request", 2, "0.005", "0.01"),
        ("image", 3, "0.04", "0.12"),
        ("megapixel", Decimal("2.5"), "0.01", "0.025"),
        ("second", Decimal("12.5"), "0.006", "0.075"),
        ("minute", Decimal("1.5"), "0.4", "0.6"),
        ("web_search", 7, "10", "0.07"),
    ],
)
def test_a_quantity_of_any_unit_costs_its_price_per_what_the_unit_is_published_per(
    unit: Any, quantity: int | Decimal, price: str, cost: str
) -> None:
    assert unit_cost(quantity, unit, Decimal(price)) == Decimal(cost)


def test_every_price_unit_has_a_meter() -> None:
    """A rate in any unit the list knows prices a quantity; the list is pinned to models.dev's elsewhere."""
    for unit in PRICE_UNITS:
        assert unit_cost(1, unit, Decimal(1)) > 0
    assert set(MODELS_DEV_UNITS) <= set(PRICE_UNITS)


@pytest.mark.parametrize("quantity", [-1, Decimal("-0.5"), Decimal("NaN"), Decimal("Infinity")])
def test_a_quantity_that_is_not_a_finite_non_negative_number_is_refused(quantity: Decimal | int) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        unit_cost(quantity, "second", Decimal(1))


def test_a_per_second_price_is_read_as_published_and_not_through_the_column_scale() -> None:
    """$0.006 per second for 12.5 seconds is $0.075: no per-million factor is applied."""
    rate = unit_rate(ModelsDevUnitPrice(price=0.006, unit="second"))

    assert rate == UnitRate("second", Decimal("0.006"))
    assert charge_unit(rate, "audio_input", Decimal("12.5")) == UnitCharge(
        Decimal("0.075"),
        [{"meter": "audio_input", "units": 12.5, "unit": "second", "unit_rate": 0.006, "cost": 0.075}],
    )


def test_a_per_million_unit_explains_itself_per_million_and_a_per_one_unit_per_one() -> None:
    characters = charge_unit(UnitRate("character", Decimal("15")), "tts_input", 2_000_000)
    searches = charge_unit(UnitRate("web_search", Decimal("10")), "web_search", 3)

    assert characters.lines == [
        {"meter": "tts_input", "units": 2_000_000, "unit": "character", "rate_per_million": 15.0, "cost": 30.0}
    ]
    assert searches.cost == Decimal("0.03")
    assert searches.lines == [
        {"meter": "web_search", "units": 3, "unit": "web_search", "unit_rate": 0.01, "cost": 0.03}
    ]


def test_a_charge_is_exact_and_rounded_once_by_the_caller() -> None:
    """Three images at a third of a micro-dollar each: exact sub-micro amounts sum, then round once."""
    rate = UnitRate("image", Decimal("0.0000004"))

    charges = [charge_unit(rate, "image", 1) for _ in range(3)]
    total = quantize_cost(sum((charge.cost for charge in charges), Decimal(0)))

    assert all(charge.cost == Decimal("0.0000004") for charge in charges)
    assert total == Decimal("0.000001")


def test_a_zero_quantity_costs_nothing_and_explains_nothing() -> None:
    assert charge_unit(UnitRate("image", Decimal("0.04")), "image", 0) == UnitCharge(Decimal(0), [])


def test_a_free_price_still_explains_a_quantity() -> None:
    charge = charge_unit(UnitRate("image", Decimal(0)), "image", 2)

    assert charge.cost == 0
    assert charge.lines == [{"meter": "image", "units": 2, "unit": "image", "unit_rate": 0.0, "cost": 0.0}]


def _slot(**fields: Any) -> ModelsDevUnitPrice:
    return ModelsDevUnitPrice(
        price=0.5,
        unit="second",
        reasoning=0.9,
        cache=ModelsDevCachePrices(read=0.05, write=0.6, write_ttls=(("5m", 0.6), ("1h", 1.0)), storage=0.02),
        **fields,
    )


@pytest.mark.parametrize(
    ("kind", "ttl", "price"),
    [
        ("price", None, "0.5"),
        ("reasoning", None, "0.9"),
        ("cache_read", None, "0.05"),
        ("cache_write", None, "0.6"),
        ("cache_write", "5m", "0.6"),
        ("cache_write", "1h", "1.0"),
        ("cache_write", "7d", "0.6"),
        ("cache_storage", None, "0.02"),
    ],
)
def test_each_price_of_a_slot_has_a_meter_in_the_slots_unit(kind: Any, ttl: str | None, price: str) -> None:
    """A TTL the slot does not price bills as a plain write, as a 1h write does on a rate with no 1h price."""
    charge = charge_unit_price(_slot(), 10, meter=kind, kind=kind, cache_write_ttl=ttl)

    assert charge is not None
    assert charge.cost == Decimal(price) * 10
    assert charge.lines[0]["unit"] == "second"


def test_a_ttl_price_that_is_no_price_bills_as_a_plain_write() -> None:
    """The reader keeps a number as written; one that is no price reads as absent, so the TTL falls back."""
    slot = ModelsDevUnitPrice(
        price=0.5, unit="second", cache=ModelsDevCachePrices(write=0.6, write_ttls=(("5m", 0.6), ("1h", -1)))
    )

    assert unit_rate(slot, "cache_write", cache_write_ttl="1h") == UnitRate("second", Decimal("0.6"))
    assert estimate_unit_cost(slot, 10, cache_write_ttl="1h") == Decimal("6")


@pytest.mark.parametrize("kind", ["reasoning", "cache_read", "cache_write", "cache_storage"])
def test_a_price_the_slot_does_not_give_prices_nothing(kind: Any) -> None:
    bare = ModelsDevUnitPrice(price=0.5, unit="second")

    assert charge_unit_price(bare, 10, meter=kind, kind=kind) is None
    assert charge_unit_price(bare, 10, meter="audio_input") is not None


def test_a_slot_with_no_unit_is_per_token() -> None:
    charge = charge_unit_price(ModelsDevUnitPrice(price=3), 1_000_000, meter="input")

    assert charge is not None
    assert (charge.cost, charge.lines[0]["unit"], charge.lines[0]["rate_per_million"]) == (Decimal(3), "token", 3.0)


def test_a_slot_in_a_unit_no_meter_knows_prices_nothing_rather_than_tokens() -> None:
    """A unit models.dev adds is named in the catalog's issues; here it bills nothing, never tokens."""
    parsec = ModelsDevUnitPrice(price=1, unit="parsec", cache=ModelsDevCachePrices(read=0.1))

    assert unit_rate(parsec) is None
    assert unit_rate(parsec, "cache_read") is None
    assert charge_unit_price(parsec, 1_000_000, meter="input") is None
    assert estimate_unit_cost(parsec, 1_000_000) == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_a_slot_price_that_is_not_a_believable_number_prices_nothing(value: Any) -> None:
    """A price the reader kept as a number but that is no price; a string never gets this far, the reader names it."""
    assert unit_rate(ModelsDevUnitPrice(price=value, unit="image")) is None


def test_a_block_is_read_by_modality_and_direction() -> None:
    image_out = ModelsDevUnitPrice(price=0.04, unit="image")
    audio_in = ModelsDevUnitPrice(price=0.006, unit="second")
    pricing = ModelsDevPricing(
        modalities={
            "text": ModelsDevModalityCost(input=ModelsDevUnitPrice(price=3), output=ModelsDevUnitPrice(price=15)),
            "image": ModelsDevModalityCost(output=image_out),
            "audio": ModelsDevModalityCost(input=audio_in),
        },
        web_search=ModelsDevUnitPrice(price=10),
    )

    assert slot_price(pricing, "image", "output") is image_out
    assert slot_price(pricing, "image", "input") is None
    assert slot_price(pricing, "audio", "input") is audio_in
    assert slot_price(pricing, "video", "output") is None
    assert web_search_rate(pricing) == UnitRate("web_search", Decimal("10"))
    assert web_search_rate(ModelsDevPricing(modalities={})) is None


def test_an_estimate_reserves_the_dearest_price_a_unit_could_attract() -> None:
    slot = _slot()

    assert estimate_unit_cost(slot, 10) == Decimal("9")
    assert estimate_unit_cost(slot, 10, cache_write_ttl="1h") == Decimal("10")
    assert estimate_unit_cost(ModelsDevUnitPrice(price=0.006, unit="second"), Decimal("12.5")) == Decimal("0.075")
    assert estimate_unit_cost(ModelsDevUnitPrice(price=None, unit="second"), 10) == 0


def test_an_estimate_in_a_rates_unit_is_never_zero_for_a_priced_rate() -> None:
    """A per-second rate, stored per million seconds, is estimated at the seconds the caller expects."""
    pricing = _pricing(
        input_price_per_million=Decimal("6000"),
        output_price_per_million=Decimal("0"),
        cache_read_price_per_million=Decimal("9000"),
        unit="second",
    )

    estimate = estimate_metered_cost(
        pricing, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=Decimal("12.5")
    )

    assert estimate == Decimal("0.1125")
    assert estimate_metered_cost(pricing, estimated_input_tokens=0, estimated_output_tokens=0) == 0


def test_an_estimate_in_a_rates_unit_covers_an_output_priced_above_input() -> None:
    """One quantity cannot say which side of the slot it is on, so the dearest rate of the row bounds it."""
    pricing = _pricing(
        input_price_per_million=Decimal("6000"), output_price_per_million=Decimal("24000"), unit="second"
    )

    assert estimate_metered_cost(
        pricing, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=Decimal("2.5")
    ) == Decimal("0.06")


def test_an_estimate_in_a_rates_unit_takes_the_tier_the_estimated_input_reaches() -> None:
    """A unit row's tiers are keyed on input tokens, as a token row's are, so the estimated input picks one."""
    pricing = _pricing(
        input_price_per_million=Decimal("6000"),
        output_price_per_million=Decimal("0"),
        pricing_tiers=[{"min_input_tokens": 100, "input_price_per_million": 12000}],
        unit="second",
    )

    below = estimate_metered_cost(pricing, estimated_input_tokens=99, estimated_output_tokens=0, estimated_units=1)
    reached = estimate_metered_cost(pricing, estimated_input_tokens=100, estimated_output_tokens=0, estimated_units=1)

    assert (below, reached) == (Decimal("0.006"), Decimal("0.012"))


def test_an_estimate_in_a_rates_unit_covers_a_claude_1h_write_priced_above_every_other_rate() -> None:
    """A Claude row bills a 1h write at twice its input rate; the unit estimate holds that too."""
    pricing = _pricing(
        model_key="anthropic:claude-x",
        input_price_per_million=Decimal("6000"),
        output_price_per_million=Decimal("0"),
        cache_write_price_per_million=Decimal("7500"),
        unit="second",
    )

    five_minute = estimate_metered_cost(
        pricing, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=1, cache_write_ttl="5m"
    )
    one_hour = estimate_metered_cost(
        pricing, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=1, cache_write_ttl="1h"
    )

    assert (five_minute, one_hour) == (Decimal("0.0075"), Decimal("0.012"))


def test_a_retired_unit_spelling_is_read_as_the_current_one() -> None:
    per_request = _pricing(
        input_price_per_million=Decimal("5000"), output_price_per_million=Decimal("0"), unit="requests"
    )

    assert priced_per_request(per_request)
    assert estimate_metered_cost(
        per_request, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=5
    ) == Decimal("0.005")


def test_a_negative_estimated_quantity_holds_nothing_rather_than_a_credit() -> None:
    pricing = _pricing(input_price_per_million=Decimal("6000"), output_price_per_million=Decimal("0"), unit="second")

    assert estimate_metered_cost(pricing, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=-5) == 0
    assert estimate_unit_cost(ModelsDevUnitPrice(price=0.006, unit="second"), -5) == 0
    with pytest.raises(ValueError, match="finite"):
        estimate_unit_cost(ModelsDevUnitPrice(price=0.006, unit="second"), Decimal("NaN"))


def test_an_estimate_in_tokens_or_requests_ignores_a_unit_quantity() -> None:
    per_request = _pricing(
        input_price_per_million=Decimal("5000"), output_price_per_million=Decimal("0"), unit="request"
    )

    assert estimate_metered_cost(
        _pricing(), estimated_input_tokens=1_000_000, estimated_output_tokens=0, estimated_units=5
    ) == Decimal("30")
    assert estimate_metered_cost(
        per_request, estimated_input_tokens=0, estimated_output_tokens=0, estimated_units=5
    ) == Decimal("0.005")
    # A row an older writer stored still spells the unit ``tokens``; it is a token rate.
    assert estimate_metered_cost(
        _pricing(unit="tokens"), estimated_input_tokens=1_000_000, estimated_output_tokens=0, estimated_units=5
    ) == Decimal("30")
