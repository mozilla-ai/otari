"""What one successful pass-through attempt costs, and which meter a rate prices.

The cases live in ``tests/fixtures/usage_pricing_cases.json`` so otari.ai's
settlement tests run the same table: a request costs the same on either edition.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from any_llm.types.completion import CompletionUsage

from gateway.core.metered_pricing import billable_usage, price_billable_usage, price_request
from gateway.models.pricing import ModelPricing
from gateway.services.pricing import (
    MEASURED_UNITS,
    Quantities,
    Slot,
    UsageCharge,
    can_price,
    estimate_settles,
    has_completion_meter,
    price_completion,
    price_usage,
    validated_units,
)
from gateway.services.pricing._usage_pricing import METERS
from gateway.types.price_unit import PRICE_UNITS, PriceUnit

_CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "usage_pricing_cases.json").read_text()
)["cases"]


def _rate(spec: dict[str, Any] | None) -> ModelPricing | None:
    if spec is None:
        return None
    return ModelPricing(
        model_key="provider:model",
        input_price_per_million=Decimal(spec["input"]),
        output_price_per_million=Decimal(spec.get("output", "0")),
        unit=spec["unit"],
    )


def _price(case: dict[str, Any]) -> UsageCharge | None:
    tokens = case["tokens"]
    return price_usage(
        _rate(case["rate"]),
        tokens=None
        if tokens is None or not (tokens["input"] or tokens["output"])
        else billable_usage(input_tokens=tokens["input"], output_tokens=tokens["output"], cache_tokens_included=True),
        units=case["units"],
    )


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_an_attempt_costs_what_the_table_says(case: dict[str, Any]) -> None:
    charge = _price(case)
    expected = case["expected"]

    if expected is None:
        assert charge is None
        return
    assert charge is not None
    assert charge.cost == Decimal(expected["cost"])
    if "meters" in expected:
        assert charge.meters == expected["meters"]
        assert charge.lines == expected["lines"]


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_every_charge_is_settled_and_explained(case: dict[str, Any]) -> None:
    """Never negative, rounded to the micro-dollar, and meters and lines present exactly when it costs something."""
    charge = _price(case)
    if charge is None:
        return
    assert charge.cost >= 0
    assert charge.cost == charge.cost.quantize(Decimal("0.000001"))
    assert bool(charge.meters) == bool(charge.lines) == (charge.cost > 0)


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_the_order_of_the_counts_does_not_change_the_price(case: dict[str, Any]) -> None:
    reversed_case = {**case, "units": dict(reversed(list(case["units"].items())))}
    assert _price(reversed_case) == _price(case)


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_the_gate_admits_exactly_what_settlement_can_price(case: dict[str, Any]) -> None:
    """A rate ``require_pricing`` admits is one that settles priced, and one it refuses would settle unpriced."""
    tokens = case["tokens"]
    reports_tokens = tokens is not None and bool(tokens["input"] or tokens["output"])
    meters = [meter for meter in ("images", "requests", "search_units") if meter in case["units"]]
    meter = meters[0] if meters else None

    admitted = can_price(_rate(case["rate"]), meter=meter, reports_tokens=reports_tokens)

    assert admitted == (_price(case) is not None)


def test_every_unit_has_a_meter_or_is_measured() -> None:
    """The table and the measured units together name every unit, and nothing else."""
    assert set(METERS) == {"token", "request", "image"}
    assert set(METERS) | MEASURED_UNITS == set(PRICE_UNITS)


@pytest.mark.parametrize("reports_tokens", [False, True])
@pytest.mark.parametrize("units", [{}, {"requests": 1}, {"search_units": 2}, {"images": 1}])
@pytest.mark.parametrize("unit", PRICE_UNITS)
def test_the_gate_and_settlement_agree_on_every_unit_and_report(
    unit: str, units: dict[str, int], *, reports_tokens: bool
) -> None:
    """Both ask the one table, so no unit can pass the gate and settle unpriced, or the reverse."""
    rate = ModelPricing(
        model_key="provider:model", input_price_per_million=Decimal(1), output_price_per_million=Decimal(2), unit=unit
    )
    tokens = billable_usage(input_tokens=100, output_tokens=50, cache_tokens_included=True) if reports_tokens else None

    admitted = can_price(rate, meter=next(iter(units), None), reports_tokens=reports_tokens)

    assert admitted == (price_usage(rate, tokens=tokens, units=units) is not None)


@pytest.mark.parametrize("count", [True, 1.5, -1, 1_000_001, "2"])
def test_a_count_that_is_not_a_whole_number_in_range_is_refused(count: object) -> None:
    with pytest.raises(ValueError, match="whole number"):
        validated_units({"images": count})
    with pytest.raises(ValueError, match="whole number"):
        price_usage(None, tokens=None, units={"requests": count})


def test_an_unknown_meter_is_dropped_rather_than_refused() -> None:
    assert validated_units({"seconds": -5, "images": 2}) == {"images": 2}


def test_a_requests_rate_settles_a_completion_as_the_cost_core_does() -> None:
    rate = ModelPricing(
        model_key="exa:exa", input_price_per_million=Decimal(5000), output_price_per_million=Decimal(0), unit="request"
    )
    tokens = billable_usage(input_tokens=2000, output_tokens=10, cache_tokens_included=True)

    cost, meters, lines = price_request(rate)

    assert price_usage(rate, tokens=tokens, units={}) == UsageCharge(cost, meters, lines)


def test_a_token_rate_with_no_counts_settles_as_the_cost_core_does() -> None:
    rate = ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=Decimal("2.5"),
        output_price_per_million=Decimal(10),
        cache_read_price_per_million=Decimal("1.25"),
    )
    tokens = billable_usage(input_tokens=1000, output_tokens=200, cache_read_tokens=400, cache_tokens_included=True)

    cost, meters, lines = price_billable_usage(rate, tokens)

    assert price_usage(rate, tokens=tokens, units={}) == UsageCharge(cost, meters, lines)


@pytest.mark.parametrize("bad", ["NaN", "-1", "Infinity"])
@pytest.mark.parametrize("unit", ["token", "request", "image"])
def test_an_unusable_rate_prices_nothing_rather_than_serving_free(unit: str, bad: str) -> None:
    rate = ModelPricing(model_key="provider:model", input_price_per_million=Decimal(bad), unit=unit)
    tokens = billable_usage(input_tokens=1000, output_tokens=0, cache_tokens_included=True)

    assert price_usage(rate, tokens=tokens, units={"images": 1, "requests": 1}) is None
    assert not can_price(rate, meter="images", reports_tokens=True)


# The five measured units, which no count or token report meets, and one no rate can name.
@pytest.mark.parametrize("unit", ["character", "megapixel", "second", "minute", "web_search", "seconds"])
def test_a_rate_in_a_unit_with_no_meter_prices_nothing(unit: str) -> None:
    rate = ModelPricing(model_key="provider:model", input_price_per_million=Decimal(1), unit=unit)

    assert price_usage(rate, tokens=None, units={"requests": 1}) is None
    assert (
        price_usage(
            rate, tokens=billable_usage(input_tokens=100, output_tokens=50, cache_tokens_included=True), units={}
        )
        is None
    )
    assert not can_price(rate, meter="requests", reports_tokens=False)
    assert not can_price(rate, meter=None, reports_tokens=True)


def _slot_rate(rates: dict[str, Any], unit: str = "token") -> ModelPricing:
    return ModelPricing(
        model_key="provider:model",
        input_price_per_million=Decimal(1),
        output_price_per_million=Decimal(0),
        unit=unit,
        rates=rates,
    )


_IMAGE_OUTPUT: Slot = ("image", "output")
_AUDIO_INPUT: Slot = ("audio", "input")
_TEXT_INPUT: Slot = ("text", "input")
_TOKENS = billable_usage(input_tokens=1000, output_tokens=0, cache_tokens_included=True)


@pytest.mark.parametrize(
    ("rates", "slot", "quantities", "cost", "meters", "line"),
    [
        (
            {"image": {"output": {"price": 0.04, "unit": "image"}}},
            _IMAGE_OUTPUT,
            {"image": 2, "megapixel": Decimal("2.097152")},
            "0.080000",
            {"images": 2},
            {"meter": "images", "units": 2, "unit": "image", "unit_rate": 0.04, "cost": 0.08},
        ),
        (
            {"image": {"output": {"price": 0.05, "unit": "megapixel"}}},
            _IMAGE_OUTPUT,
            {"image": 2, "megapixel": Decimal("2.097152")},
            "0.104858",
            {"megapixels": 2.097152},
            {"meter": "megapixels", "units": 2.097152, "unit": "megapixel", "unit_rate": 0.05, "cost": 0.1048576},
        ),
        (
            {"audio": {"input": {"price": 0.006, "unit": "minute"}}},
            _AUDIO_INPUT,
            {"second": Decimal(90), "minute": Decimal("1.5")},
            "0.009000",
            {"minutes": 1.5},
            {"meter": "minutes", "units": 1.5, "unit": "minute", "unit_rate": 0.006, "cost": 0.009},
        ),
        (
            {"audio": {"input": {"price": 0.0001, "unit": "second"}}},
            _AUDIO_INPUT,
            {"second": Decimal(90), "minute": Decimal("1.5")},
            "0.009000",
            {"seconds": 90},
            {"meter": "seconds", "units": 90, "unit": "second", "unit_rate": 0.0001, "cost": 0.009},
        ),
        (
            {"text": {"input": {"price": 15, "unit": "character"}}},
            _TEXT_INPUT,
            {"character": 2000},
            "0.030000",
            {"characters": 2000},
            {"meter": "characters", "units": 2000, "unit": "character", "rate_per_million": 15.0, "cost": 0.03},
        ),
    ],
    ids=["per image", "per megapixel", "per minute", "per second", "per character"],
)
def test_a_slot_in_a_measured_unit_prices_the_quantity_in_that_unit(
    rates: dict[str, Any],
    slot: Slot,
    quantities: Quantities,
    cost: str,
    meters: dict[str, Any],
    line: dict[str, Any],
) -> None:
    charge = price_usage(_slot_rate(rates), tokens=_TOKENS, units={"images": 2}, quantities=quantities, slot=slot)

    assert charge is not None
    assert charge.cost == Decimal(cost)
    assert charge.meters == meters
    assert charge.lines == [line]
    assert can_price(_slot_rate(rates), meter="images", reports_tokens=True, slot=slot, measures=frozenset(quantities))


def test_a_slot_in_a_measured_unit_prices_nothing_the_attempt_did_not_measure() -> None:
    """A per-megapixel rate never bills the image count, and the gate refuses an endpoint that cannot measure it."""
    rate = _slot_rate({"image": {"output": {"price": 0.05, "unit": "megapixel"}}})

    assert price_usage(rate, tokens=_TOKENS, units={"images": 2}, quantities={"image": 2}, slot=_IMAGE_OUTPUT) is None
    assert not can_price(rate, meter="images", reports_tokens=True, slot=_IMAGE_OUTPUT, measures=frozenset({"image"}))


def test_a_slot_in_tokens_or_requests_prices_as_the_table_says() -> None:
    """The slot chooses only among measured units; tokens and requests keep their meters."""
    per_token = _slot_rate({"text": {"input": {"price": 1}, "image": {"output": {"price": 1}}}})
    per_request = _slot_rate({"text": {"input": {"price": 2000, "unit": "request"}}}, unit="request")
    quantities: Quantities = {"image": 2, "character": 10}

    tokens = price_usage(per_token, tokens=_TOKENS, units={"images": 2}, quantities=quantities, slot=_IMAGE_OUTPUT)
    requests = price_usage(per_request, tokens=None, units={"requests": 1}, quantities=quantities, slot=_TEXT_INPUT)

    assert tokens is not None and tokens.cost == Decimal("0.001")
    assert requests is not None and requests.meters == {"requests": 1}
    assert price_usage(per_token, tokens=None, units={"requests": 1}, quantities=quantities, slot=_TEXT_INPUT) is None


def test_a_rate_with_no_price_in_the_slot_prices_as_the_table_says() -> None:
    """An image rate on the text slot has no slot there, so it is read by its unit: per image."""
    rate = _slot_rate({"image": {"output": {"price": 0.04, "unit": "image"}}}, unit="image")

    charge = price_usage(rate, tokens=None, units={"images": 3}, quantities={"character": 5}, slot=_TEXT_INPUT)

    assert charge is not None
    assert charge.meters == {"images": 3}


def test_a_free_token_rate_settles_free_with_no_tokens_reported() -> None:
    free = ModelPricing(
        model_key="openai:free", input_price_per_million=Decimal(0), output_price_per_million=Decimal(0)
    )
    priced = ModelPricing(
        model_key="openai:priced", input_price_per_million=Decimal(0), output_price_per_million=Decimal(1)
    )

    assert price_usage(free, tokens=None, units={"requests": 1}) == UsageCharge(Decimal(0), {}, [])
    assert can_price(free, meter="requests", reports_tokens=False)
    assert price_usage(priced, tokens=None, units={"requests": 1}) is None
    assert not can_price(priced, meter="requests", reports_tokens=False)


@pytest.mark.parametrize(
    "extra",
    [
        {"cache_read_price_per_million": Decimal("0.1")},
        {"cache_write_price_per_million": Decimal("0.1")},
        {"pricing_tiers": [{"min_input_tokens": 200_000, "input_price_per_million": 2.0}]},
    ],
    ids=["cache read", "cache write", "tier"],
)
def test_a_token_rate_with_any_price_above_zero_is_not_free(extra: dict[str, Any]) -> None:
    """Only a rate priced at zero everywhere settles free with nothing reported."""
    rate = ModelPricing(
        model_key="openai:m", input_price_per_million=Decimal(0), output_price_per_million=Decimal(0), **extra
    )

    assert price_usage(rate, tokens=None, units={"requests": 1}) is None
    assert not can_price(rate, meter="requests", reports_tokens=False)


def test_a_token_rate_free_in_its_text_columns_but_not_in_its_block_is_not_free() -> None:
    """A speech model whose text is free and whose audio output is not settles nothing at zero."""
    rate = ModelPricing(
        model_key="gemini:tts",
        input_price_per_million=Decimal(0),
        output_price_per_million=Decimal(0),
        rates={
            "text": {"input": {"price": 0, "unit": "token"}, "output": {"price": 0, "unit": "token"}},
            "audio": {"output": {"price": 10, "unit": "token"}},
        },
    )
    speech: frozenset[PriceUnit] = frozenset({"character"})

    assert price_usage(rate, tokens=None, units={"requests": 1}, quantities={"character": 5}, slot=_TEXT_INPUT) is None
    assert not can_price(rate, meter="requests", reports_tokens=False, slot=_TEXT_INPUT, measures=speech)
    free_block = ModelPricing(
        model_key="local:tts",
        input_price_per_million=Decimal(0),
        output_price_per_million=Decimal(0),
        rates={"text": {"input": {"price": 0, "unit": "token"}}, "audio": {"output": {"price": 0, "unit": "token"}}},
    )
    assert can_price(free_block, meter="requests", reports_tokens=False, slot=_TEXT_INPUT, measures=speech)


_REPORTED = CompletionUsage(prompt_tokens=12, completion_tokens=34, total_tokens=46)


def test_a_completion_at_a_token_rate_settles_its_tokens_even_when_it_failed() -> None:
    rate = _rate({"input": "1000", "output": "2000", "unit": "token"})
    assert rate is not None

    for succeeded in (True, False):
        priced = price_completion(rate, _REPORTED, succeeded=succeeded)
        assert priced is not None
        assert priced[0] == Decimal("0.080000")
    assert price_completion(rate, None, succeeded=True) is None


def test_a_completion_at_a_per_request_rate_settles_one_request_only_when_it_succeeded() -> None:
    rate = _rate({"input": "5000", "unit": "request"})
    assert rate is not None

    priced = price_completion(rate, _REPORTED, succeeded=True)
    assert priced is not None
    assert (priced[0], priced[1]) == (Decimal("0.005"), {"requests": 1})
    assert price_completion(rate, _REPORTED, succeeded=False) is None
    assert price_completion(rate, None, succeeded=True) is not None


@pytest.mark.parametrize("unit", ["character", "megapixel", "second", "minute", "web_search"])
def test_a_completion_at_a_rate_with_no_completion_meter_prices_nothing(unit: str) -> None:
    rate = _rate({"input": "1000", "output": "2000", "unit": unit})

    assert rate is not None
    assert price_completion(rate, _REPORTED, succeeded=True) is None


@pytest.mark.parametrize(
    ("unit", "metered"),
    [
        ("token", True),
        ("image", True),
        ("request", True),
        # Stored ahead of their meters.
        ("character", False),
        ("megapixel", False),
        ("second", False),
        ("minute", False),
        ("web_search", False),
    ],
)
def test_a_completion_has_a_meter_only_for_tokens_images_and_requests(unit: str, *, metered: bool) -> None:
    rate = _rate({"input": "1", "unit": unit})

    assert has_completion_meter(rate) is metered
    assert estimate_settles(rate) is metered


def test_no_rate_has_no_completion_meter_and_its_zero_estimate_settles() -> None:
    assert not has_completion_meter(None)
    assert estimate_settles(None)
