"""What one successful pass-through attempt costs, and which meter a rate prices.

The cases live in ``tests/fixtures/usage_pricing_cases.json`` so otari.ai's
settlement tests run the same table: a request costs the same on either edition.
"""

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from gateway.core.metered_pricing import billable_usage, price_billable_usage, price_request
from gateway.models.pricing import ModelPricing
from gateway.services.pricing import UsageCharge, can_price, price_usage, validated_units

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
        model_key="exa:exa", input_price_per_million=Decimal(5000), output_price_per_million=Decimal(0), unit="requests"
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
@pytest.mark.parametrize("unit", ["tokens", "requests", "images"])
def test_an_unusable_rate_prices_nothing_rather_than_serving_free(unit: str, bad: str) -> None:
    rate = ModelPricing(model_key="provider:model", input_price_per_million=Decimal(bad), unit=unit)
    tokens = billable_usage(input_tokens=1000, output_tokens=0, cache_tokens_included=True)

    assert price_usage(rate, tokens=tokens, units={"images": 1, "requests": 1}) is None
    assert not can_price(rate, meter="images", reports_tokens=True)


def test_an_unknown_rate_unit_prices_nothing() -> None:
    rate = ModelPricing(model_key="provider:model", input_price_per_million=Decimal(1), unit="seconds")

    assert price_usage(rate, tokens=None, units={"requests": 1}) is None
    assert not can_price(rate, meter="requests", reports_tokens=False)


def test_a_token_rate_read_per_unit_warns_once_per_model_and_meter(caplog: pytest.LogCaptureFixture) -> None:
    rate = ModelPricing(model_key="legacy:warned-once", input_price_per_million=Decimal("0.04"), unit="tokens")

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            price_usage(rate, tokens=None, units={"images": 1})
        price_usage(rate, tokens=None, units={"requests": 1})

    warnings = [record.getMessage() for record in caplog.records if "legacy:warned-once" in record.getMessage()]
    assert len(warnings) == 2
    assert "set its unit to images and multiply its input price by 1,000,000" in warnings[0]
    assert "set its unit to requests, keeping its price" in warnings[1]
