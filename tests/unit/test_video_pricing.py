"""What a video costs: the per-second price a model's rates give video output."""

from decimal import Decimal
from typing import Any

import pytest

from gateway.core.metered_pricing import UnitRate
from gateway.models.pricing import ModelPricing
from gateway.services.pricing import (
    VIDEO_SECONDS,
    ModelSearchRate,
    UsageCharge,
    estimate_video_cost,
    model_search_rate,
    price_video,
    search_rate,
    video_rate,
)

_PER_SECOND: dict[str, Any] = {"video": {"output": {"price": 0.1, "unit": "second"}}}


def _rate(rates: dict[str, Any] | None) -> ModelPricing:
    return ModelPricing(
        model_key="gemini:veo-3.0-generate-001",
        input_price_per_million=Decimal(0),
        output_price_per_million=Decimal(0),
        unit="second",
        rates=rates,
    )


def test_a_block_that_prices_video_per_second_gives_the_rate() -> None:
    assert video_rate(_rate(_PER_SECOND)) == UnitRate("second", Decimal("0.1"))


def test_a_video_rate_is_read_beside_a_text_rate() -> None:
    rates = {"text": {"input": {"price": 3}, "output": {"price": 15}}, **_PER_SECOND}

    assert video_rate(_rate(rates)) == UnitRate("second", Decimal("0.1"))


@pytest.mark.parametrize(
    "rates",
    [
        None,
        {},
        {"text": {"input": {"price": 3}, "output": {"price": 15}}},
        {"video": {"input": {"price": 0.1, "unit": "second"}}},
        {"video": {"output": {"price": 6, "unit": "minute"}}},
        {"video": {"output": {"price": 0.1}}},
        {"video": {"output": {"unit": "second"}}},
    ],
    ids=["no block", "empty", "text only", "input side", "per minute", "per token", "no price"],
)
def test_a_block_with_no_per_second_video_output_price_gives_no_rate(rates: dict[str, Any] | None) -> None:
    assert video_rate(_rate(rates)) is None


def test_no_rate_gives_no_video_rate_and_a_zero_estimate() -> None:
    assert video_rate(None) is None
    assert estimate_video_cost(None, 8) == Decimal(0)
    assert price_video(None, 8) is None


def test_the_estimate_is_the_length_at_the_per_second_price() -> None:
    assert estimate_video_cost(_rate(_PER_SECOND), 8) == Decimal("0.8")


def test_a_negative_length_estimates_nothing() -> None:
    assert estimate_video_cost(_rate(_PER_SECOND), -8) == Decimal(0)


def test_a_block_with_a_web_search_price_gives_the_models_search_rate() -> None:
    rates = {**_PER_SECOND, "web_search": {"price": 20}}

    assert search_rate(_rate(rates)) == UnitRate("web_search", Decimal(20))
    assert model_search_rate(_rate(rates), "deployment") == ModelSearchRate(Decimal("0.02"), "deployment")
    assert search_rate(_rate(_PER_SECOND)) is None
    assert model_search_rate(None, "defaults") is None


def test_a_completed_video_costs_its_seconds_with_a_line_that_explains_them() -> None:
    charge = price_video(_rate(_PER_SECOND), 6)

    assert charge is not None
    assert charge.cost == Decimal("0.6")
    assert charge.meters == {VIDEO_SECONDS: 6}
    assert charge.lines == [{"meter": VIDEO_SECONDS, "units": 6, "unit": "second", "unit_rate": 0.1, "cost": 0.6}]


def test_a_slot_priced_at_zero_charges_nothing_and_carries_no_meter() -> None:
    charge = price_video(_rate({"video": {"output": {"price": 0, "unit": "second"}}}), 6)

    assert charge == UsageCharge(Decimal(0), {}, [])


def test_the_cost_is_rounded_once_to_the_micro_dollar() -> None:
    charge = price_video(_rate({"video": {"output": {"price": 0.0333333, "unit": "second"}}}), 7)

    assert charge is not None
    assert charge.cost == Decimal("0.233333")


def test_a_video_of_no_length_costs_nothing_and_explains_nothing() -> None:
    charge = price_video(_rate(_PER_SECOND), 0)

    assert charge is not None
    assert (charge.cost, charge.meters, charge.lines) == (Decimal(0), {}, [])


def test_a_negative_length_cannot_be_priced() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        price_video(_rate(_PER_SECOND), -1)
