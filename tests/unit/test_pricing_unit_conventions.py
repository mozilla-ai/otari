"""The three unit conventions that overload ``input_price_per_million``.

Embeddings and rerank read it as USD per million tokens, moderations, audio and
search read it as a per-request rate scaled by 1e6, and images read it as raw
USD per image. All three are the only rate readers that do not go through
``effective_rates``, so they carry their own coercion and this is where it is
pinned.
"""

from decimal import Decimal

import pytest

from gateway.core.metered_pricing import price_request
from gateway.models.pricing import ModelPricing
from gateway.services.budgets import estimate_cost
from gateway.services.pricing_service import (
    flat_request_cost,
    input_token_cost,
    per_image_cost,
    per_request_meters,
    search_unit_cost,
    search_unit_meters,
)


def _pricing(rate: object) -> ModelPricing:
    return ModelPricing(
        model_key="openai:whatever",
        input_price_per_million=rate,
        output_price_per_million=Decimal(1),
    )


@pytest.mark.parametrize("rate", [Decimal("0.15"), 0.15, "0.15"])
def test_a_rate_is_read_the_same_however_the_row_carries_it(rate: object) -> None:
    """A stored row hands back a ``Decimal``; a transient one may not."""
    pricing = _pricing(rate)

    assert input_token_cost(1000, pricing) == Decimal("0.00015")
    assert per_image_cost(2, pricing) == Decimal("0.30")
    assert flat_request_cost(pricing) == Decimal("0.00000015")


def test_the_result_is_a_decimal_whatever_the_row_carried() -> None:
    """The annotation is load-bearing: a float here reaches ``quantize_cost``."""
    assert all(
        isinstance(value, Decimal)
        for value in (
            input_token_cost(1000, _pricing(0.15)),
            per_image_cost(2, _pricing(0.15)),
            flat_request_cost(_pricing(0.15)),
        )
    )


def test_a_billable_convention_refuses_an_unusable_rate() -> None:
    """Pricing tokens or images at nothing would bill the request at nothing."""
    with pytest.raises(ValueError, match="no usable input rate"):
        input_token_cost(1000, _pricing(float("nan")))
    with pytest.raises(ValueError, match="no usable input rate"):
        per_image_cost(1, _pricing(float("-inf")))


def test_the_per_request_convention_treats_an_unusable_rate_as_unpriced() -> None:
    """Its routes are exempt from ``require_pricing`` and settle unpriced at $0."""
    assert flat_request_cost(_pricing(float("nan"))) == Decimal(0)
    assert flat_request_cost(None) == Decimal(0)


def test_a_per_request_route_and_a_per_request_model_write_the_same_charge_line() -> None:
    """Audio and moderations build their line from the cost core's, so the shape cannot drift."""
    pricing = ModelPricing(
        model_key="exa:exa",
        input_price_per_million=Decimal(5000),
        output_price_per_million=Decimal(0),
        unit="requests",
    )

    cost, meters, lines = price_request(pricing)

    assert cost == flat_request_cost(pricing)
    assert per_request_meters(cost) == (meters, lines)


def test_the_budget_estimate_for_a_per_request_model_is_its_flat_price() -> None:
    pricing = ModelPricing(
        model_key="exa:exa",
        input_price_per_million=Decimal(5000),
        output_price_per_million=Decimal(0),
        unit="requests",
    )

    estimate = estimate_cost(pricing, prompt_chars=40_000, max_output_tokens=None, default_output_tokens=4096)

    assert estimate == Decimal("0.005")


def _per_request_pricing(rate: object = Decimal(2000)) -> ModelPricing:
    return ModelPricing(
        model_key="cohere:rerank-v3.5",
        input_price_per_million=rate,
        output_price_per_million=Decimal(0),
        unit="requests",
    )


def test_search_units_bill_at_the_per_request_rate() -> None:
    """Each search unit costs what one request costs, exactly."""
    pricing = _per_request_pricing()

    assert search_unit_cost(1, pricing) == flat_request_cost(pricing) == Decimal("0.002")
    assert search_unit_cost(3, pricing) == Decimal("0.006")
    assert isinstance(search_unit_cost(3, pricing), Decimal)


def test_search_units_on_an_unpriced_or_unusable_rate_cost_nothing() -> None:
    assert search_unit_cost(2, None) == Decimal(0)
    assert search_unit_cost(2, _per_request_pricing(float("nan"))) == Decimal(0)
    assert search_unit_cost(-1, _per_request_pricing()) == Decimal(0)


def test_search_unit_meters_name_the_units_and_carry_a_unit_rate() -> None:
    """The line is a unit line (``unit_rate``), so it renders on the per-call branch."""
    meters, lines = search_unit_meters(3, Decimal("0.006")) or (None, None)

    assert meters == {"search_units": 3}
    assert lines == [{"meter": "search_units", "units": 3, "unit_rate": 0.002, "cost": 0.006}]


def test_search_unit_meters_are_absent_when_free() -> None:
    assert search_unit_meters(2, Decimal(0)) is None
    assert search_unit_meters(0, Decimal("0.002")) is None
