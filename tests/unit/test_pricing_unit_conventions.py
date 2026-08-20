"""The three unit conventions that overload ``input_price_per_million``.

Embeddings and rerank read it as USD per million tokens, moderations, audio and
search read it as a per-request rate scaled by 1e6, and images read it as raw
USD per image. All three are the only rate readers that do not go through
``effective_rates``, so they carry their own coercion and this is where it is
pinned.
"""

from decimal import Decimal

import pytest

from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import flat_request_cost, input_token_cost, per_image_cost


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
