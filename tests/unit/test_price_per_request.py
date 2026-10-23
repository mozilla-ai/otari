"""``price_per_request``: a flat USD price per call, the readable form of ``unit: requests``."""

import pytest
from pydantic import ValidationError

from gateway.core.settings.pricing import PricingConfig


@pytest.mark.parametrize(("price", "per_million"), [(0.005, 5000.0), (0.003, 3000.0), (0, 0.0), ("0.01", 10000.0)])
def test_a_price_per_request_is_a_rate_per_million_requests(price: object, per_million: float) -> None:
    pricing = PricingConfig.model_validate({"price_per_request": price})

    assert pricing.unit == "requests"
    # Exact: 0.003 does not become 3000.0000000000005.
    assert pricing.input_price_per_million == per_million
    assert pricing.output_price_per_million == 0.0


def test_a_price_per_request_validates_again_as_itself() -> None:
    pricing = PricingConfig.model_validate({"price_per_request": 0.003})

    assert PricingConfig.model_validate(pricing.model_dump()) == pricing


def test_an_explicit_requests_unit_is_accepted() -> None:
    assert PricingConfig.model_validate({"price_per_request": 0.01, "unit": "requests"}).unit == "requests"


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        ({"input_price_per_million": 1.0}, "input_price_per_million"),
        ({"output_price_per_million": 2.0}, "output_price_per_million"),
        ({"cache_read_price_per_million": 0.5}, "cache_read_price_per_million"),
        ({"pricing_tiers": [{"min_input_tokens": 1000, "input_price_per_million": 1.0}]}, "pricing_tiers"),
        ({"unit": "tokens"}, "unit"),
    ],
)
def test_a_price_per_request_is_not_combined_with_token_prices(extra: dict[str, object], named: str) -> None:
    with pytest.raises(ValidationError, match=named):
        PricingConfig.model_validate({"price_per_request": 0.005, **extra})


@pytest.mark.parametrize("price", [-1, "abc", "inf", float("nan")])
def test_a_price_per_request_that_is_not_a_finite_price_is_refused(price: object) -> None:
    with pytest.raises(ValidationError, match="price_per_request"):
        PricingConfig.model_validate({"price_per_request": price})


def test_a_malformed_input_price_beside_a_price_per_request_is_refused() -> None:
    with pytest.raises(ValidationError, match="input_price_per_million"):
        PricingConfig.model_validate({"price_per_request": 0.005, "input_price_per_million": "abc"})


def test_token_pricing_is_unchanged() -> None:
    pricing = PricingConfig(input_price_per_million=1.0, output_price_per_million=2.0)

    assert pricing.unit == "tokens"
    assert pricing.price_per_request is None
