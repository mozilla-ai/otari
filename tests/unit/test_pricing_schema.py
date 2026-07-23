"""Unit tests for pricing API request validation."""

import pytest
from pydantic import ValidationError

from gateway.api.routes.pricing import SetPricingRequest


def test_set_pricing_request_rejects_tier_without_rate_override() -> None:
    with pytest.raises(ValidationError, match="pricing tier must override at least one price field"):
        SetPricingRequest.model_validate(
            {
                "model_key": "anthropic:claude-sonnet-4",
                "input_price_per_million": 3.0,
                "output_price_per_million": 15.0,
                "pricing_tiers": [{"min_input_tokens": 200_000}],
            }
        )


def test_set_pricing_request_allows_a_free_tier_override() -> None:
    request = SetPricingRequest.model_validate(
        {
            "model_key": "anthropic:claude-sonnet-4",
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "pricing_tiers": [{"min_input_tokens": 200_000, "cache_write_1h_price_per_million": 0.0}],
        }
    )

    assert request.pricing_tiers is not None
    assert request.pricing_tiers[0].cache_write_1h_price_per_million == 0.0
