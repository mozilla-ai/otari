"""Unit tests for pricing API request validation."""

from typing import get_args

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


def test_both_rate_surfaces_name_the_rungs_the_same_way() -> None:
    """The Models page and the offered-models panel answer the same reader.

    Two spellings of one rung read as two different facts, and both fields
    reach the browser as free strings unless something holds them together.
    """
    from gateway.models.pricing import PriceSource
    from gateway.schemas.providers import (
    OrgProviderKeyModelPublic,
)
    from gateway.services.merged_catalog_service import ViewerPrice

    rungs = set(get_args(PriceSource))
    assert rungs == {"organization", "deployment", "defaults"}

    offered = OrgProviderKeyModelPublic.model_fields["price_source"].annotation
    assert set(get_args(get_args(offered)[0])) == rungs
    assert set(get_args(get_args(ViewerPrice.__annotations__["source"])[0])) == rungs
