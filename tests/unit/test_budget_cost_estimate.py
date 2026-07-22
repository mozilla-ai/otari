"""Unit tests for threshold-aware budget reservation estimates."""

from typing import Any

import pytest

from gateway.models.entities import ModelPricing
from gateway.services.budget_service import estimate_cost


def _pricing(**overrides: Any) -> ModelPricing:
    defaults = {
        "model_key": "anthropic:claude-sonnet-4",
        "input_price_per_million": 3.0,
        "output_price_per_million": 15.0,
        "cache_write_price_per_million": 3.75,
        "cache_write_1h_price_per_million": 6.0,
        "pricing_tiers": [
            {
                "min_input_tokens": 200_000,
                "input_price_per_million": 6.0,
                "output_price_per_million": 22.5,
                "cache_write_price_per_million": 7.5,
                "cache_write_1h_price_per_million": 12.0,
            }
        ],
    }
    defaults.update(overrides)
    return ModelPricing(**defaults)


def test_estimate_cost_uses_base_rates_below_context_threshold() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=199_999 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
    )

    assert estimate == pytest.approx((199_999 * 3.0 + 100 * 15.0) / 1_000_000)


def test_estimate_cost_uses_context_tier_for_all_meters_at_threshold() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=200_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
    )

    assert estimate == pytest.approx((200_000 * 6.0 + 100 * 22.5) / 1_000_000)


def test_estimate_cost_reserves_explicit_cache_write_as_additional_input() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=100_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
        cache_write_ttl="5m",
    )

    # The potential write doubles billable input, so the long-context tier
    # applies to ordinary input, output, and cache creation.
    assert estimate == pytest.approx((100_000 * 6.0 + 100 * 22.5 + 100_000 * 7.5) / 1_000_000)


def test_estimate_cost_uses_requested_one_hour_cache_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == pytest.approx((1_000 * 3.0 + 1_000 * 6.0) / 1_000_000)


def test_estimate_cost_falls_back_to_input_when_cache_write_is_unpriced() -> None:
    estimate = estimate_cost(
        _pricing(cache_write_price_per_million=None, cache_write_1h_price_per_million=None),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == pytest.approx(1_000 * 3.0 / 1_000_000)


def test_estimate_cost_preserves_a_free_cache_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(cache_write_1h_price_per_million=0.0),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == pytest.approx(1_000 * 3.0 / 1_000_000)
