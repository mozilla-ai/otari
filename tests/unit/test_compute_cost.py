"""Unit tests for ``_compute_cost`` — the standalone-mode cost calculation.

These pin the cache-token pricing conventions:

* OpenAI / Gemini: ``cache_read_tokens`` is a subset of ``prompt_tokens``,
  so the cached portion is discounted (re-priced at the cache-read rate)
  rather than double-counted at the full input rate.
* Anthropic: ``cache_read_tokens`` and ``cache_write_tokens`` are separate
  from ``prompt_tokens``, so they are billed as additive charges.
"""

from typing import Any

import pytest
from any_llm.types.completion import CompletionUsage
from pydantic import ValidationError

from gateway.api.routes._pipeline import _compute_cost
from gateway.core.config import PricingConfig
from gateway.core.usage import GatewayUsage
from gateway.models.entities import ModelPricing


def _pricing(**overrides: float | None) -> ModelPricing:
    defaults: dict[str, Any] = {
        "model_key": "openai:gpt-4",
        "input_price_per_million": 30.0,
        "output_price_per_million": 60.0,
        "cache_read_price_per_million": None,
        "cache_write_price_per_million": None,
    }
    defaults.update(overrides)
    return ModelPricing(**defaults)


# ---------------------------------------------------------------------------
# No cache pricing configured — behaves exactly as before
# ---------------------------------------------------------------------------


def test_no_cache_pricing_uses_plain_rates() -> None:
    """Without cache rates, cost is prompt + completion only (unchanged)."""
    pricing = _pricing()
    usage = GatewayUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = _compute_cost(pricing, usage)
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert cost == expected


def test_no_cache_pricing_with_cache_tokens_ignores_them() -> None:
    """Cache tokens present but no cache rates configured: no cache charge."""
    pricing = _pricing()
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=0,
    )
    cost = _compute_cost(pricing, usage)
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert cost == expected


# ---------------------------------------------------------------------------
# OpenAI / Gemini: cache_read is a subset of prompt_tokens (discounted)
# ---------------------------------------------------------------------------


def test_openai_cache_read_discounted_not_double_counted() -> None:
    """Cached tokens are re-priced at the cache-read rate, not the input rate."""
    pricing = _pricing(cache_read_price_per_million=5.0)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=0,
    )
    cost = _compute_cost(pricing, usage)
    # 600 non-cached at full input rate + 400 cached at discount rate
    expected = (600 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (400 / 1_000_000) * 5.0
    assert cost == expected


def test_openai_cache_read_zero_tokens_no_discount() -> None:
    """When cache_read_tokens is 0, no discount is applied even if rate is set."""
    pricing = _pricing(cache_read_price_per_million=5.0)
    usage = GatewayUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = _compute_cost(pricing, usage)
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert cost == expected


def test_openai_cache_read_with_cache_write_price_ignored() -> None:
    """cache_write_price is irrelevant when cache_write_tokens is 0."""
    pricing = _pricing(cache_read_price_per_million=5.0, cache_write_price_per_million=15.0)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=0,
    )
    cost = _compute_cost(pricing, usage)
    # Same as test_openai_cache_read_discounted_not_double_counted — cache_write_price unused
    expected = (600 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (400 / 1_000_000) * 5.0
    assert cost == expected


# ---------------------------------------------------------------------------
# Anthropic: cache_read / cache_write are separate from prompt_tokens (additive)
# ---------------------------------------------------------------------------


def test_anthropic_cache_read_and_write_additive() -> None:
    """Anthropic cache tokens are billed as extra charges on top of prompt."""
    pricing = _pricing(cache_read_price_per_million=0.75, cache_write_price_per_million=3.0)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=100,
    )
    cost = _compute_cost(pricing, usage)
    expected = (
        (1000 / 1_000_000) * 30.0  # prompt (non-cached)
        + (500 / 1_000_000) * 60.0  # completion
        + (400 / 1_000_000) * 0.75  # cache read
        + (100 / 1_000_000) * 3.0  # cache write
    )
    assert cost == expected


def test_anthropic_cache_write_without_read_rate() -> None:
    """cache_write billed even when cache_read_price is None."""
    pricing = _pricing(cache_write_price_per_million=3.0)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=100,
    )
    cost = _compute_cost(pricing, usage)
    # cache_read_price is None so no cache_read charge; cache_write is billed
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (100 / 1_000_000) * 3.0
    assert cost == expected


def test_anthropic_cache_read_without_write_rate() -> None:
    """cache_read billed additively even when cache_write_price is None."""
    pricing = _pricing(cache_read_price_per_million=0.75)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=100,
    )
    cost = _compute_cost(pricing, usage)
    # cache_write_price is None so no cache_write charge; cache_read is additive
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (400 / 1_000_000) * 0.75
    assert cost == expected


# ---------------------------------------------------------------------------
# Plain CompletionUsage (OpenAI-style fallback via prompt_tokens_details)
# ---------------------------------------------------------------------------


def test_plain_completion_usage_uses_fallback_cache_read() -> None:
    """A plain CompletionUsage with prompt_tokens_details.cached_tokens is
    treated as OpenAI-style (subset of prompt, discounted)."""
    from any_llm.types.completion import PromptTokensDetails

    pricing = _pricing(cache_read_price_per_million=5.0)
    usage = CompletionUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=400),
    )
    cost = _compute_cost(pricing, usage)
    expected = (600 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (400 / 1_000_000) * 5.0
    assert cost == expected


# ---------------------------------------------------------------------------
# PricingConfig schema: cache fields are optional and default to None
# ---------------------------------------------------------------------------


def test_pricing_config_cache_fields_default_to_none() -> None:
    config = PricingConfig(input_price_per_million=30.0, output_price_per_million=60.0)
    assert config.cache_read_price_per_million is None
    assert config.cache_write_price_per_million is None


def test_pricing_config_accepts_cache_rates() -> None:
    config = PricingConfig(
        input_price_per_million=30.0,
        output_price_per_million=60.0,
        cache_read_price_per_million=5.0,
        cache_write_price_per_million=15.0,
    )
    assert config.cache_read_price_per_million == 5.0
    assert config.cache_write_price_per_million == 15.0


def test_pricing_config_rejects_negative_cache_rate() -> None:
    with pytest.raises(ValidationError):
        PricingConfig(
            input_price_per_million=30.0,
            output_price_per_million=60.0,
            cache_read_price_per_million=-1.0,
        )
