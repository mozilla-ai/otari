"""Unit tests for ``_compute_cost``, the standalone-mode cost calculation.

These pin the cache-token pricing model, which follows the genai-prices dataset
this project already uses: the input/prompt token count is the grand total that
*includes* cache reads and writes, and each physical token is charged once.

Providers report cache tokens two ways, distinguished by
``GatewayUsage.cache_tokens_in_prompt``:

* OpenAI / Gemini: ``cache_read_tokens`` is already a subset of ``prompt_tokens``
  (``cache_tokens_in_prompt=True``). The cached portion is discounted, re-priced
  at the cache-read rate instead of the full input rate.
* Anthropic: ``cache_read_tokens`` / ``cache_write_tokens`` are separate from
  ``prompt_tokens`` (``cache_tokens_in_prompt=False``), so they are folded into
  the total and billed additively.

When a cache rate is not configured, those tokens stay in the uncached-input
bucket and are billed at the full input rate rather than dropped.
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
# No cache pricing configured: behaves as before
# ---------------------------------------------------------------------------


def test_no_cache_pricing_uses_plain_rates() -> None:
    """Without cache rates, cost is prompt + completion only (unchanged)."""
    pricing = _pricing()
    usage = GatewayUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = _compute_cost(pricing, usage)
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert cost == expected


def test_no_cache_pricing_with_openai_cache_tokens_billed_at_input_rate() -> None:
    """OpenAI cache reads with no cache rate stay inside prompt_tokens and are
    billed at the full input rate (no discount, no double count)."""
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


def test_no_cache_pricing_anthropic_cache_tokens_billed_at_input_rate() -> None:
    """Anthropic cache reads/writes with no cache rate are folded into the total
    and billed at the input rate (real input tokens, never free)."""
    pricing = _pricing()
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=100,
        cache_tokens_in_prompt=False,
    )
    cost = _compute_cost(pricing, usage)
    # total input = 1000 + 400 + 100 = 1500, all at the input rate
    expected = (1500 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
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
    # Same as test_openai_cache_read_discounted_not_double_counted; cache_write_price unused
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
        cache_tokens_in_prompt=False,
    )
    cost = _compute_cost(pricing, usage)
    expected = (
        (1000 / 1_000_000) * 30.0  # uncached prompt input
        + (500 / 1_000_000) * 60.0  # completion
        + (400 / 1_000_000) * 0.75  # cache read
        + (100 / 1_000_000) * 3.0  # cache write
    )
    assert cost == expected


def test_anthropic_cache_read_only_is_positive_and_additive() -> None:
    """Regression: a warm-cache Anthropic turn (cache read hit, no new write) is
    the common multi-turn case. cache_write is 0, but the charge must still be
    additive and strictly positive, never the negative cost the old
    ``cache_write > 0`` heuristic produced by discounting cache reads against a
    prompt that never included them."""
    pricing = _pricing(cache_read_price_per_million=0.30)
    usage = GatewayUsage(
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=10300,
        cache_read_tokens=10000,
        cache_write_tokens=0,
        cache_tokens_in_prompt=False,
    )
    cost = _compute_cost(pricing, usage)
    expected = (
        (100 / 1_000_000) * 30.0  # uncached prompt input
        + (200 / 1_000_000) * 60.0  # completion
        + (10000 / 1_000_000) * 0.30  # cache read (additive, discounted rate)
    )
    assert cost == expected
    assert cost > 0


def test_anthropic_cache_write_without_read_rate_falls_back_to_input() -> None:
    """cache_write billed at its rate; cache_read with no rate falls back to the
    input rate (folded into the uncached total, not dropped)."""
    pricing = _pricing(cache_write_price_per_million=3.0)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=100,
        cache_tokens_in_prompt=False,
    )
    cost = _compute_cost(pricing, usage)
    # total input = 1500; cache_write (100) priced at its rate, pulled out of the
    # uncached bucket; cache_read (400) has no rate so stays at the input rate.
    expected = (1400 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (100 / 1_000_000) * 3.0
    assert cost == expected


def test_anthropic_cache_read_without_write_rate_falls_back_to_input() -> None:
    """cache_read billed at its rate; cache_write with no rate falls back to the
    input rate."""
    pricing = _pricing(cache_read_price_per_million=0.75)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=400,
        cache_write_tokens=100,
        cache_tokens_in_prompt=False,
    )
    cost = _compute_cost(pricing, usage)
    # total input = 1500; cache_read (400) priced at its rate; cache_write (100)
    # has no rate so stays at the input rate.
    expected = (1100 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0 + (400 / 1_000_000) * 0.75
    assert cost == expected


def test_malformed_cache_counts_are_clamped_never_negative() -> None:
    """A broken OpenAI-shape payload where cached tokens exceed prompt_tokens must
    not drive the uncached remainder negative (which would credit the budget).
    The cache buckets are clamped to the input total; cost stays >= 0."""
    pricing = _pricing(cache_read_price_per_million=5.0, cache_write_price_per_million=15.0)
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=0,
        total_tokens=1000,
        cache_read_tokens=5000,  # nonsense: larger than prompt_tokens
        cache_write_tokens=5000,
        cache_tokens_in_prompt=True,  # OpenAI shape: cache is supposed to be a subset
    )
    cost = _compute_cost(pricing, usage)
    # cache_read clamped to 1000, cache_write clamped to 0, uncached clamped to 0.
    expected = (1000 / 1_000_000) * 5.0
    assert cost == expected
    assert cost >= 0


def test_convention_flag_drives_discount_vs_additive() -> None:
    """The same token counts price differently by convention: OpenAI discounts
    the cached subset of the prompt, Anthropic adds the cache on top."""
    pricing = _pricing(cache_read_price_per_million=5.0)
    kwargs: dict[str, Any] = dict(
        prompt_tokens=1000, completion_tokens=0, total_tokens=1400, cache_read_tokens=400, cache_write_tokens=0
    )
    openai_cost = _compute_cost(pricing, GatewayUsage(**kwargs, cache_tokens_in_prompt=True))
    anthropic_cost = _compute_cost(pricing, GatewayUsage(**kwargs, cache_tokens_in_prompt=False))
    # OpenAI: 600 input + 400 cache-read. Anthropic: 1000 input + 400 cache-read.
    assert openai_cost == (600 / 1_000_000) * 30.0 + (400 / 1_000_000) * 5.0
    assert anthropic_cost == (1000 / 1_000_000) * 30.0 + (400 / 1_000_000) * 5.0
    assert anthropic_cost > openai_cost


def test_context_pricing_tier_is_a_whole_request_cliff() -> None:
    pricing = _pricing(cache_read_price_per_million=5.0)
    pricing.pricing_tiers = [
        {
            "min_input_tokens": 200_000,
            "input_price_per_million": 60.0,
            "output_price_per_million": 120.0,
            "cache_read_price_per_million": 10.0,
        }
    ]
    usage = GatewayUsage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
        cache_read_tokens=100_000,
    )

    cost = _compute_cost(pricing, usage)

    assert cost == (150_000 / 1_000_000) * 60.0 + (1_000 / 1_000_000) * 120.0 + (100_000 / 1_000_000) * 10.0


def test_anthropic_1h_cache_write_uses_its_own_rate() -> None:
    pricing = _pricing(cache_write_price_per_million=3.75)
    pricing.cache_write_1h_price_per_million = 6.0
    usage = GatewayUsage(
        prompt_tokens=1_000,
        completion_tokens=0,
        total_tokens=1_100,
        cache_write_tokens=300,
        cache_write_1h_tokens=100,
        cache_tokens_in_prompt=False,
    )

    cost = _compute_cost(pricing, usage)

    assert cost == (1_000 / 1_000_000) * 30.0 + (200 / 1_000_000) * 3.75 + (100 / 1_000_000) * 6.0


def test_anthropic_1h_cache_write_allows_a_free_rate() -> None:
    """A configured $0 1-hour rate must not fall back to the 5-minute rate."""
    pricing = _pricing(cache_write_price_per_million=3.75)
    pricing.cache_write_1h_price_per_million = 0.0
    usage = GatewayUsage(
        prompt_tokens=1_000,
        completion_tokens=0,
        total_tokens=1_100,
        cache_write_tokens=100,
        cache_write_1h_tokens=100,
        cache_tokens_in_prompt=False,
    )

    assert _compute_cost(pricing, usage) == (1_000 / 1_000_000) * 30.0


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
