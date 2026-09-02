"""Unit tests for threshold-aware budget reservation estimates.

Compared exactly: the estimate is a ``Decimal`` quantized to the micro-dollar
(mozilla-ai/otari#691), so the expected amount is spelled as the arithmetic that
produces it rather than approximated. ``pytest.approx`` here would pass on an
estimate that had gone back to binary floating point.
"""

from decimal import Decimal
from typing import Any

from gateway.models.entities import ModelPricing
from gateway.services.budget_service import estimate_cost, estimate_tokens


def _micro_dollars(*terms: Decimal | int) -> Decimal:
    """The USD total of per-million-token charges, exactly.

    Each term is ``tokens * rate``, which the cost core divides by a million
    once. Written this way so the expected value stays the arithmetic under
    test rather than a constant somebody would have to re-derive.
    """
    return sum((Decimal(term) for term in terms), Decimal(0)) / 1_000_000


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

    assert estimate == _micro_dollars(199_999 * 3, 100 * 15)


def test_estimate_cost_uses_context_tier_for_all_meters_at_threshold() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=200_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
    )

    assert estimate == _micro_dollars(200_000 * 6, Decimal(100) * Decimal("22.5"))


def test_estimate_cost_reserves_explicit_cache_write_at_the_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=1_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
        cache_write_ttl="5m",
    )

    # Any prompt token could become a 5m cache write, so the input side is
    # reserved at the (dearer) cache-write rate rather than stacked on top of it.
    assert estimate == _micro_dollars(Decimal(1_000) * Decimal("3.75"), 100 * 15)


def test_estimate_cost_reserves_cache_write_using_the_context_tier() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=200_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
        cache_write_ttl="5m",
    )

    # The estimated input alone crosses the tier, so the tier's write and output
    # rates apply; the tier is selected from the real billable total, not double.
    assert estimate == _micro_dollars(Decimal(200_000) * Decimal("7.5"), Decimal(100) * Decimal("22.5"))


def test_estimate_cost_uses_requested_one_hour_cache_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == _micro_dollars(1_000 * 6)


def test_estimate_cost_falls_back_to_input_when_cache_write_is_unpriced() -> None:
    estimate = estimate_cost(
        _pricing(cache_write_price_per_million=None, cache_write_1h_price_per_million=None),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == _micro_dollars(1_000 * 3)


def test_estimate_cost_preserves_a_free_cache_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(cache_write_1h_price_per_million=0.0),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == _micro_dollars(1_000 * 3)


def test_estimate_tokens_sums_the_two_figures_the_cost_is_priced_from() -> None:
    """Prompt chars at four to the token, rounded up, plus the declared output bound."""
    assert estimate_tokens(prompt_chars=401, max_output_tokens=1_000, default_output_tokens=4_096) == 101 + 1_000


def test_estimate_tokens_falls_back_to_the_default_output_bound() -> None:
    """An unbounded request reserves the deployment's cap, as the cost estimate does."""
    assert estimate_tokens(prompt_chars=0, max_output_tokens=None, default_output_tokens=4_096) == 4_096


def test_estimate_tokens_treats_a_zero_output_bound_as_a_bound() -> None:
    """``max_output_tokens=0`` is an explicit "no output", not an omitted field."""
    assert estimate_tokens(prompt_chars=8, max_output_tokens=0, default_output_tokens=4_096) == 2


def test_estimate_tokens_clamps_hostile_inputs() -> None:
    """Negatives cannot shrink a hold: both figures floor at zero."""
    assert estimate_tokens(prompt_chars=-100, max_output_tokens=-50, default_output_tokens=4_096) == 0
