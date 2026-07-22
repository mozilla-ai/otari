"""Provider-neutral token meters and threshold-aware pricing helpers."""

from dataclasses import dataclass
from typing import Any, Literal

from gateway.core.usage import (
    cache_read_tokens_of,
    cache_tokens_in_prompt_of,
    cache_write_1h_tokens_of,
    cache_write_tokens_of,
)
from gateway.models.entities import ModelPricing

RATE_FIELDS = (
    "input_price_per_million",
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)


@dataclass(frozen=True)
class BillableUsage:
    """Canonical token meters without changing provider-reported usage fields."""

    total_input_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int

    @property
    def cache_write_base_tokens(self) -> int:
        return self.cache_write_tokens - self.cache_write_1h_tokens


def billable_usage(usage: Any) -> BillableUsage:
    """Normalize additive and subset cache conventions into billable meters."""
    prompt_tokens = max(int(getattr(usage, "prompt_tokens", 0) or 0), 0)
    completion_tokens = max(int(getattr(usage, "completion_tokens", 0) or 0), 0)
    cache_read = max(cache_read_tokens_of(usage), 0)
    cache_write = max(cache_write_tokens_of(usage), 0)
    cache_write_1h = min(max(cache_write_1h_tokens_of(usage), 0), cache_write)

    if cache_tokens_in_prompt_of(usage):
        cache_read = min(cache_read, prompt_tokens)
        cache_write = min(cache_write, prompt_tokens - cache_read)
        cache_write_1h = min(cache_write_1h, cache_write)
        total_input = prompt_tokens
    else:
        total_input = prompt_tokens + cache_read + cache_write

    return BillableUsage(
        total_input_tokens=total_input,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cache_write_1h_tokens=cache_write_1h,
    )


def effective_rates(pricing: ModelPricing, total_input_tokens: int | float) -> dict[str, float | None]:
    """Select base or whole-request threshold rates for a usage event."""
    rates = {field: getattr(pricing, field) for field in RATE_FIELDS}
    tiers = pricing.pricing_tiers or []
    applicable = [tier for tier in tiers if int(tier.get("min_input_tokens", 0)) <= total_input_tokens]
    if not applicable:
        return rates
    selected = max(applicable, key=lambda tier: int(tier["min_input_tokens"]))
    for field in RATE_FIELDS:
        value = selected.get(field)
        if value is not None:
            rates[field] = float(value)
    return rates


def estimate_metered_cost(
    pricing: ModelPricing,
    *,
    estimated_input_tokens: float,
    estimated_output_tokens: int,
    cache_write_ttl: Literal["5m", "1h"] | None = None,
) -> float:
    """Conservatively price a request before provider usage is available.

    A requested Anthropic cache entry is reported as an additional input meter,
    not as a discounted portion of ordinary prompt input. Until the provider
    reports its exact meter, reserve for every estimated prompt token becoming
    a cache write. This also selects threshold rates from the corresponding
    total billable input.
    """
    input_tokens = max(estimated_input_tokens, 0.0)
    output_tokens = max(estimated_output_tokens, 0)
    cache_write_tokens = input_tokens if cache_write_ttl is not None else 0.0
    rates = effective_rates(pricing, input_tokens + cache_write_tokens)

    input_rate = rates["input_price_per_million"]
    output_rate = rates["output_price_per_million"]
    assert input_rate is not None
    assert output_rate is not None
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    if cache_write_ttl == "1h":
        cache_write_rate = rates["cache_write_1h_price_per_million"]
        if cache_write_rate is None:
            cache_write_rate = rates["cache_write_price_per_million"]
    else:
        cache_write_rate = rates["cache_write_price_per_million"]
    if cache_write_rate is not None:
        cost += cache_write_tokens * cache_write_rate / 1_000_000
    return cost


def calculate_metered_cost(
    pricing: ModelPricing,
    usage: Any,
) -> tuple[float, dict[str, int], list[dict[str, float | int | str]]]:
    """Price normalized token meters and return auditable charge lines."""
    billable = billable_usage(usage)
    rates = effective_rates(pricing, billable.total_input_tokens)
    input_rate = rates["input_price_per_million"]
    output_rate = rates["output_price_per_million"]
    assert input_rate is not None
    assert output_rate is not None

    read_rate = rates["cache_read_price_per_million"]
    write_rate = rates["cache_write_price_per_million"]
    configured_1h_rate = rates["cache_write_1h_price_per_million"]
    write_1h_rate = configured_1h_rate if configured_1h_rate is not None else write_rate

    fresh_input = billable.total_input_tokens
    if read_rate is not None:
        fresh_input -= billable.cache_read_tokens
    if write_rate is not None:
        fresh_input -= billable.cache_write_base_tokens
    if write_1h_rate is not None:
        fresh_input -= billable.cache_write_1h_tokens

    meters = {
        "total_input_tokens": billable.total_input_tokens,
        "fresh_input_tokens": fresh_input,
        "cache_read_tokens": billable.cache_read_tokens,
        "cache_write_tokens": billable.cache_write_tokens,
        "cache_write_1h_tokens": billable.cache_write_1h_tokens,
        "completion_tokens": billable.completion_tokens,
    }
    lines: list[dict[str, float | int | str]] = []

    def add_line(meter: str, units: int, rate: float) -> None:
        if units:
            lines.append({"meter": meter, "units": units, "rate_per_million": rate, "cost": units * rate / 1_000_000})

    add_line("input", fresh_input, input_rate)
    cost = fresh_input * input_rate / 1_000_000
    add_line("output", billable.completion_tokens, output_rate)
    cost += billable.completion_tokens * output_rate / 1_000_000
    if read_rate is not None:
        add_line("cache_read", billable.cache_read_tokens, read_rate)
        cost += billable.cache_read_tokens * read_rate / 1_000_000
    if write_rate is not None:
        add_line("cache_write_5m", billable.cache_write_base_tokens, write_rate)
        cost += billable.cache_write_base_tokens * write_rate / 1_000_000
    if write_1h_rate is not None:
        add_line("cache_write_1h", billable.cache_write_1h_tokens, write_1h_rate)
        cost += billable.cache_write_1h_tokens * write_1h_rate / 1_000_000
    return cost, meters, lines
