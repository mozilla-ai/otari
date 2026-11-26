"""How fast an offering answered, read off the timings its served requests recorded."""

import statistics
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import timedelta

from gateway.core.usage import output_tokens_per_second
from gateway.repositories.usage_and_telemetry import SpeedSample

# Speed changes with a provider's load, so a short window says more about today
# than the month the spend columns cover.
SPEED_WINDOW = timedelta(days=7)

# The latest requests read per offering, which bounds the read on a busy one.
SPEED_SAMPLE_CAP = 500

# Fewer samples than this and a 95th percentile is one or two requests.
MIN_SPEED_SAMPLES = 20


@dataclass(frozen=True)
class OfferingSpeed:
    """Time to first token and throughput over an offering's recent requests.

    A 95th percentile is the slow tail of each: the time to first token 95% of
    requests beat, and the rate 95% of requests reached or beat. A value is null
    when fewer than ``MIN_SPEED_SAMPLES`` requests measured it: only a streamed
    request has a first token, and only a long enough response has a rate.
    """

    ttft_samples: int
    ttft_ms_p50: int | None
    ttft_ms_p95: int | None
    throughput_samples: int
    output_tokens_per_second_p50: float | None
    output_tokens_per_second_p95: float | None


def _cut_points(values: list[float]) -> list[float] | None:
    """Every fifth percentile of ``values``, from the 5th to the 95th, or None on too few samples."""
    if len(values) < MIN_SPEED_SAMPLES:
        return None
    return statistics.quantiles(values, n=20, method="inclusive")


def offering_speed(samples: Collection[SpeedSample]) -> OfferingSpeed | None:
    """The percentiles over one offering's samples, or None when neither measure has enough of them."""
    ttfts = [float(sample.ttft_ms) for sample in samples if sample.ttft_ms is not None]
    rates = [
        rate
        for sample in samples
        if (rate := output_tokens_per_second(sample.completion_tokens, sample.latency_ms)) is not None
    ]
    ttft_cuts = _cut_points(ttfts)
    rate_cuts = _cut_points(rates)
    if ttft_cuts is None and rate_cuts is None:
        return None
    return OfferingSpeed(
        ttft_samples=len(ttfts),
        ttft_ms_p50=round(ttft_cuts[9]) if ttft_cuts else None,
        ttft_ms_p95=round(ttft_cuts[18]) if ttft_cuts else None,
        throughput_samples=len(rates),
        output_tokens_per_second_p50=round(rate_cuts[9], 1) if rate_cuts else None,
        # A rate's slow tail is its low end.
        output_tokens_per_second_p95=round(rate_cuts[0], 1) if rate_cuts else None,
    )


def speed_by_selector(
    samples: Iterable[SpeedSample], offerings: Collection[tuple[str, str, str]]
) -> dict[str, OfferingSpeed]:
    """Each offering's speed, keyed by selector, from samples spelled either way a usage row names one."""
    by_pair = {(instance, model_id): selector for selector, instance, model_id in offerings}
    selectors = set(by_pair.values())
    grouped: dict[str, list[SpeedSample]] = {}
    for sample in samples:
        selector = by_pair.get((sample.provider or "", sample.model))
        # The whole selector in ``model`` only on a row from before ``provider`` was recorded.
        if selector is None and not sample.provider and sample.model in selectors:
            selector = sample.model
        if selector is not None:
            grouped.setdefault(selector, []).append(sample)
    return {selector: speed for selector, rows in grouped.items() if (speed := offering_speed(rows)) is not None}
