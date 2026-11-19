"""Unit tests for reading an offering's time to first token and throughput off its samples."""

import pytest

from gateway.core.usage import output_tokens_per_second
from gateway.repositories.usage_and_telemetry import SpeedSample
from gateway.services.usage_and_telemetry._speed import MIN_SPEED_SAMPLES, offering_speed, speed_by_selector


def _sample(
    *,
    ttft_ms: int | None = 200,
    latency_ms: int = 1000,
    completion_tokens: int | None = 100,
    provider: str | None = "nebius",
    model: str = "glm",
) -> SpeedSample:
    return SpeedSample(
        provider=provider, model=model, ttft_ms=ttft_ms, latency_ms=latency_ms, completion_tokens=completion_tokens
    )


@pytest.mark.parametrize(
    ("completion_tokens", "latency_ms", "expected"),
    [(300, 2000, 150.0), (19, 1000, None), (None, 1000, None), (300, 0, None), (300, None, None)],
)
def test_output_tokens_per_second(
    completion_tokens: int | None, latency_ms: int | None, expected: float | None
) -> None:
    assert output_tokens_per_second(completion_tokens, latency_ms) == expected


def test_too_few_samples_read_as_unknown() -> None:
    assert offering_speed([_sample() for _ in range(MIN_SPEED_SAMPLES - 1)]) is None


def test_the_slow_tail_of_a_rate_is_its_low_end() -> None:
    # 100 to 290 output tokens over one second each.
    samples = [_sample(ttft_ms=100 + step, completion_tokens=100 + 10 * step) for step in range(20)]

    speed = offering_speed(samples)

    assert speed is not None
    assert speed.ttft_ms_p50 == 110
    assert speed.ttft_ms_p95 == 118
    assert speed.output_tokens_per_second_p50 == 195.0
    assert speed.output_tokens_per_second_p95 == pytest.approx(109.5)


def test_each_measure_needs_its_own_samples() -> None:
    # Unstreamed requests have a rate and no first token.
    speed = offering_speed([_sample(ttft_ms=None) for _ in range(MIN_SPEED_SAMPLES)])

    assert speed is not None
    assert speed.ttft_samples == 0
    assert speed.ttft_ms_p50 is None
    assert speed.throughput_samples == MIN_SPEED_SAMPLES
    assert speed.output_tokens_per_second_p50 == 100.0


def test_samples_are_matched_to_an_offering_by_either_spelling() -> None:
    offerings = [("nebius:glm", "nebius", "glm"), ("fireworks:glm", "fireworks", "glm")]
    # An older row carries the whole selector in ``model`` and no provider.
    samples = [_sample() for _ in range(10)] + [
        _sample(provider=None, model="nebius:glm") for _ in range(MIN_SPEED_SAMPLES - 10)
    ]
    samples += [_sample(provider="other", model="glm") for _ in range(MIN_SPEED_SAMPLES)]

    speed = speed_by_selector(samples, offerings)

    assert set(speed) == {"nebius:glm"}
    assert speed["nebius:glm"].ttft_samples == MIN_SPEED_SAMPLES


def test_a_named_instance_is_not_read_as_the_offering_its_model_spells() -> None:
    # ``home_lab:nebius:glm`` is recorded as provider ``home_lab``, model ``nebius:glm``.
    samples = [_sample(provider="home_lab", model="nebius:glm") for _ in range(MIN_SPEED_SAMPLES)]

    assert speed_by_selector(samples, [("nebius:glm", "nebius", "glm")]) == {}
