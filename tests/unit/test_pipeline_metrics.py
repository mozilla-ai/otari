"""Unit tests for the token, cost and inline settlement metrics the pipeline records."""

import pytest

from gateway.api.routes._attempt_metrics import record_attempt
from gateway.api.routes._pipeline import record_cost, record_inline_cost_settlement, record_tokens
from gateway.metrics import REGISTRY


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Read a metric sample value from the registry, returning 0.0 if not found."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_record_tokens_increments_counters() -> None:
    input_labels = {"provider": "test-prov", "model": "test-model", "type": "input"}
    output_labels = {"provider": "test-prov", "model": "test-model", "type": "output"}
    before_in = _sample("gateway_tokens_total", input_labels)
    before_out = _sample("gateway_tokens_total", output_labels)

    record_tokens("test-prov", "test-model", 100, 50)

    assert _sample("gateway_tokens_total", input_labels) - before_in == 100.0
    assert _sample("gateway_tokens_total", output_labels) - before_out == 50.0


def test_record_tokens_skips_zero_values() -> None:
    labels_in = {"provider": "zero-prov", "model": "zero-model", "type": "input"}
    labels_out = {"provider": "zero-prov", "model": "zero-model", "type": "output"}
    before_in = _sample("gateway_tokens_total", labels_in)
    before_out = _sample("gateway_tokens_total", labels_out)

    record_tokens("zero-prov", "zero-model", 0, 0)

    assert _sample("gateway_tokens_total", labels_in) == before_in
    assert _sample("gateway_tokens_total", labels_out) == before_out


def test_record_cost_observes_histogram() -> None:
    labels = {"provider": "cost-prov", "model": "cost-model"}
    before_count = _sample("gateway_request_cost_dollars_count", labels)

    record_cost("cost-prov", "cost-model", 1.23)

    assert _sample("gateway_request_cost_dollars_count", labels) - before_count == 1.0
    assert _sample("gateway_request_cost_dollars_sum", labels) >= 1.23


@pytest.mark.parametrize("outcome", ["attached", "unattached", "timeout"])
def test_record_inline_cost_settlement_increments_counter(outcome: str) -> None:
    labels = {"outcome": outcome}
    before = _sample("gateway_inline_cost_settlements_total", labels)

    record_inline_cost_settlement(outcome)

    assert _sample("gateway_inline_cost_settlements_total", labels) - before == 1.0


def test_record_attempt_counts_a_success_and_its_timings() -> None:
    labels = {"provider": "att-prov", "model": "att-model", "outcome": "success", "status_code": "200"}
    before = _sample("gateway_upstream_attempts_total", {**labels, "fallback": "false"})
    ttft_before = _sample("gateway_time_to_first_token_seconds_count", {"provider": "att-prov", "model": "att-model"})

    record_attempt(
        "att-prov", "att-model", failed=False, status_code=None, fallback=False, latency_ms=1500, ttft_ms=300
    )

    assert _sample("gateway_upstream_attempts_total", {**labels, "fallback": "false"}) - before == 1.0
    assert (
        _sample("gateway_time_to_first_token_seconds_count", {"provider": "att-prov", "model": "att-model"})
        - ttft_before
        == 1.0
    )
    duration = {"provider": "att-prov", "model": "att-model", "outcome": "success"}
    assert _sample("gateway_attempt_duration_seconds_sum", duration) >= 1.5


@pytest.mark.parametrize(("status_code", "label"), [(503, "503"), (None, "none")])
def test_record_attempt_counts_a_failure_by_status(status_code: int | None, label: str) -> None:
    labels = {
        "provider": "fail-prov",
        "model": "fail-model",
        "outcome": "error",
        "status_code": label,
        "fallback": "true",
    }
    before = _sample("gateway_upstream_attempts_total", labels)

    record_attempt(
        "fail-prov", "fail-model", failed=True, status_code=status_code, fallback=True, latency_ms=None, ttft_ms=None
    )

    assert _sample("gateway_upstream_attempts_total", labels) - before == 1.0


def test_record_attempt_skips_timings_for_a_fallback() -> None:
    duration = {"provider": "fb-prov", "model": "fb-model", "outcome": "success"}
    before = _sample("gateway_attempt_duration_seconds_count", duration)

    record_attempt("fb-prov", "fb-model", failed=False, status_code=None, fallback=True, latency_ms=900, ttft_ms=200)

    assert _sample("gateway_attempt_duration_seconds_count", duration) - before == 0.0
