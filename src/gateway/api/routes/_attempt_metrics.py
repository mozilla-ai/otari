"""Metrics on upstream attempts: how each ended, how long it took, and the ones abandoned before their first chunk.

Labeled by the provider instance and model a usage row names, the same pair the
token and cost metrics carry, so a dashboard joins them on one key.
"""

from __future__ import annotations

from gateway.metrics import REGISTRY, Counter, Histogram

ABANDONED_ATTEMPTS = Counter(
    "gateway_abandoned_attempts",
    "Total upstream attempts abandoned before their first chunk (provider fallback / timeout waste)",
    ["provider", "model", "reason", "position"],
    registry=REGISTRY,
)


def record_abandoned_attempt(provider: str, model: str, reason: str, position: int) -> None:
    """Record an upstream attempt abandoned before it produced its first chunk.

    ``reason`` is one of ``timeout`` (the first-chunk wait elapsed),
    ``build_error`` (opening the upstream stream failed), or ``upstream_error``
    (the upstream raised before yielding a chunk). ``position`` is the attempt's
    index in the resolved routing plan; label cardinality stays bounded by the
    plan length.
    """
    ABANDONED_ATTEMPTS.labels(provider=provider, model=model, reason=reason, position=str(position)).inc()


UPSTREAM_ATTEMPTS = Counter(
    "gateway_upstream_attempts",
    "Upstream attempts by how they ended, failed fallback candidates included",
    ["provider", "model", "outcome", "status_code", "fallback"],
    registry=REGISTRY,
)

# Seconds, as Prometheus expects; buckets span a fast cached answer to a slow reasoning model.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0)

TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "gateway_time_to_first_token_seconds",
    "Seconds from the start of the request to a streamed attempt's first chunk",
    ["provider", "model"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

ATTEMPT_DURATION_SECONDS = Histogram(
    "gateway_attempt_duration_seconds",
    "Seconds from the start of the request to the end of an upstream attempt",
    ["provider", "model", "outcome"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)


def record_attempt(
    provider: str,
    model: str,
    *,
    failed: bool,
    status_code: int | None,
    fallback: bool,
    latency_ms: int | None,
    ttft_ms: int | None,
) -> None:
    """Record how one upstream attempt ended.

    ``status_code`` is the HTTP status that classified a failure (a timeout is a
    504, an unclassified connection failure a 502), ``none`` when the failure
    carried none (a stream that ended without its usage), and ``200`` for a
    success. ``fallback`` is whether the attempt was a candidate after a routing
    plan's first.

    The timings are measured from the start of the request, so they are observed
    only for a plan's first candidate, where that is also the attempt's start: a
    fallback's would include the time its predecessors took.
    """
    outcome = "error" if failed else "success"
    status = "200" if not failed else str(status_code) if status_code is not None else "none"
    UPSTREAM_ATTEMPTS.labels(
        provider=provider, model=model, outcome=outcome, status_code=status, fallback=str(fallback).lower()
    ).inc()
    if fallback:
        return
    if latency_ms is not None:
        ATTEMPT_DURATION_SECONDS.labels(provider=provider, model=model, outcome=outcome).observe(latency_ms / 1000)
    if ttft_ms is not None:
        TIME_TO_FIRST_TOKEN_SECONDS.labels(provider=provider, model=model).observe(ttft_ms / 1000)
