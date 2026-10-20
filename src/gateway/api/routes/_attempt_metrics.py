"""The counter of upstream attempts abandoned before their first chunk."""

from __future__ import annotations

from gateway.metrics import REGISTRY, Counter

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
