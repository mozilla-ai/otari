"""Internal usage carrier that extends the provider usage shape with cache counts.

The provider-neutral usage object that flows through the request pipeline is
``any_llm``'s :class:`CompletionUsage` (the OpenAI shape). It exposes a single
``prompt_tokens_details.cached_tokens`` read slot and has no representation for a
cache *write* (creation) charge, which Anthropic bills separately.

:class:`GatewayUsage` subclasses ``CompletionUsage`` so it remains type-compatible
with every signature that already accepts a ``CompletionUsage`` while carrying two
explicit integers: ``cache_read_tokens`` and ``cache_write_tokens``. Capturing these
as plain integers (rather than relying on ``prompt_tokens_details``) lets the report
builder forward them uniformly across providers, including Anthropic cache writes.
"""

import math
from typing import Any

from any_llm.types.completion import CompletionUsage


class GatewayUsage(CompletionUsage):
    """Usage with explicit provider cache-token counts.

    ``cache_read_tokens`` and ``cache_write_tokens`` default to ``0`` so the carrier
    behaves like a plain ``CompletionUsage`` for providers that report no cache usage.
    """

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    cache_tokens_in_prompt: bool = True
    """Whether ``cache_read_tokens`` / ``cache_write_tokens`` are already counted
    within ``prompt_tokens``.

    ``True`` for OpenAI-shaped usage, where cached tokens are a subset of
    ``prompt_tokens`` (the whole prompt is billed and the cached slice is merely a
    re-priced discount). ``False`` for the Anthropic path, where ``input_tokens``
    excludes cache reads/writes and they are reported as separate additive buckets.
    The cost calculation reads this to normalize both shapes onto a single
    convention (see ``_compute_cost`` in ``_pipeline.py``). Defaults to ``True`` so a
    plain ``CompletionUsage`` and every OpenAI-style path need no change.
    """

    @staticmethod
    def external_extras(usage: CompletionUsage) -> dict[str, Any]:
        """``usage.model_extra`` with any key that names a ``GatewayUsage`` field dropped.

        A provider is free to put anything under ``model_extra`` (any-llm forwards
        the wire response's extra keys verbatim), and this carrier is built by
        splatting that dict as constructor kwargs. Without this filter, a
        provider-supplied key that happens to share a name with one of our own
        accounting fields (``cache_write_tokens``, ``cache_write_1h_tokens``,
        ``cache_tokens_in_prompt``, ...) would set that field directly instead of
        being forwarded as an inert extra, which is a billing-integrity hole, not a
        cosmetic one. Callers that also pass explicit values for some of these
        fields should still list them explicitly after merging this in; this only
        strips names that could otherwise sneak through unrequested.
        """
        return {
            k: v for k, v in (usage.model_extra or {}).items() if k not in GatewayUsage.model_fields
        }

    @classmethod
    def from_completion_usage(
        cls,
        usage: CompletionUsage,
        *,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cache_write_1h_tokens: int | None = None,
        cache_tokens_in_prompt: bool | None = None,
    ) -> "GatewayUsage":
        """Build a ``GatewayUsage`` from a base ``CompletionUsage`` plus cache counts.

        When ``cache_read_tokens`` / ``cache_write_tokens`` are left as ``None`` they
        fall back to whatever ``usage`` already carries: the explicit fields when
        ``usage`` is itself a :class:`GatewayUsage`, otherwise the OpenAI-style
        ``prompt_tokens_details.cached_tokens`` (a subset of ``prompt_tokens``, purely
        informational for re-pricing). An explicit ``0`` is honored and does not
        trigger the fallback. ``cache_tokens_in_prompt`` defaults to the source's
        value (``True`` for a plain ``CompletionUsage``).
        """
        if cache_read_tokens is None:
            cache_read_tokens = cache_read_tokens_of(usage)
        if cache_write_tokens is None:
            cache_write_tokens = cache_write_tokens_of(usage)
        if cache_write_1h_tokens is None:
            cache_write_1h_tokens = cache_write_1h_tokens_of(usage)
        if cache_tokens_in_prompt is None:
            cache_tokens_in_prompt = cache_tokens_in_prompt_of(usage)
        # Forward any-llm's own extras (otari#337) so provider_latency_ms_of can
        # still read them, minus anything that would collide with one of our own
        # fields (see external_extras); explicit fields below are still listed so
        # the cache counts resolved above always win regardless of the filter.
        fields = cls.external_extras(usage)
        fields.update(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            completion_tokens_details=usage.completion_tokens_details,
            prompt_tokens_details=usage.prompt_tokens_details,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            cache_tokens_in_prompt=cache_tokens_in_prompt,
        )
        return cls(**fields)


def cache_read_tokens_of(usage: CompletionUsage) -> int:
    """Return the cache-read count carried by ``usage``.

    Handles both :class:`GatewayUsage` (explicit field) and a plain
    ``CompletionUsage`` (falls back to ``prompt_tokens_details.cached_tokens``).
    """
    if isinstance(usage, GatewayUsage):
        return usage.cache_read_tokens
    if usage.prompt_tokens_details is not None:
        return usage.prompt_tokens_details.cached_tokens or 0
    return 0


def cache_write_tokens_of(usage: CompletionUsage) -> int:
    """Return the cache-write count carried by ``usage`` (0 for non-Anthropic)."""
    if isinstance(usage, GatewayUsage):
        return usage.cache_write_tokens
    return 0


def cache_write_1h_tokens_of(usage: CompletionUsage) -> int:
    """Return the Anthropic 1-hour cache-write subset carried by ``usage``."""
    if isinstance(usage, GatewayUsage):
        return usage.cache_write_1h_tokens
    return 0


def cache_tokens_in_prompt_of(usage: CompletionUsage) -> bool:
    """Whether the usage's cache counts are already included in ``prompt_tokens``.

    A plain ``CompletionUsage`` is always OpenAI-shaped (``cached_tokens`` is a
    subset of ``prompt_tokens``), so it returns ``True``. A :class:`GatewayUsage`
    reports its explicit flag; the Anthropic path sets it ``False``.
    """
    if isinstance(usage, GatewayUsage):
        return usage.cache_tokens_in_prompt
    return True


# Provider-specific field(s) any-llm leaves in ``usage.model_extra`` that sum to
# actual compute time, and the multiplier that converts the sum's unit to
# milliseconds (otari#337). Absent here means "not reported": most providers
# expose no server-side timing in the response body at all.
#
# Ollama's own ``total_duration`` is the whole server-side request, including
# ``load_duration`` (loading the model into memory), which dominates on a cold
# model and is not compute time in the sense this column reports for Groq.
# any-llm forwards ``prompt_eval_duration`` and ``eval_duration`` separately
# (see any_llm.providers.ollama.utils._extract_ollama_timing_details); their
# sum is Ollama's actual prompt+generation compute time, excluding load.
_PROVIDER_LATENCY_FIELDS: dict[str, tuple[tuple[str, ...], float]] = {
    "groq": (("total_time",), 1_000.0),  # seconds
    "ollama": (("prompt_eval_duration", "eval_duration"), 1e-6),  # nanoseconds
}


def provider_latency_ms_of(usage: CompletionUsage, provider: str | None) -> int | None:
    """Best-effort provider-reported compute time for ``provider``, in ms.

    ``None`` when the provider is not in the table, any of its fields is
    absent, negative, or not a plain finite number: this only enriches a
    row and must never raise or affect billing.
    """
    if provider is None:
        return None
    field = _PROVIDER_LATENCY_FIELDS.get(provider)
    if field is None:
        return None
    keys, to_ms = field
    extras = usage.model_extra or {}
    raw_values: list[int | float] = []
    for key in keys:
        raw = extras.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
            return None
        raw_values.append(raw)
    try:
        scaled = sum(raw_values) * to_ms
    except OverflowError:
        return None
    if not math.isfinite(scaled):
        return None
    return round(scaled)
