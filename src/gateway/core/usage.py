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

from typing import Any

from any_llm.types.completion import CompletionUsage
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage


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
        return cls(
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


def anthropic_cache_write_1h_tokens(usage: Any) -> int:
    """Read Anthropic's optional 1-hour cache-creation breakdown."""
    cache_creation = getattr(usage, "cache_creation", None)
    return getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0


def anthropic_billable_usage(usage: Any) -> GatewayUsage:
    """Read an Anthropic usage object, using per-iteration totals when present.

    Anthropic reports compaction sampling as an ``iterations`` list, whose parts
    sum to what the request is actually billed; the top-level counts describe only
    the last one. ``cache_tokens_in_prompt`` is ``False`` because ``input_tokens``
    excludes the cache buckets on this path.

    Lives in ``core`` because both the Messages route (for billing) and the
    Messages tool-loop strategy (for the Reprise usage snapshot) need it, and a
    service may not import the API layer.
    """
    billable_parts = list(getattr(usage, "iterations", None) or []) or [usage]
    input_tokens = sum((getattr(part, "input_tokens", None) or 0) for part in billable_parts)
    output_tokens = sum((getattr(part, "output_tokens", None) or 0) for part in billable_parts)
    return GatewayUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=sum((getattr(part, "cache_read_input_tokens", None) or 0) for part in billable_parts),
        cache_write_tokens=sum(
            (getattr(part, "cache_creation_input_tokens", None) or 0) for part in billable_parts
        ),
        cache_write_1h_tokens=sum(anthropic_cache_write_1h_tokens(part) for part in billable_parts),
        cache_tokens_in_prompt=False,
    )


def responses_usage(usage: ResponseUsage | OpenResponsesUsage | None) -> GatewayUsage | None:
    """Read an OpenAI Responses usage object, or ``None`` when there is none.

    OpenAI-shaped, so ``cached_tokens`` is a slice of ``input_tokens`` rather than
    an additive bucket and ``cache_tokens_in_prompt`` keeps its ``True`` default.
    The Responses API reports no cache-write charge.

    In ``core`` for the same reason as :func:`anthropic_billable_usage`: both the
    Responses route and its tool-loop strategy read it, on opposite sides of the
    service/API boundary.
    """
    if usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    return GatewayUsage(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cache_read_tokens=(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0,
    )


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
