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

from any_llm.types.completion import CompletionUsage


class GatewayUsage(CompletionUsage):
    """Usage with explicit provider cache-token counts.

    ``cache_read_tokens`` and ``cache_write_tokens`` default to ``0`` so the carrier
    behaves like a plain ``CompletionUsage`` for providers that report no cache usage.
    """

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def from_completion_usage(
        cls,
        usage: CompletionUsage,
        *,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> "GatewayUsage":
        """Build a ``GatewayUsage`` from a base ``CompletionUsage`` plus cache counts.

        When ``cache_read_tokens`` / ``cache_write_tokens`` are left as ``None`` they
        fall back to whatever ``usage`` already carries: the explicit fields when
        ``usage`` is itself a :class:`GatewayUsage`, otherwise the OpenAI-style
        ``prompt_tokens_details.cached_tokens`` (a subset of ``prompt_tokens``, purely
        informational for re-pricing). An explicit ``0`` is honored and does not
        trigger the fallback.
        """
        if cache_read_tokens is None:
            cache_read_tokens = cache_read_tokens_of(usage)
        if cache_write_tokens is None:
            cache_write_tokens = cache_write_tokens_of(usage)
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            completion_tokens_details=usage.completion_tokens_details,
            prompt_tokens_details=usage.prompt_tokens_details,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
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
