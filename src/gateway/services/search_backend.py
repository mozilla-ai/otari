"""Standalone search dispatch for ``POST /v1/search``.

Otari's other web-search surface (:mod:`gateway.services.web_search_backend`)
is an in-loop tool: the model emits ``web_search(query=…)`` and the gateway
answers it mid-completion. This module is the direct one. A caller posts a
query and gets ranked results back, on the same auth, budget, and usage-logging
path as a completion, which is what a sidecar proxying a search API through the
gateway needs.

Tools are declared in ``config.yml``, keyed by the name callers pass as
``search_tool_name`` (or in the ``/v1/search/{tool}`` path)::

    search_tools:
      exa-search:
        provider: exa
        api_key: ${EXA_API_KEY}
        options:
          type: fast

``options`` is a mapping of provider-native request fields used as defaults;
fields derived from the request win over it, and ``query`` always comes from
the caller. Every entry is validated at startup by
``GatewayConfig.validate_search_tools``, so an unsupported provider or a
missing API key fails before the first request.

The wire contract of the route follows LiteLLM's ``/v1/search`` (itself shaped
after Perplexity's Search API) so a caller migrating off the LiteLLM proxy
needs no request changes. Translating that contract to and from each provider's
native shape happens here: the route never sees a provider payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.core.config import SEARCH_PROVIDERS, GatewayConfig

EXA_PROVIDER = "exa"

_DEFAULT_API_BASE = {EXA_PROVIDER: "https://api.exa.ai"}
_DEFAULT_TIMEOUT_S = 30.0

# Result-count defaults and the ceiling the route also enforces on the request
# field. Applied again here because a tool's ``options`` can carry its own
# ``numResults``, which must not escape the cap either.
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_CAP = 20

# Perplexity meters per-page content in tokens and Exa in characters, so the
# request's ``max_tokens_per_page`` is converted at the usual rough ratio and
# clamped to the range Exa documents for ``contents.text.maxCharacters``.
DEFAULT_MAX_TOKENS_PER_PAGE = 1024
_CHARS_PER_TOKEN = 4
_EXA_MAX_CHARACTERS = 10_000

# How much of an upstream error body is kept for the usage log's error_message.
# The full body can be large and is never returned to the caller.
_ERROR_BODY_CHARS = 500


class SearchToolError(ValueError):
    """The requested search tool is unknown, ambiguous, or not configured."""


class SearchProviderError(RuntimeError):
    """The search provider could not be reached or returned malformed data."""


@dataclass(frozen=True)
class SearchTool:
    """A resolved ``search_tools`` entry, ready to dispatch against."""

    name: str
    provider: str
    api_key: str
    api_base: str
    timeout_s: float
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchQuery:
    """One search request in the gateway's provider-neutral terms."""

    query: str
    max_results: int | None = None
    domain_filter: tuple[str, ...] = ()
    country: str | None = None
    max_tokens_per_page: int | None = None


@dataclass(frozen=True)
class SearchHit:
    """One normalized result."""

    url: str
    title: str | None = None
    snippet: str | None = None
    date: str | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """A completed provider search.

    ``cost_usd`` is the provider's own reported charge when it reports one (Exa
    returns ``costDollars.total``), which the route bills in preference to a
    configured flat rate. ``None`` means the provider said nothing about cost.
    """

    results: list[SearchHit]
    cost_usd: float | None = None


def resolve_search_tool(config: GatewayConfig, name: str | None) -> SearchTool:
    """Resolve a request's tool name against ``config.search_tools``.

    The name may be omitted when exactly one tool is configured, which is the
    common single-provider deployment. Raises :class:`SearchToolError` when no
    tool is configured, when the name is unknown, and when it is omitted with
    several configured; the route maps all three to a 400, since each is
    something the caller can act on.
    """
    configured = config.search_tools
    if not configured:
        msg = "No search tools are configured. Declare one under 'search_tools' in config.yml."
        raise SearchToolError(msg)

    if name is None:
        if len(configured) > 1:
            msg = (
                "Several search tools are configured; name one in 'search_tool_name' or use "
                f"POST /v1/search/{{tool}}. Available: {', '.join(sorted(configured))}."
            )
            raise SearchToolError(msg)
        name = next(iter(configured))

    entry = configured.get(name)
    if entry is None:
        msg = f"Unknown search tool '{name}'. Available: {', '.join(sorted(configured))}."
        raise SearchToolError(msg)

    provider = str(entry.get("provider") or name)
    api_key = entry.get("api_key")
    # Defense in depth: startup validation already guarantees both, so this only
    # fires for a config built in-process. Refusing here beats calling an unknown
    # provider or calling a known one unauthenticated.
    if provider not in SEARCH_PROVIDERS or not api_key:
        msg = f"Search tool '{name}' is not configured correctly."
        raise SearchToolError(msg)

    api_base = str(entry.get("api_base") or _DEFAULT_API_BASE[provider]).rstrip("/")
    options = entry.get("options")
    return SearchTool(
        name=name,
        provider=provider,
        api_key=str(api_key),
        api_base=api_base,
        timeout_s=float(entry.get("timeout") or _DEFAULT_TIMEOUT_S),
        options=dict(options) if isinstance(options, dict) else {},
    )


async def run_search(tool: SearchTool, query: SearchQuery) -> SearchOutcome:
    """Dispatch one search against the tool's provider."""
    if tool.provider == EXA_PROVIDER:
        return await _search_exa(tool, query)
    # Unreachable via config: startup validation rejects unknown providers.
    msg = f"Unsupported search provider '{tool.provider}'."
    raise SearchProviderError(msg)


# --------------------------------------------------------------------------- #
# Exa
# --------------------------------------------------------------------------- #


def build_exa_payload(tool: SearchTool, query: SearchQuery) -> dict[str, Any]:
    """Build Exa's ``POST /search`` body from the neutral request.

    The tool's ``options`` are the base, so an operator can pin Exa-native knobs
    (``type``, ``category``, ``moderation``) that the LiteLLM-shaped request has
    no field for. Request-derived fields are layered on top, and ``query`` is
    written last so no configured option can displace the caller's query.
    """
    payload: dict[str, Any] = dict(tool.options)

    requested = query.max_results or payload.get("numResults") or DEFAULT_MAX_RESULTS
    payload["numResults"] = max(1, min(int(requested), MAX_RESULTS_CAP))

    # Perplexity's convention: a leading '-' excludes a domain instead of
    # restricting to it. Mapped to Exa's two separate lists so a '-example.com'
    # entry excludes rather than being sent as a domain that matches nothing.
    include = [d for d in query.domain_filter if not d.startswith("-")]
    exclude = [d[1:] for d in query.domain_filter if d.startswith("-") and len(d) > 1]
    if include:
        payload["includeDomains"] = include
    if exclude:
        payload["excludeDomains"] = exclude

    if query.country:
        payload["userLocation"] = query.country

    # Exa returns page text only when asked for it, and the response's
    # ``snippet`` is that text, so contents are always requested. A tool that
    # pins its own ``contents.text`` keeps it unless the caller asked for a
    # specific per-page size.
    contents = dict(payload.get("contents") or {})
    if query.max_tokens_per_page is not None or "text" not in contents:
        max_tokens = query.max_tokens_per_page or DEFAULT_MAX_TOKENS_PER_PAGE
        max_chars = max(1, min(max_tokens * _CHARS_PER_TOKEN, _EXA_MAX_CHARACTERS))
        # Merge rather than replace: ``text`` has siblings (``verbosity``,
        # ``includeHtmlTags``) that a tool may have pinned, and only the size is
        # the caller's to set.
        pinned = contents.get("text")
        text_options = dict(pinned) if isinstance(pinned, dict) else {}
        text_options["maxCharacters"] = max_chars
        contents["text"] = text_options
    payload["contents"] = contents

    payload["query"] = query.query
    return payload


async def _search_exa(tool: SearchTool, query: SearchQuery) -> SearchOutcome:
    payload = build_exa_payload(tool, query)
    try:
        async with httpx.AsyncClient(timeout=tool.timeout_s) as client:
            response = await client.post(
                f"{tool.api_base}/search",
                json=payload,
                headers={"x-api-key": tool.api_key},
            )
    except httpx.HTTPError as exc:
        msg = f"exa search request failed: {exc}"
        raise SearchProviderError(msg) from exc

    if response.status_code >= 400:
        msg = f"exa search returned HTTP {response.status_code}: {response.text[:_ERROR_BODY_CHARS]}"
        raise SearchProviderError(msg)

    try:
        body = response.json()
    except ValueError as exc:
        msg = "exa search returned a body that is not JSON"
        raise SearchProviderError(msg) from exc
    if not isinstance(body, dict):
        msg = f"exa search returned a {type(body).__name__} body, expected an object"
        raise SearchProviderError(msg)

    return SearchOutcome(results=_exa_hits(body), cost_usd=_exa_cost(body))


def _exa_hits(body: dict[str, Any]) -> list[SearchHit]:
    raw = body.get("results")
    if not isinstance(raw, list):
        msg = "exa search response has no 'results' list"
        raise SearchProviderError(msg)

    hits: list[SearchHit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        hits.append(
            SearchHit(
                url=url,
                title=_text_or_none(item.get("title")),
                # Full text when it was returned, else the highlight sentences,
                # which is what a snippet-only search mode yields.
                snippet=_text_or_none(item.get("text")) or _highlights(item.get("highlights")),
                date=_text_or_none(item.get("publishedDate")),
            )
        )
    return hits


def _exa_cost(body: dict[str, Any]) -> float | None:
    costs = body.get("costDollars")
    if not isinstance(costs, dict):
        return None
    total = costs.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    return max(float(total), 0.0)


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _highlights(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    joined = " ".join(part.strip() for part in value if isinstance(part, str) and part.strip())
    return joined or None
