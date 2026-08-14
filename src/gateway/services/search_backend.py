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

The other provider is ``searxng``, which speaks the SearXNG-shaped
``GET {api_base}/search?format=json`` contract that
:mod:`gateway.services.web_search_backend` already dispatches the in-loop tool
to. It needs no API key, so the bundled SearXNG container and the Brave and
Tavily fronting adapters are reachable from this endpoint too, and a tool that
declares no ``api_base`` inherits ``web_search_url``::

    search_tools:
      local:
        provider: searxng   # api_base defaults to web_search_url

Provider calls go through one pooled ``httpx.AsyncClient`` for the process
(:func:`get_search_client`), closed on shutdown by :func:`close_search_client`.
A search is a single request, so a client per call would pay a fresh TCP and TLS
handshake every time and pool nothing.

The wire contract of the route follows LiteLLM's ``/v1/search`` (itself shaped
after Perplexity's Search API) so a caller migrating off the LiteLLM proxy
needs no request changes. Translating that contract to and from each provider's
native shape happens here: the route never sees a provider payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from gateway.core.config import SEARCH_PROVIDERS, SEARCH_PROVIDERS_REQUIRING_API_KEY, GatewayConfig

EXA_PROVIDER = "exa"
SEARXNG_PROVIDER = "searxng"

_DEFAULT_API_BASE = {EXA_PROVIDER: "https://api.exa.ai"}
_DEFAULT_TIMEOUT_S = 30.0

# Query params the gateway owns on a SearXNG request, so a tool's ``options``
# cannot displace the caller's query or ask for a non-JSON response.
_RESERVED_SEARXNG_PARAMS = frozenset({"q", "format"})

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

# Pooled client, shared by every search in the process so connections survive
# between requests. Created lazily and replaced when closed; no lock is needed
# because there is no await between the check and the assignment, so two
# coroutines cannot interleave and build two clients.
_client: httpx.AsyncClient | None = None


def get_search_client() -> httpx.AsyncClient:
    """The process-wide pooled client search requests are dispatched on.

    Timeouts are per tool, so they are passed on each request rather than baked
    into the client.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient()
    return _client


async def close_search_client() -> None:
    """Close the pooled client. A no-op when no search was ever dispatched."""
    global _client
    client, _client = _client, None
    if client is not None and not client.is_closed:
        await client.aclose()


class SearchToolError(ValueError):
    """The requested search tool is unknown, ambiguous, or not configured."""


class SearchProviderError(RuntimeError):
    """The search provider could not be reached or returned malformed data."""


@dataclass(frozen=True)
class SearchTool:
    """A resolved ``search_tools`` entry, ready to dispatch against."""

    name: str
    provider: str
    api_base: str
    timeout_s: float
    # None for a keyless backend, which is the usual self-hosted SearXNG case.
    api_key: str | None = None
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
    api_base = str(entry.get("api_base") or _default_api_base(config, provider) or "").strip().rstrip("/")
    # Defense in depth: startup validation already guarantees all three, so this
    # only fires for a config built in-process. Refusing here beats calling an
    # unknown provider, calling a keyed one unauthenticated, or calling a backend
    # whose address nothing supplied.
    missing_key = provider in SEARCH_PROVIDERS_REQUIRING_API_KEY and not api_key
    if provider not in SEARCH_PROVIDERS or missing_key or not api_base:
        msg = f"Search tool '{name}' is not configured correctly."
        raise SearchToolError(msg)

    options = entry.get("options")
    return SearchTool(
        name=name,
        provider=provider,
        api_key=str(api_key) if api_key else None,
        api_base=api_base,
        timeout_s=float(entry.get("timeout") or _DEFAULT_TIMEOUT_S),
        options=dict(options) if isinstance(options, dict) else {},
    )


def _default_api_base(config: GatewayConfig, provider: str) -> str | None:
    """The base URL a tool inherits when it declares no ``api_base``.

    A ``searxng`` tool falls back to ``web_search_url``, the backend the in-loop
    ``otari_web_search`` tool already speaks to over the same contract, so a
    deployment that runs one exposes it on ``POST /v1/search`` with a single
    ``provider: searxng`` line.
    """
    if provider == SEARXNG_PROVIDER:
        return (config.web_search_url or "").strip() or None
    return _DEFAULT_API_BASE.get(provider)


async def run_search(tool: SearchTool, query: SearchQuery) -> SearchOutcome:
    """Dispatch one search against the tool's provider."""
    if tool.provider == EXA_PROVIDER:
        return await _search_exa(tool, query)
    if tool.provider == SEARXNG_PROVIDER:
        return await _search_searxng(tool, query)
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
    # ``snippet`` is that text, so contents are requested by default. A tool that
    # pins its own ``contents.text`` keeps it unless the caller asked for a
    # specific per-page size. Pinning ``text: null`` (or ``false``) opts out
    # entirely: page content is what Exa charges per page for, and a caller that
    # wants only ranked URLs, or highlights, should not have to pay for it. The
    # opt-out is the operator's, so it holds even when the request carries a
    # ``max_tokens_per_page``.
    contents = dict(payload.get("contents") or {})
    pinned = contents.get("text")
    if "text" in contents and (pinned is None or pinned is False):
        del contents["text"]
    elif query.max_tokens_per_page is not None or "text" not in contents:
        max_tokens = query.max_tokens_per_page or DEFAULT_MAX_TOKENS_PER_PAGE
        max_chars = max(1, min(max_tokens * _CHARS_PER_TOKEN, _EXA_MAX_CHARACTERS))
        # Merge rather than replace: ``text`` has siblings (``verbosity``,
        # ``includeHtmlTags``) that a tool may have pinned, and only the size is
        # the caller's to set.
        text_options = dict(pinned) if isinstance(pinned, dict) else {}
        text_options["maxCharacters"] = max_chars
        contents["text"] = text_options
    # An opted-out tool with nothing else pinned wants no ``contents`` block at
    # all, rather than an empty one Exa would have to interpret.
    if contents:
        payload["contents"] = contents
    else:
        payload.pop("contents", None)

    payload["query"] = query.query
    return payload


async def _search_exa(tool: SearchTool, query: SearchQuery) -> SearchOutcome:
    if not tool.api_key:
        # Unreachable via config: exa is in SEARCH_PROVIDERS_REQUIRING_API_KEY, so
        # both startup validation and tool resolution refuse a keyless entry.
        msg = "exa search requires an api_key"
        raise SearchProviderError(msg)

    payload = build_exa_payload(tool, query)
    client = get_search_client()
    try:
        response = await client.post(
            f"{tool.api_base}/search",
            json=payload,
            headers={"x-api-key": tool.api_key},
            timeout=tool.timeout_s,
        )
    except httpx.HTTPError as exc:
        msg = f"exa search request failed: {exc}"
        raise SearchProviderError(msg) from exc

    body = _json_object(EXA_PROVIDER, response)
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


# --------------------------------------------------------------------------- #
# SearXNG
# --------------------------------------------------------------------------- #


def build_searxng_params(tool: SearchTool, query: SearchQuery) -> dict[str, str | int | float]:
    """Build the SearXNG ``GET /search`` query params from the neutral request.

    The tool's ``options`` are the base, so an operator can pin backend-native
    knobs (``engines``, ``language``, ``time_range``, or whatever a commercial
    API's fronting adapter reads) the same way ``provider_options`` does on the
    in-loop path. These ride in a query string, so only scalars are forwarded
    (bools as lowercase ``true`` / ``false``); ``q`` and ``format`` are the
    gateway's, so no option can displace the caller's query.

    ``country`` is forwarded for an adapter that can localize; a plain SearXNG
    ignores an unknown param. The remaining request fields have no SearXNG
    equivalent and are honored after the response instead: ``max_results`` and
    ``search_domain_filter`` are applied to the hits, and
    ``max_tokens_per_page`` does not apply because a SearXNG result carries the
    engine's snippet rather than fetched page content.
    """
    params: dict[str, str | int | float] = {}
    for key, value in tool.options.items():
        if key in _RESERVED_SEARXNG_PARAMS or value is None:
            continue
        if isinstance(value, bool):
            params[key] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            params[key] = value

    if query.country:
        params["country"] = query.country

    params["q"] = query.query
    params["format"] = "json"
    return params


async def _search_searxng(tool: SearchTool, query: SearchQuery) -> SearchOutcome:
    client = get_search_client()
    # A self-hosted SearXNG needs no credential and ignores the header. A tool
    # that does carry an api_key is fronting something that authenticates the
    # gateway, and gets the header the in-loop backend already sends.
    headers = {"X-Gateway-Token": tool.api_key} if tool.api_key else None
    try:
        response = await client.get(
            f"{tool.api_base}/search",
            params=build_searxng_params(tool, query),
            headers=headers,
            timeout=tool.timeout_s,
        )
    except httpx.HTTPError as exc:
        msg = f"searxng search request failed: {exc}"
        raise SearchProviderError(msg) from exc

    body = _json_object(SEARXNG_PROVIDER, response)
    hits = _apply_domain_filter(_searxng_hits(body), query.domain_filter)
    limit = max(1, min(query.max_results or DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP))
    # SearXNG reports no cost, so the tool's configured flat rate is what bills.
    return SearchOutcome(results=hits[:limit], cost_usd=None)


def _searxng_hits(body: dict[str, Any]) -> list[SearchHit]:
    raw = body.get("results")
    if not isinstance(raw, list):
        msg = "searxng search response has no 'results' list"
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
                # ``extracted_content`` is the page text an adapter already had,
                # the same optional field the in-loop backend prefers; otherwise
                # the engine's own snippet.
                snippet=_text_or_none(item.get("extracted_content")) or _text_or_none(item.get("content")),
                # ``published_date`` is the adapter convention (the Brave one
                # sets it); ``publishedDate`` is what SearXNG's own news engines
                # return.
                date=_text_or_none(item.get("published_date")) or _text_or_none(item.get("publishedDate")),
            )
        )
    return hits


def _apply_domain_filter(hits: list[SearchHit], domain_filter: tuple[str, ...]) -> list[SearchHit]:
    """Filter hits by host for a provider whose API has no domain filter.

    Exa takes include and exclude lists in the request, so its adapter passes
    them upstream; SearXNG has no equivalent, and dropping the filter would
    silently widen the search the caller asked for. Perplexity's convention
    holds either way: a leading '-' excludes rather than restricts.
    """
    include = tuple(d for d in (_normalize_domain(d) for d in domain_filter if not d.startswith("-")) if d)
    exclude = tuple(d for d in (_normalize_domain(d[1:]) for d in domain_filter if d.startswith("-")) if d)
    if not include and not exclude:
        return hits

    kept: list[SearchHit] = []
    for hit in hits:
        host = (urlparse(hit.url).hostname or "").lower()
        if exclude and _host_matches(host, exclude):
            continue
        if include and not _host_matches(host, include):
            continue
        kept.append(hit)
    return kept


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower().lstrip(".")


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    """Whether ``host`` is one of ``domains`` or a subdomain of one."""
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #


def _json_object(provider: str, response: httpx.Response) -> dict[str, Any]:
    """The provider's JSON body, or a :class:`SearchProviderError` explaining why not.

    The upstream status and body stay in the message, which reaches the usage log
    and the gateway log; the route never puts it in the response.
    """
    if response.status_code >= 400:
        msg = f"{provider} search returned HTTP {response.status_code}: {response.text[:_ERROR_BODY_CHARS]}"
        raise SearchProviderError(msg)

    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{provider} search returned a body that is not JSON"
        raise SearchProviderError(msg) from exc
    if not isinstance(body, dict):
        msg = f"{provider} search returned a {type(body).__name__} body, expected an object"
        raise SearchProviderError(msg)
    return body


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _highlights(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    joined = " ".join(part.strip() for part in value if isinstance(part, str) and part.strip())
    return joined or None
