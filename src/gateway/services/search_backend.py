"""The direct search endpoint's old clients, for what does not run on any-search yet.

``POST /api/v1/search`` runs on any-search (``gateway.services.tools``). Two
calls still run here until their adapters exist:

- a ``searxng`` search tool, which speaks the SearXNG-shaped
  ``GET {api_base}/search?format=json`` contract that the in-loop backend
  already dispatches to, until any-search's SearXNG adapter. A tool that
  declares no ``api_base`` inherits ``web_search_url``, and with no ``engines``
  option of its own ``web_search_engines``, as the resolver gives it::

      search_tools:
        local:
          provider: searxng   # api_base defaults to web_search_url

- a workspace's Tavily or Brave organization key (#2015), through the provider
  mapping the in-loop backend uses, until any-search's Tavily and Brave adapters.

Both read the request in the gateway's provider-neutral terms
(``SearchQuery``) and send through the client their caller passes, the app's
web provider client. They keep their own ceiling of 20 results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from gateway.core.settings.tools import validate_search_tool_transport
from gateway.services.tools import (
    DIRECT_DEFAULT_MAX_RESULTS,
    ResolvedSearch,
    SearchQuery,
    filter_hits_by_domain,
    page_budget_chars,
)
from gateway.services.web_search_providers import WebSearchProviderError, provider_search

if TYPE_CHECKING:
    from gateway.models.tools import WebSearchCredential

SEARXNG_PROVIDER = "searxng"

# Query params the gateway owns on a SearXNG request, so a tool's ``options``
# cannot displace the caller's query, ask for a non-JSON response, or widen the
# result count past what the request asked for and this client caps.
_RESERVED_SEARXNG_PARAMS = frozenset({"q", "format", "max_results"})

# The most results these clients return, until any-search's adapters replace
# them (decision 65): Tavily and Brave stop at 20 anyway.
MAX_RESULTS_CAP = 20

# How much of an upstream error body is kept for the usage log's error_message.
# The full body can be large and is never returned to the caller.
_ERROR_BODY_CHARS = 500


class SearchProviderError(RuntimeError):
    """The search provider could not be reached or returned malformed data.

    ``tag`` names the failure in the search library's terms (``timeout``,
    ``network``, ``http_error``, ``invalid_response``), and ``status`` is the
    provider's HTTP status when it answered with one, for a caller that reports
    the failure without the message, which can carry the upstream body.
    """

    def __init__(self, message: str, *, tag: str = "provider_error", status: int | None = None) -> None:
        """Carry the message for the log, and the tag and status for a caller that reports only those."""
        super().__init__(message)
        self.tag = tag
        self.status = status


def _transport_tag(exc: httpx.HTTPError | httpx.InvalidURL) -> str:
    """The tag for a request that never got an answer: it timed out, or it could not be sent."""
    return "timeout" if isinstance(exc, httpx.TimeoutException) else "network"


@dataclass(frozen=True)
class SearchTool:
    """A ``searxng`` search tool, ready to dispatch against."""

    name: str
    provider: str
    api_base: str
    timeout_s: float
    # None for a keyless backend, which is the usual self-hosted SearXNG case.
    api_key: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


def search_tool(resolved: ResolvedSearch) -> SearchTool:
    """The old client's tool for a search the resolver gave to the old code."""
    return SearchTool(
        name=resolved.name or resolved.provider,
        provider=resolved.provider,
        api_base=resolved.api_base,
        timeout_s=resolved.timeout,
        api_key=resolved.api_key or None,
        options=dict(resolved.options),
    )


@dataclass(frozen=True)
class SearchHit:
    """One normalized result."""

    url: str
    title: str | None = None
    snippet: str | None = None
    date: str | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """A completed provider search. Neither client's provider reports a cost, so ``cost_usd`` stays ``None``."""

    results: list[SearchHit]
    cost_usd: float | None = None


async def run_search(tool: SearchTool, query: SearchQuery, *, client: httpx.AsyncClient) -> SearchOutcome:
    """Dispatch one search against a ``searxng`` tool."""
    try:
        validate_search_tool_transport(tool.name, tool.api_base, tool.api_key)
    except ValueError as exc:
        # Defense in depth for a caller that constructs SearchTool directly
        # instead of through the resolver. No credential reaches the client.
        raise SearchProviderError(str(exc)) from None
    if tool.provider == SEARXNG_PROVIDER:
        return await _search_searxng(tool, query, client)
    # Unreachable through the resolver, which gives this client only searxng tools.
    msg = f"Unsupported search provider '{tool.provider}'."
    raise SearchProviderError(msg)


async def run_keyed_search(
    credential: WebSearchCredential, query: SearchQuery, *, timeout_s: float, client: httpx.AsyncClient
) -> SearchOutcome:
    """Dispatch one search on a workspace's own provider key, in place of the named tool's backend.

    ``timeout_s`` is the named tool's, so the key changes who pays and not how long a search may take.
    The provider bills the key's owner directly, so no cost is reported.
    """
    limit = _result_limit(query)
    options: dict[str, Any] = {"max_results": limit}
    if query.country:
        options["country"] = query.country
    try:
        raw = await provider_search(
            provider=credential.provider,
            api_key=credential.api_key or "",
            query=query.query,
            options=options,
            client=client,
            timeout_s=timeout_s,
        )
    except WebSearchProviderError as exc:
        msg = f"{credential.provider} search request failed: {exc}"
        raise SearchProviderError(msg, tag=exc.tag, status=exc.status) from exc
    except ValueError as exc:
        msg = f"{credential.provider} search request failed: {exc}"
        raise SearchProviderError(msg) from exc
    # The hits come back SearXNG-shaped, so they are read and capped like a SearXNG answer.
    hits = filter_hits_by_domain(_searxng_hits({"results": raw}, page_budget_chars(query)), query.domain_filter)
    return SearchOutcome(results=hits[:limit], cost_usd=None)


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

    ``max_results`` and ``country`` are forwarded for a backend that reads them
    (the bundled Tavily adapter honors ``max_results``, and would otherwise cap
    the page at its own default of 5 whatever the caller asked for); a plain
    SearXNG ignores an unknown param. ``search_domain_filter`` has no equivalent
    on either, so it is applied to the hits after the response, as is
    ``max_tokens_per_page``, since an adapter fronting a commercial API can
    return a whole page in ``extracted_content`` rather than the engine's short
    snippet.
    """
    params: dict[str, str | int | float] = {}
    for key, value in tool.options.items():
        if key in _RESERVED_SEARXNG_PARAMS or value is None:
            continue
        if isinstance(value, bool):
            params[key] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            params[key] = value

    # The same bound the hits are sliced to, so a backend that reads it is asked
    # for what the caller will actually be given and no more.
    params["max_results"] = _result_limit(query)
    if query.country:
        params["country"] = query.country

    params["q"] = query.query
    params["format"] = "json"
    return params


def _result_limit(query: SearchQuery) -> int:
    """How many hits a search returns, under this client's own ceiling."""
    return max(1, min(query.max_results or DIRECT_DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP))


async def _search_searxng(tool: SearchTool, query: SearchQuery, client: httpx.AsyncClient) -> SearchOutcome:
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
    # InvalidURL is not an HTTPError: a base URL whose port does not parse, say.
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        msg = f"searxng search request failed: {exc}"
        raise SearchProviderError(msg, tag=_transport_tag(exc)) from exc

    body = _json_object(SEARXNG_PROVIDER, response)
    # A backend that fronts a commercial API can put a whole extracted page in
    # ``extracted_content`` (the bundled Tavily adapter always does), so the
    # caller's per-page cap is applied here rather than dropped.
    hits = filter_hits_by_domain(_searxng_hits(body, page_budget_chars(query)), query.domain_filter)
    # Sliced again on this side: a plain SearXNG ignores the count param, and the
    # domain filter above can only have been applied here.
    # SearXNG reports no cost, so the tool's configured flat rate is what bills.
    return SearchOutcome(results=hits[: _result_limit(query)], cost_usd=None)


def _searxng_hits(body: dict[str, Any], max_chars: int) -> list[SearchHit]:
    raw = body.get("results")
    if not isinstance(raw, list):
        msg = "searxng search response has no 'results' list"
        raise SearchProviderError(msg, tag="invalid_response")

    hits: list[SearchHit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        # ``extracted_content`` is the page text an adapter already had, the same
        # optional field the in-loop backend prefers; otherwise the engine's own
        # snippet. Bounded by the caller's per-page budget, because the former can
        # be a whole page and nothing upstream applied the cap.
        snippet = _text_or_none(item.get("extracted_content")) or _text_or_none(item.get("content"))
        hits.append(
            SearchHit(
                url=url,
                title=_text_or_none(item.get("title")),
                snippet=snippet[:max_chars] if snippet else None,
                # ``published_date`` is the adapter convention (the Brave one
                # sets it); ``publishedDate`` is what SearXNG's own news engines
                # return.
                date=_text_or_none(item.get("published_date")) or _text_or_none(item.get("publishedDate")),
            )
        )
    return hits


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
        raise SearchProviderError(msg, tag="http_error", status=response.status_code)

    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{provider} search returned a body that is not JSON"
        raise SearchProviderError(msg, tag="invalid_response") from exc
    if not isinstance(body, dict):
        msg = f"{provider} search returned a {type(body).__name__} body, expected an object"
        raise SearchProviderError(msg, tag="invalid_response")
    return body


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
