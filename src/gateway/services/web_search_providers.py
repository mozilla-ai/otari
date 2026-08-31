"""Commercial search APIs as first-party web-search backends.

:mod:`gateway.services.web_search_backend` dispatches the in-loop
``otari_web_search`` tool to a SearXNG-shaped backend. Reaching a licensed API
(Tavily, Brave) that way used to mean running a translating adapter container
next to the gateway, one per provider, each holding the provider's key. This
module is that translation, in-process: one function per provider, mapping a
query onto the provider's native request and its answer back onto the
SearXNG-shaped hits ``WebSearchBackend`` already understands.

Two callers, and the difference between them is where the credential may sit:

* **Standalone.** ``WebSearchBackend`` calls :func:`provider_search` directly
  when ``web_search_provider`` is configured, so a self-hosted deployment
  reaches Tavily or Brave with no adapter container, no second URL, and no
  extra hop.
* **Hosted.** The data plane is a separate process on separate hardware, and a
  deployment-owned search key must not be on it. There the control plane serves
  :mod:`gateway.api.routes.web_search_backend` over the same SearXNG shape, calls
  this module on the gateway's behalf, and the key stays where it was configured.

The key is the deployment's, never a workspace's: nothing here reads tenancy,
and no response carries a credential. ``options`` is the opaque
``provider_options`` bag a workspace or a request may set, and each provider
whitelists the keys it understands rather than forwarding the bag upstream, so
an unrecognized key is dropped instead of becoming a provider request field
nobody chose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from gateway.core.config import WEB_SEARCH_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Mapping

TAVILY_PROVIDER = "tavily"
BRAVE_PROVIDER = "brave"

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Brave documents 20 as the ceiling on ``count``; asking for more is an error
# rather than a clamp, so the request is clamped here.
_BRAVE_MAX_COUNT = 20
_BRAVE_DEFAULT_COUNT = 10

# Brave spells recency as ``freshness``. Both the single and the plural forms of
# each window are accepted, matching Tavily's own ``time_range`` vocabulary, so
# one workspace's ``provider_options`` reads the same whichever provider serves
# it.
_BRAVE_FRESHNESS = {
    "d": "pd",
    "day": "pd",
    "w": "pw",
    "week": "pw",
    "m": "pm",
    "month": "pm",
    "y": "py",
    "year": "py",
}

_DEFAULT_TIMEOUT_S = 15.0

# Tavily request fields a ``provider_options`` bag may set. Anything else is
# dropped: the bag is opaque to the gateway and reaches here unvalidated, so
# forwarding it wholesale would let a workspace set Tavily request fields the
# deployment never chose. Tavily takes ``time_range`` by that name; Brave spells
# the same knob ``freshness`` and is mapped in :data:`_BRAVE_FRESHNESS`.
_TAVILY_OPTION_KEYS = ("max_results", "search_depth", "topic", "time_range", "include_answer")


class WebSearchProviderError(RuntimeError):
    """The search provider could not be reached or returned malformed data."""


async def provider_search(
    *,
    provider: str,
    api_key: str,
    query: str,
    options: Mapping[str, Any] | None = None,
    client: httpx.AsyncClient,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Search ``provider`` and return SearXNG-shaped hits.

    Each hit carries ``url``, ``title`` and ``content``, and optionally
    ``extracted_content`` when the provider already returns the page text, which
    is what lets ``WebSearchBackend`` skip its own fetch-and-extract pass.

    Raises:
        WebSearchProviderError: If the provider refused the request, could not
            be reached, or answered a shape this cannot read.
        ValueError: If ``provider`` is not one of :data:`WEB_SEARCH_PROVIDERS`.

    """
    normalized = (provider or "").strip().lower()
    if normalized == TAVILY_PROVIDER:
        return await _search_tavily(api_key, query, options or {}, client, timeout_s)
    if normalized == BRAVE_PROVIDER:
        return await _search_brave(api_key, query, options or {}, client, timeout_s)
    msg = f"web_search_provider must be one of {sorted(WEB_SEARCH_PROVIDERS)}, got '{provider}'"
    raise ValueError(msg)


async def _search_tavily(
    api_key: str,
    query: str,
    options: Mapping[str, Any],
    client: httpx.AsyncClient,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Tavily's ``/search``, which returns extracted page text alongside snippets."""
    body: dict[str, Any] = {"query": query, "include_raw_content": True}
    for key in _TAVILY_OPTION_KEYS:
        value = options.get(key)
        if value is not None:
            body[key] = value

    payload = await _request(
        TAVILY_PROVIDER,
        client,
        "POST",
        _TAVILY_ENDPOINT,
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout_s=timeout_s,
    )

    hits = payload.get("results")
    if not isinstance(hits, list):
        # A missing or non-list ``results`` is an upstream contract break, not
        # "no hits": returning an empty list would report a broken provider as a
        # search that found nothing.
        msg = "tavily search returned no results list"
        raise WebSearchProviderError(msg)

    results: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict) or not hit.get("url"):
            continue
        result: dict[str, Any] = {
            "url": str(hit["url"]),
            "title": str(hit.get("title", "")),
            "content": str(hit.get("content", "")),
        }
        raw_content = hit.get("raw_content")
        if raw_content:
            result["extracted_content"] = str(raw_content)
        results.append(result)
    return results


async def _search_brave(
    api_key: str,
    query: str,
    options: Mapping[str, Any],
    client: httpx.AsyncClient,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Brave's web search, which returns snippets only.

    ``extracted_content`` is deliberately left unset, so the caller fetches and
    extracts each page itself exactly as it does behind SearXNG.
    """
    count = _BRAVE_DEFAULT_COUNT
    max_results = options.get("max_results")
    if isinstance(max_results, int) and max_results > 0:
        count = min(max_results, _BRAVE_MAX_COUNT)

    params: dict[str, Any] = {"q": query, "count": count}
    freshness = _BRAVE_FRESHNESS.get(str(options.get("time_range") or "").strip().lower())
    if freshness is not None:
        params["freshness"] = freshness

    payload = await _request(
        BRAVE_PROVIDER,
        client,
        "GET",
        _BRAVE_ENDPOINT,
        params=params,
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout_s=timeout_s,
    )

    web = payload.get("web")
    hits = web.get("results") if isinstance(web, dict) else []
    if not isinstance(hits, list):
        msg = "brave search returned a results field that is not a list"
        raise WebSearchProviderError(msg)

    results: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict) or not hit.get("url"):
            continue
        result: dict[str, Any] = {
            "url": str(hit["url"]),
            "title": str(hit.get("title", "")),
            "content": str(hit.get("description", "")),
        }
        # ``page_age`` is an ISO 8601 timestamp and ``age`` is human-readable
        # ("3 days ago"); they are different formats rather than alternatives, so
        # prefer the parseable one and fall back only where Brave sent none.
        # Forwarded as the opaque string it arrived as, which is what
        # ``_format_results_for_model`` renders as a result's recency.
        published_date = hit.get("page_age") or hit.get("age")
        if published_date:
            result["published_date"] = str(published_date)
        results.append(result)
    return results


async def _request(
    provider: str,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    timeout_s: float,
    headers: dict[str, str],
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One provider call, with every failure collapsed onto one error type.

    The message names the provider and the upstream status and stops there. It
    is raised through ``WebSearchNotReachableError`` into the request log and an
    OTel span, and a provider's error body can carry back what was sent to it,
    so it is one of the raw bodies the security guidance keeps out of logs. The
    status is what identifies the fault: 401 is the key, 429 is the quota.
    """
    try:
        response = await client.request(
            method,
            url,
            json=json,
            params=params,
            headers=headers,
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        msg = f"{provider} search could not be reached: {exc}"
        raise WebSearchProviderError(msg) from exc

    if response.status_code >= httpx.codes.BAD_REQUEST:
        msg = f"{provider} search returned HTTP {response.status_code}"
        raise WebSearchProviderError(msg)

    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{provider} search returned a body that is not JSON"
        raise WebSearchProviderError(msg) from exc
    if not isinstance(body, dict):
        msg = f"{provider} search returned a {type(body).__name__} body, expected an object"
        raise WebSearchProviderError(msg)
    return body
