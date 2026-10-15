"""Serply: Google results from ``GET /v1/search``, with the key in the ``X-Api-Key`` header.

The query travels in the URL, as ``q``, beside ``num`` and the native options
that are Google's own parameters. The adapter passes them to ``self._http`` as
``params`` and never hands the URL to a logger; httpx logs it at INFO, which
the log filter (``install_log_filter``) redacts during a provider call. The
key is a header, so it never reaches the URL. Each request also names the
library in its ``User-Agent``, so Serply can tell its traffic from a script's.

It sends ``num``, from ``max_results`` and at most 10: Serply serves one page
of Google's results, so ``num=20`` answers with the same page as ``num=10``.
Below 10 it is honored, but a page can hold a row more than asked for, so the
hits are cut to the count as well. ``time_range`` becomes Google's own
``tbs`` window, ``qdr:d`` to ``qdr:y``, which Serply passes through to Google,
and wins over the ``tbs`` option.

Serply signals every failure with an HTTP status and a ``detail`` sentence
in the body, such as ``Invalid API key`` for a 401. The sentence never
reaches an error: the tag is the adapter's own, by how it starts, else
``http_error``. A successful search carries no error of its own.

Every field read from a successful answer is checked, so a change on Serply's
side fails the call rather than degrading it: ``results`` and each item's
``link``, ``title`` and ``description``, which Serply documents on every
result, must be there and be strings. Anything else raises
``unexpected_response``. A ``link`` that is not an http or https URL, such as
an opaque Google redirect token, which Serply sometimes returns in its place,
is dropped rather than failing the search. Ads, places, the knowledge graph
and Serply's other SERP features come in arrays of their own beside
``results``, which the adapter does not read, so a field Serply adds breaks
nothing. A result's ``metadata.published_time`` becomes ``published`` when it is a
``May 27, 2026`` date, and Serply reports no cost.

API: https://serply.io/docs/resources/google-search. ``tbs`` and ``tbm`` are
Google's own parameters, which the documentation does not list; Serply passes
them through, checked live 2026-10-15. Pricing, read 2026-10-15,
https://serply.io/pricing: prepaid credits, one per successful request that is
not served from Serply's cache, from $2.00 per 1,000 (2,500 for $5) down to
$0.75 per 1,000 in the largest packs; a new account gets 2,500 credits to use
in its first 30 days.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 10
"""The most results Serply returns for one search: one page of Google's results."""

USER_AGENT = "any-search"
"""Sent with every request, so Serply can tell the library's traffic apart."""

_TBS = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}

# Native options sent as query parameters, beside ``q`` and ``num``.
_QUERY_OPTIONS = ("gl", "start", "tbs", "tbm")

# Native options sent as request headers, under Serply's own header names.
_HEADER_OPTIONS = ("X-Proxy-Location", "X-User-Agent")

# The details Serply was seen to send, by how they start, as tags; any other is "http_error".
_DETAIL_TAGS = (
    ("invalid api key", "invalid_api_key"),
    ("api key missing", "missing_api_key"),
)


class SerplyProvider(AnySearch):
    """Serply's Google Search API."""

    METADATA = ProviderMetadata(
        name="serply",
        doc_url="https://serply.io/docs/resources/google-search",
        env_key="SERPLY_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.serply.io",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=True,
        key_in_url=False,
        options=[
            OptionSpec(
                name="gl",
                type="string",
                description=(
                    "The country to localize results for, a two-letter code such as us, de or jp. "
                    "Without it or X-Proxy-Location, results can differ between calls."
                ),
            ),
            OptionSpec(
                name="start",
                type="integer",
                default=0,
                description="The zero-based offset into the results, to page through them: 10 is the second page.",
            ),
            OptionSpec(
                name="tbs",
                type="string",
                description="Google's own time filter, such as qdr:w for the past week. time_range, when passed, wins.",
            ),
            OptionSpec(
                name="tbm",
                type="string",
                enum=["nws"],
                description="Search Google News instead of the web; its results come in the same shape.",
            ),
            OptionSpec(
                name="X-Proxy-Location",
                type="string",
                enum=["EU", "CA", "US", "IE", "GB", "FR", "DE", "SE", "IN", "JP", "KR", "SG", "AU", "BR"],
                description="The region the search is run from, sent as a header. gl reaches any country.",
            ),
            OptionSpec(
                name="X-User-Agent",
                type="string",
                enum=["desktop", "mobile"],
                description="The device Google is searched as, sent as a header.",
            ),
        ],
    )

    async def _search(
        self,
        query: str,
        *,
        max_results: int | None,
        time_range: TimeRange | None,
        options: dict[str, Any],
    ) -> SearchResult:
        count = MAX_RESULTS if max_results is None else min(max_results, MAX_RESULTS)
        params: dict[str, Any] = {"q": query, "num": count}
        # An option set to null is not sent.
        params.update({name: options[name] for name in _QUERY_OPTIONS if options.get(name) is not None})
        if time_range is not None:
            params["tbs"] = _TBS[time_range]
        headers = {name: str(options[name]) for name in _HEADER_OPTIONS if options.get(name) is not None}
        headers.update({"X-Api-Key": self._api_key or "", "User-Agent": USER_AGENT})

        endpoint = f"{(self._api_base or '').rstrip('/')}/v1/search"
        response = await self._http.request("GET", endpoint, params=params, headers=headers)
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        # Only the items read until the count is reached are checked, as only they reach the caller.
        hits: list[SearchHit] = []
        for item in results:
            if len(hits) == count:
                break
            hit = _hit(item, response)
            if hit is not None:
                hits.append(hit)
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Serply answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        raise ProviderError(SerplyProvider.METADATA.name, response.status_code, _detail_tag(detail))
    if not isinstance(body, dict):
        raise ProviderError(SerplyProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _detail_tag(detail: Any) -> str:
    """The tag for Serply's ``detail``, from how it starts, so none of Serply's text reaches an error."""
    lowered = detail.strip().lower() if isinstance(detail, str) else ""
    return next((tag for start, tag in _DETAIL_TAGS if lowered.startswith(start)), "http_error")


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Serply documents, or carries one of the wrong type."""
    return ProviderError(SerplyProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit | None:
    """One result as a hit: ``link``, ``title`` and ``description``, which every result carries.

    ``None`` for a result whose ``link`` is not an http or https URL, such as a
    Google redirect token, since a hit needs a URL a caller can fetch.
    """
    if not isinstance(item, dict):
        raise _unexpected(response)
    link, title, description = item.get("link"), item.get("title"), item.get("description")
    if not isinstance(link, str) or not isinstance(title, str) or not isinstance(description, str):
        raise _unexpected(response)
    if not link.startswith(("https://", "http://")):
        return None
    return SearchHit(url=link, title=title, snippet=description, published=_published(item), raw=item)


def _published(item: dict[str, Any]) -> datetime | None:
    """``metadata.published_time``, which Google prints as ``May 27, 2026``; ``None`` for any other form."""
    metadata = item.get("metadata")
    value = metadata.get("published_time") if isinstance(metadata, dict) else None
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%b %d, %Y").replace(tzinfo=UTC)
    except ValueError:
        return None
