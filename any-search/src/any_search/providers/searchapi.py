"""SearchApi: Google results from ``GET /api/v1/search?engine=google``, with the key as a Bearer token.

The query travels in the URL, as ``q``, beside ``engine``, ``num`` and the
native options, which are SearchApi's own parameters for its Google engine.
The adapter passes them to ``self._http`` as ``params`` and never hands the
URL to a logger; httpx logs it at INFO, which the log filter
(``install_log_filter``) redacts during a provider call. The key is a header,
so it never reaches the URL.

It sends ``num``, from ``max_results`` and at most 10. SearchApi's
documentation says Google phased ``num`` out in September 2025 and a page
holds 10 results; checked live on 2026-11-23, a ``num`` below 10 was still
honored (``num=3`` answered a page of 3 and paged by 3) and a larger one was
answered as 10. The hits are cut to the count as well, so a page that ignores
``num`` answers the same. ``time_range`` becomes ``time_period``, ``last_day``
to ``last_year``, and wins over the ``time_period`` option; it also drops the
``time_period_min`` and ``time_period_max`` options, so a request never
carries two time filters.

SearchApi signals every failure with an HTTP status and an ``error``
sentence in the body, such as ``Invalid API key.`` for a 401 and
``Unsupported value `zz` in time_period parameter.`` for a 400. The sentence
can quote the request, so it never reaches an error: the tag comes from the
status, ``invalid_request`` (400), ``invalid_api_key`` (401),
``payment_required`` (402), ``rate_limited`` (429), else ``http_error``.
Only the 400 and the 401 were seen live. A search that found nothing is a 200
with no ``organic_results``: either with the sentence ``Google didn't return
any results.`` in ``error``, or, when Google answered only other sections
such as ``top_stories``, with no ``error`` at all. Both are an empty answer,
not an error. Any other ``error`` sentence in a 200 without
``organic_results`` is returned as the result's ``error``, tagged
``provider_error``, with the sentence left in ``raw``.
``search_metadata.status`` was ``Success`` on every 200 seen;
any other status is returned as the result's ``error``, with a tag made from
the status, never SearchApi's text.

Every field read from a successful answer is checked, so a change on
SearchApi's side fails the call rather than degrading it: ``search_metadata``
and its ``status``, and, when ``organic_results`` is there, the list and each
item's ``link`` and ``title``, which SearchApi documents on every result, must
be strings; ``snippet``, which some results lack, is a string or absent.
Anything else raises ``unexpected_response``. A ``link`` that is not an http
or https URL with a host is dropped rather than failing the search, and the
count is filled from the items after it. A Google redirect
(``https://www.google.com/goto?url=...``), which SearchApi returns for some
results with a ``link_token`` beside it, usually holds a session token in
``url``, not an address: a hit with that URL would look like ``google.com``
to a host filter and fetch Google, so it is dropped like any other link a
caller cannot fetch, and unwrapped only when ``url`` decodes to an http or
https URL. Ads, the AI overview, the knowledge graph, top stories and SearchApi's other SERP
sections come in fields of their own, which the adapter does not read, so a
field SearchApi adds breaks nothing. SearchApi reports no cost, and the
adapter dates no hit: a result's ``date`` is display text in the language
``hl`` names, such as ``Nov 11, 2026`` or ``7 days ago``, so it stays in
``raw``.

API: https://www.searchapi.io/docs/google. Pricing, read 2026-11-23,
https://www.searchapi.io/pricing: monthly plans from $40 for 10,000 searches
($4 per 1,000) down to $1 per 1,000 on the largest, only answers with a 200
are billed, and a plan's searches are rate limited to 20% of the plan an
hour; a new account gets 100 searches to try it.
"""

from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._http import is_web_url
from any_search._types import OptionSpec, ProviderMetadata, SearchError, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 10
"""The most results SearchApi returns for one search: one page of Google's results."""

_TIME_PERIODS = {"day": "last_day", "week": "last_week", "month": "last_month", "year": "last_year"}
# The sentence SearchApi puts in ``error`` when Google answered no results; any other sentence is an error.
_NO_RESULTS = "Google didn't return any results."
# The options ``time_range`` replaces.
_TIME_OPTIONS = ("time_period", "time_period_min", "time_period_max")

# Native options sent as query parameters, beside ``engine``, ``q`` and ``num``.
_QUERY_OPTIONS = (
    "gl",
    "hl",
    "location",
    "device",
    "safe",
    "page",
    "time_period",
    "time_period_min",
    "time_period_max",
    "lr",
    "cr",
    "nfpr",
    "filter",
)

# The statuses SearchApi documents or was seen to send, as tags; any other is "http_error".
_STATUS_TAGS = {
    400: "invalid_request",
    401: "invalid_api_key",
    402: "payment_required",
    429: "rate_limited",
}


class SearchApiProvider(AnySearch):
    """SearchApi's Google Search API."""

    METADATA = ProviderMetadata(
        name="searchapi",
        doc_url="https://www.searchapi.io/docs/google",
        env_key="SEARCHAPI_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://www.searchapi.io",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=True,
        key_in_url=False,
        options=[
            OptionSpec(
                name="gl",
                type="string",
                description=(
                    "The country to search from, a two-letter code such as us, de or jp. SearchApi's default is us."
                ),
            ),
            OptionSpec(
                name="hl",
                type="string",
                description=(
                    "The interface language, such as en or de: the language Google renders the page "
                    "and the result dates in, not a filter on the pages found (that is lr). "
                    "SearchApi's default is en."
                ),
            ),
            OptionSpec(
                name="location",
                type="string",
                description=(
                    "The place the search is run from, as Google names it, such as Kansas City, Kansas; "
                    "the most popular match is taken."
                ),
            ),
            OptionSpec(
                name="device",
                type="string",
                enum=["desktop", "mobile", "tablet"],
                description="The device Google is searched as. SearchApi's default is desktop.",
            ),
            OptionSpec(
                name="safe",
                type="string",
                enum=["active", "blur", "off"],
                description="Google's SafeSearch level. SearchApi's default is blur.",
            ),
            OptionSpec(
                name="page",
                type="integer",
                description="The page of results, from 1; a page holds max_results results, at most 10.",
            ),
            OptionSpec(
                name="time_period",
                type="string",
                enum=[
                    "last_1_minute",
                    "last_5_minutes",
                    "last_15_minutes",
                    "last_30_minutes",
                    "last_hour",
                    "last_day",
                    "last_week",
                    "last_month",
                    "last_year",
                ],
                description=(
                    "SearchApi's own recency filter, finer than time_range below a day. time_range, when passed, wins."
                ),
            ),
            OptionSpec(
                name="time_period_min",
                type="string",
                description="Only results from this date on, as MM/DD/YYYY.",
            ),
            OptionSpec(
                name="time_period_max",
                type="string",
                description="Only results up to this date, as MM/DD/YYYY.",
            ),
            OptionSpec(
                name="lr",
                type="string",
                description="Only pages in these languages, such as lang_de or lang_it|lang_de.",
            ),
            OptionSpec(
                name="cr",
                type="string",
                description=(
                    "Only pages from this country, such as countryDE, by their domain or where they are served from."
                ),
            ),
            OptionSpec(
                name="nfpr",
                type="integer",
                description="1 leaves out the results of Google's spelling correction. SearchApi's default is 0.",
            ),
            OptionSpec(
                name="filter",
                type="integer",
                description=(
                    "0 turns off Google's duplicate content and host crowding filters. SearchApi's default is 1."
                ),
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
        params: dict[str, Any] = {"engine": "google", "q": query, "num": count}
        # An option set to null is not sent.
        params.update({name: options[name] for name in _QUERY_OPTIONS if options.get(name) is not None})
        if time_range is not None:
            # One time filter only: time_range replaces time_period and drops the min/max pair.
            for name in _TIME_OPTIONS:
                params.pop(name, None)
            params["time_period"] = _TIME_PERIODS[time_range]
        headers = {"Authorization": f"Bearer {self._api_key or ''}"}

        endpoint = f"{(self._api_base or '').rstrip('/')}/api/v1/search"
        response = await self._http.request("GET", endpoint, params=params, headers=headers)
        body = _body(response)
        metadata = body.get("search_metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("status"), str):
            raise _unexpected(response)
        if metadata["status"] != "Success":
            # Inside a 200, so the error has no HTTP status of its own.
            error = SearchError(tag=_status_tag(metadata["status"]))
            return SearchResult(provider=self.METADATA.name, hits=[], cost_source="none", error=error, raw=body)
        if "organic_results" not in body:
            # No results is a 200 without the key, with Google's no-results sentence in ``error`` or no ``error``
            # at all; any other sentence is SearchApi reporting a failure inside a 200. An ``error`` that is
            # present but not a sentence is a body SearchApi does not document.
            error_text = body.get("error")
            if "error" in body and not isinstance(error_text, str):
                raise _unexpected(response)
            if isinstance(error_text, str) and error_text != _NO_RESULTS:
                error = SearchError(tag="provider_error")
                return SearchResult(provider=self.METADATA.name, hits=[], cost_source="none", error=error, raw=body)
        # An explicit null is not a list and is rejected below like any other malformed body.
        results = body.get("organic_results", [])
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
    """Return the JSON object SearchApi answered with, or raise the error it signaled."""
    if response.status_code >= 400:
        tag = _STATUS_TAGS.get(response.status_code, "http_error")
        raise ProviderError(SearchApiProvider.METADATA.name, response.status_code, tag)
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise ProviderError(SearchApiProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _status_tag(status: str) -> str:
    """A tag for a ``search_metadata.status`` other than ``Success``, from the status when it looks like one."""
    return safe_tag(status.strip().lower().replace(" ", "_")) or "error"


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field SearchApi documents, or carries one of the wrong type."""
    return ProviderError(SearchApiProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit | None:
    """One result as a hit: ``link`` and ``title``, which every result carries, and ``snippet`` when it has one.

    ``None`` for a result whose ``link`` is not an http or https URL with a
    host, since a hit needs a URL a caller can fetch. A Google redirect is
    unwrapped to its ``url`` when that is such a URL, and dropped otherwise.
    """
    if not isinstance(item, dict):
        raise _unexpected(response)
    # ``snippet`` is absent from some results, never null: an absent one is empty, a null one is malformed.
    link, title, snippet = item.get("link"), item.get("title"), item.get("snippet", "")
    if not isinstance(link, str) or not isinstance(title, str) or not isinstance(snippet, str):
        raise _unexpected(response)
    url = _destination(link)
    if url is None:
        return None
    return SearchHit(url=url, title=title, snippet=snippet, raw=item)


def _destination(link: str) -> str | None:
    """The URL a caller can fetch for ``link``, or ``None`` when there is none.

    ``https://www.google.com/goto?url=...`` is a Google redirect: its ``url``
    is usually an opaque session token, which no caller can fetch, and
    sometimes a URL, which is the destination. Anything else is itself
    when it is an http or https URL with a host.
    """
    if not is_web_url(link):
        return None
    parts = urlsplit(link)
    if (parts.hostname or "").lower() in ("google.com", "www.google.com") and parts.path == "/goto":
        target = parse_qs(parts.query).get("url", [""])[0]
        return target if is_web_url(target) else None
    return link
