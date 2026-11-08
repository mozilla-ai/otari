"""SearXNG: ``GET /search`` on an instance its operator runs, with ``format=json``.

SearXNG is a metasearch engine: it asks the engines it is configured with and
merges their answers. Every instance is someone's own, so there is no default
address and no key; ``api_base`` is the instance's root URL.

It searches with GET, SearXNG's own default, passing the query and the options
to ``self._http`` as ``params`` and never handing the URL to a logger; httpx
logs it at INFO, which the log filter (``install_log_filter``) redacts during a
provider call. The ``method`` option ``post`` sends the same fields as form
data, which keeps the query out of the instance's access log. Either
way it sends ``q``, ``format=json``, the native options under SearXNG's own
names and ``time_range`` as it is, since SearXNG takes ``day``, ``week``,
``month`` and ``year``. An engine that has no time filter is left out of a
search that sets one, so such a search can come back empty. SearXNG has no
count parameter: it answers one page of merged results, which the adapter cuts
to ``max_results``. It also returns no page text, so a hit's ``text`` is
``None``.

An HTTP status of 400 or more raises a ``ProviderError``. SearXNG answers a
parameter it refuses with a 400 and a sentence in ``error``, such as
``Invalid value abc for parameter pageno`` or ``Empty pageno parameter``, which
becomes ``invalid_parameter`` by how it starts and never reaches the error
itself. A 403 is ``forbidden``:
SearXNG answers one when ``json`` is not among the instance's
``search.formats``, and so can a proxy in front of it. Any other status from
400 is ``http_error``. An engine that fails, by a timeout, a CAPTCHA or a
suspension, is named in ``unresponsive_engines`` inside a 200. When no hit came
back and at least one engine failed, the search failed: the result's ``error``
is ``engines_unresponsive``, with no HTTP status. With hits, the failures stay
in ``raw``.

Every field read from a successful answer is checked, so a change on
SearXNG's side fails the call. SearXNG's API documentation describes no field
of its answer, so only ``results``, without which the adapter cannot answer,
is required, and must be a list. ``unresponsive_engines`` must be a list when
present, and each result's ``title`` and ``content`` strings, empty when
absent. ``publishedDate`` is optional and often ``null``; it is read as an
ISO 8601 date, UTC when it has no zone, and one in another form stays in
``raw``, since SearXNG passes on what each engine gives. A value that starts
like an ISO 8601 date and is not one, such as ``2026-13-40``, raises
``unexpected_response``, as does anything else the checks refuse. A result whose ``url`` is missing, or is not an http
or https URL with a host, such as a ``magnet:`` link from a torrent engine, is
dropped, and the search goes on. Fields the adapter does not read are ignored,
so a field SearXNG adds breaks nothing. SearXNG reports no cost.

API: https://docs.searxng.org/dev/search_api.html, read 2026-11-05 (version
2026.11.5), and the JSON response's keys from ``searx/webutils.py``. Pricing:
none; the operator runs the instance, and its engines are free to call within
their own terms and limits.
"""

import re
from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._http import is_web_url
from any_search._types import OptionSpec, ProviderMetadata, SearchError, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 100
"""The most hits the adapter returns from one search, which is one page of SearXNG's merged results."""

# Native options sent as query parameters or form fields, beside ``q`` and ``format``.
_FIELD_OPTIONS = ("engines", "categories", "language", "safesearch", "pageno")

# How an ISO 8601 date starts, to tell a broken one from a date in another form.
_ISO_DATE_START = re.compile(r"\d{4}-\d{2}-\d{2}")

# The values the ``method`` option takes.
_METHODS = ("get", "post")

# The sentences SearXNG was seen to send in a 400's ``error``, by how they start, as tags; any other is "http_error".
_ERROR_TAGS = (("invalid value", "invalid_parameter"), ("empty ", "invalid_parameter"))


class SearxngProvider(AnySearch):
    """A SearXNG instance's search API."""

    METADATA = ProviderMetadata(
        name="searxng",
        doc_url="https://docs.searxng.org/dev/search_api.html",
        env_key=None,
        env_api_base="SEARXNG_BASE_URL",
        requires_api_key=False,
        requires_api_base=True,
        default_api_base=None,
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=True,
        key_in_url=False,
        options=[
            OptionSpec(
                name="engines",
                type="string",
                description=(
                    "The engines to search, comma-separated, such as wikinews,arxiv. SearXNG ignores a name it does "
                    "not know, and with no name it knows searches its default engines. An engine that fails is "
                    "suspended for everyone the instance serves."
                ),
                operator_only=True,
            ),
            OptionSpec(
                name="method",
                type="string",
                enum=list(_METHODS),
                default=_METHODS[0],
                description=(
                    "How the search is sent: get puts the query in the URL, which the instance's access log can "
                    "record; post sends it as form data."
                ),
                operator_only=True,
            ),
            OptionSpec(
                name="categories",
                type="string",
                description="The categories to search, comma-separated, such as general, news or science.",
            ),
            OptionSpec(
                name="language",
                type="string",
                description="The language to search in, such as en, de-CH or all. The instance's setting by default.",
            ),
            OptionSpec(
                name="safesearch",
                type="integer",
                description="Filter adult content: 0 none, 1 moderate, 2 strict. The instance's setting by default.",
            ),
            OptionSpec(name="pageno", type="integer", default=1, description="The page of results, from 1."),
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
        fields: dict[str, Any] = {"q": query, "format": "json"}
        # An option set to null is not sent.
        fields.update({name: options[name] for name in _FIELD_OPTIONS if options.get(name) is not None})
        if time_range is not None:
            fields["time_range"] = time_range

        method = options.get("method")
        if method is None:
            method = _METHODS[0]
        if method not in _METHODS:
            raise ValueError(f"method must be one of {', '.join(_METHODS)}")

        endpoint = f"{(self._api_base or '').rstrip('/')}/search"
        if method == "post":
            response = await self._http.request("POST", endpoint, data=fields)
        else:
            response = await self._http.request("GET", endpoint, params=fields)
        body = _body(response)
        results, unresponsive = body.get("results"), body.get("unresponsive_engines", [])
        if not isinstance(results, list) or not isinstance(unresponsive, list):
            raise _unexpected(response)
        # Only the items read until the count is reached are checked, as only they reach the caller.
        hits: list[SearchHit] = []
        for item in results:
            if len(hits) == count:
                break
            hit = _hit(item, response)
            if hit is not None:
                hits.append(hit)
        if not hits and unresponsive:
            # Inside a 200, so the error has no HTTP status of its own.
            error = SearchError(tag="engines_unresponsive")
            return SearchResult(provider=self.METADATA.name, hits=[], cost_source="none", error=error, raw=body)
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object SearXNG answered with, or raise the HTTP failure."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        raise ProviderError(SearxngProvider.METADATA.name, response.status_code, _status_tag(response, body))
    if not isinstance(body, dict):
        raise ProviderError(SearxngProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _status_tag(response: httpx.Response, body: Any) -> str:
    """The tag for a failed answer, so none of SearXNG's text reaches an error."""
    if response.status_code == 403:
        return "forbidden"
    if response.status_code != 400:
        return "http_error"
    error = body.get("error") if isinstance(body, dict) else None
    lowered = error.strip().lower() if isinstance(error, str) else ""
    return next((tag for start, tag in _ERROR_TAGS if lowered.startswith(start)), "http_error")


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field SearXNG always sends, or carries one of the wrong type."""
    return ProviderError(SearxngProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit | None:
    """One result as a hit; ``None`` for one without an http or https ``url`` with a host."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, content = item.get("url"), item.get("title", ""), item.get("content", "")
    if not isinstance(url, str | None) or not isinstance(title, str) or not isinstance(content, str):
        raise _unexpected(response)
    if url is None or not is_web_url(url):
        return None
    return SearchHit(url=url, title=title, snippet=content, published=_published(item, response), raw=item)


def _published(item: dict[str, Any], response: httpx.Response) -> datetime | None:
    """``publishedDate`` as an aware datetime, UTC when it carries no zone.

    ``None`` when SearXNG has none, or one in another form, which stays in ``raw``.
    """
    value = item.get("publishedDate")
    if value is None:
        return None
    if not isinstance(value, str):
        raise _unexpected(response)
    try:
        published = datetime.fromisoformat(value)
    except ValueError:
        if _ISO_DATE_START.match(value):
            raise _unexpected(response) from None
        return None
    return published if published.tzinfo is not None else published.replace(tzinfo=UTC)
