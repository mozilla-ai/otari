"""Mojeek: ``GET /search``, with the key and the query in the URL.

Mojeek's Web Search API takes everything as query parameters, the key
included, so the request URL is a secret. The adapter passes them to
``self._http``, which turns a transport failure into a ``ProviderError``
without its text, and never hands the URL to a logger. httpx logs it at INFO,
and over HTTP/2 hpack logs the path with its query at DEBUG; the log filter
(``install_log_filter``) redacts both during a provider call. httpcore, between
the two, logs at DEBUG only the method, the host and the response headers.

It always sends ``fmt=json``, since Mojeek answers in XML otherwise; ``t``,
from ``max_results`` and at most 10, the most Mojeek's entry plan returns; and
``date=1``, unless the caller turns it off, so each hit carries the page's
last-modified time as its date.

Mojeek signals a refused key, or a spent daily quota, inside a 200, in
``response.status``: any status but ``OK`` is the result's ``error``, with a
tag of the adapter's own, never Mojeek's text, and no HTTP status. Other
failures are HTTP statuses with a plain body, such as a 503 for a malformed
parameter.

Every field read from a successful answer is checked, so a change on Mojeek's
side fails the call rather than degrading it: ``response`` and its ``status``
must be there and, when the status is ``OK``, ``results`` and each item's
``url``, ``title`` and ``desc``, which Mojeek documents on every result and
sends on every one. ``pdate`` and ``timestamp`` are documented on every result
too, but sent only on request, so neither is required. The date is read from
``pdate`` when it is present and not 0, else from ``timestamp``, and the field
read must be a Unix time, 0 meaning none; a ``timestamp`` behind a usable
``pdate`` is not read. Anything else raises ``unexpected_response``.
Fields the adapter does not read are ignored, so a field Mojeek adds breaks
nothing. Mojeek reports no cost.

API: https://www.mojeek.com/support/api/search/request_parameters.html and
https://www.mojeek.com/support/api/search/json_response.html. Pricing, read
2026-10-13, https://www.mojeek.com/services/search/web-search-api/: per query,
before taxes, £2 per 1,000 on the Startup plan, which returns up to 10 results
a query; £3 on Business, up to 40; Enterprise by arrangement, up to 100.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._types import OptionSpec, ProviderMetadata, SearchError, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 10
"""The most results Mojeek's entry plan returns for one search. Higher plans return more, but ask for at most this."""

# The statuses Mojeek documents or was seen to send, by how they start, as tags; any other is "error".
_STATUS_TAGS = (
    ("access denied", "access_denied"),
    ("error: daily limit reached", "daily_limit_reached"),
)


class MojeekProvider(AnySearch):
    """Mojeek's Web Search API."""

    METADATA = ProviderMetadata(
        name="mojeek",
        doc_url="https://www.mojeek.com/support/api/search/request_parameters.html",
        env_key="MOJEEK_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.mojeek.com",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=True,
        key_in_url=True,
        options=[
            OptionSpec(
                name="qm", type="string", description="Words to exclude: results that contain them are dropped."
            ),
            OptionSpec(
                name="s", type="integer", description="The number of the first result, from 1, to page through results."
            ),
            OptionSpec(name="site", type="string", description="Only results from this site, such as mojeek.com."),
            OptionSpec(
                name="since",
                type="string",
                description=(
                    "Only results from this period: day, month, year, or a date as YYYYMMDD. "
                    "time_range, when passed, wins."
                ),
            ),
            OptionSpec(
                name="before",
                type="string",
                description=(
                    "Only results dated before this, not including it: day, month, year, or a date as YYYYMMDD."
                ),
            ),
            OptionSpec(
                name="rb",
                type="string",
                description="Rank pages from this region higher: an ISO 3166-1 alpha-2 code, such as GB.",
            ),
            OptionSpec(
                name="rbb",
                type="integer",
                description="How much rb boosts its region, from 1 to 100. Mojeek recommends 10.",
            ),
            OptionSpec(
                name="reg",
                type="string",
                description=(
                    "Only pages from the UK, Germany, France or the EU, by country code. Mojeek recommends rb instead."
                ),
            ),
            OptionSpec(
                name="lb",
                type="string",
                description="Rank pages in this language higher: an ISO 639-1 code, such as EN.",
            ),
            OptionSpec(
                name="lbb",
                type="integer",
                description="How much lb boosts its language, from 1 to 100. Mojeek recommends 100.",
            ),
            OptionSpec(
                name="lr", type="string", description="Only pages in this language, an ISO 639-1 code. In beta."
            ),
            OptionSpec(
                name="clufmt",
                type="integer",
                description=(
                    "Clustering by host: 1 to 5 is the most results per host in each 10; 0 leaves that to si. "
                    "Mojeek recommends leaving it unset."
                ),
            ),
            OptionSpec(
                name="si",
                type="integer",
                description=(
                    "With clufmt 0, the most results per host in the whole result set; 0 turns clustering off. "
                    "Mojeek recommends leaving it unset."
                ),
            ),
            OptionSpec(
                name="date",
                type="boolean",
                default=True,
                description=(
                    "Return each page's last-modified time, which becomes the hit's date. Sent unless set to false."
                ),
            ),
            OptionSpec(
                name="cdate",
                type="boolean",
                default=False,
                description="Return the date each page was last crawled, in raw.",
            ),
            OptionSpec(name="size", type="boolean", default=False, description="Return each page's size, in raw."),
            OptionSpec(
                name="datewr",
                type="integer",
                default=0,
                description="Date weight ratio: 100 ranks results by date, 0 does not.",
            ),
            OptionSpec(
                name="tlen", type="integer", default=56, description="Title length in characters, from 0 to 127."
            ),
            OptionSpec(
                name="dlen", type="integer", default=160, description="Snippet length in characters, from 0 to 511."
            ),
            OptionSpec(name="safe", type="boolean", default=False, description="Safe search: leave out adult content."),
            OptionSpec(
                name="fi",
                type="string",
                description=(
                    "Only results from these domains, comma-separated, at most 25. A leading dot takes in subdomains."
                ),
            ),
            OptionSpec(
                name="fe",
                type="string",
                description=(
                    "No results from these domains, comma-separated, at most 25. A leading dot takes in subdomains."
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
        params: dict[str, Any] = {"q": query, "t": count, "fmt": "json", "date": 1}
        # Mojeek's booleans are 0 and 1, and an option set to null is not sent.
        for name, value in options.items():
            if value is None:
                params.pop(name, None)
            else:
                params[name] = int(value) if isinstance(value, bool) else value
        if time_range is not None:
            params["since"] = _since(time_range)
        params["api_key"] = self._api_key or ""

        endpoint = f"{(self._api_base or '').rstrip('/')}/search"
        response = await self._http.request("GET", endpoint, params=params)
        body = _body(response)
        answer = body.get("response")
        if not isinstance(answer, dict) or not isinstance(answer.get("status"), str):
            raise _unexpected(response)
        status = answer["status"].strip()
        if status.upper() != "OK":
            # Inside a 200, so the error has no HTTP status of its own.
            error = SearchError(tag=_status_tag(status))
            return SearchResult(provider=self.METADATA.name, hits=[], cost_source="none", error=error, raw=body)
        results = answer.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        # Only the items the caller gets are read, so only they are checked.
        hits = [_hit(item, response) for item in results[:count]]
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _since(time_range: TimeRange) -> str:
    """Mojeek's own word for a day, a month or a year; a week, which it has none for, is the date it starts on."""
    if time_range == "week":
        return (datetime.now(UTC) - timedelta(weeks=1)).strftime("%Y%m%d")
    return time_range


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Mojeek answered with, or raise the HTTP failure.

    Mojeek's failures carry a plain body and no code, so the tag is always
    ``http_error``.
    """
    if response.status_code >= 400:
        raise ProviderError(MojeekProvider.METADATA.name, response.status_code, "http_error")
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise ProviderError(MojeekProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _status_tag(status: str) -> str:
    """The tag for a status other than ``OK``, from how it starts, so none of Mojeek's text reaches an error."""
    lowered = status.lower()
    return next((tag for start, tag in _STATUS_TAGS if lowered.startswith(start)), "error")


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Mojeek documents, or carries one of the wrong type."""
    return ProviderError(MojeekProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    """One result as a hit: ``url``, ``title`` and ``desc``, which every result carries, and its date when sent."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, desc = item.get("url"), item.get("title"), item.get("desc")
    if not isinstance(url, str) or not url or not isinstance(title, str) or not isinstance(desc, str):
        raise _unexpected(response)
    return SearchHit(url=url, title=title, snippet=desc, published=_published(item, response), raw=item)


def _published(item: dict[str, Any], response: httpx.Response) -> datetime | None:
    """``pdate``, the publication time, else ``timestamp``, the last-modified one; ``None`` when neither came.

    Mojeek sends 0 for a page whose time it does not know, so 0 counts as none.
    """
    for name in ("pdate", "timestamp"):
        if name not in item:
            continue
        value = item[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _unexpected(response)
        if value == 0:
            continue
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OverflowError, OSError, ValueError):
            raise _unexpected(response) from None
    return None
