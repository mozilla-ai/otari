"""How a web search's hits and a fetch's URL are shown to a caller, in every dialect that reports them.

A hit's title and date are whatever a provider sent, so each dialect bounds and
collapses them the same way before they go on the wire. A hit is any-search's
``SearchHit``, whose date is shown as an ISO date. A fetch's URL loses its
query string and fragment, which a fetch result never displays.
"""

from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from typing import Any

from any_search import SearchHit
from gateway.services.web_retrieval_network import truncate_utf8
from gateway.services.web_retrieval_policy import (
    MAX_WEB_URL_LENGTH,
    CanonicalWebURL,
    WebURLValidationError,
    canonicalize_web_url,
)

# Backend-supplied titles are unbounded. Generous for a real page title, and small
# enough that a full result set of them cannot approach the tool-result byte limit
# on its own.
_HIT_TITLE_MAX_BYTES = 512


def hit_title(title: object, url: str) -> str:
    """A hit's title, capped, or its URL when it has none."""
    return truncate_utf8(str(title or url).strip(), _HIT_TITLE_MAX_BYTES, suffix="…").text


def published_date(value: datetime | None) -> str | None:
    """A hit's publication time as an ISO date, or date-time when it has a time of day; ``None`` without one."""
    if value is None:
        return None
    if (value.hour, value.minute, value.second, value.microsecond) == (0, 0, 0, 0):
        return value.date().isoformat()
    return value.isoformat()


def search_query(arguments: Mapping[str, Any]) -> str:
    """The query a search call asked for, empty when it named none."""
    return str(arguments.get("query") or "")


def hits_to_show(results: Iterable[SearchHit]) -> Iterator[tuple[str, str, str | None]]:
    """Each hit's URL, title and date as a caller is shown them; a hit without a URL has nothing to show."""
    for result in results:
        url = result.url.strip()
        if url:
            yield url, hit_title(result.title, url), published_date(result.published)


def take_last_hits(pool: object) -> list[SearchHit] | None:
    """The hits the search backend that ran a call kept, taking them.

    ``None`` for a backend that keeps none, which is how a gateway-run search is
    told apart from an MCP server's tool of the same name.
    """
    take = getattr(pool, "take_last_hits", None)
    return take() if take is not None else None


def fetched_url(value: str) -> CanonicalWebURL | None:
    """A fetched page's URL in canonical form, ``None`` when it does not parse.

    Not held to the request's length limit again: the URL was accepted, and
    percent-encoding can lengthen its canonical form past that limit.
    """
    try:
        return canonicalize_web_url(value, max_length=max(MAX_WEB_URL_LENGTH, len(value)))
    except WebURLValidationError:
        return None


def fetched_display_url(value: str) -> str:
    """A fetched page's URL without its query or fragment, empty when it does not parse."""
    canonical = fetched_url(value)
    return canonical.display_url if canonical is not None else ""


def fetch_display_url(value: object) -> str:
    """The URL without its query or fragment, which a fetch result never displays; empty when unparseable."""
    try:
        return canonicalize_web_url(str(value or "")).display_url
    except WebURLValidationError:
        return ""
