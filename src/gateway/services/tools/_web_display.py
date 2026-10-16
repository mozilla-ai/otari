"""How a web search's hits and a fetch's URL are shown to a caller, in every dialect that reports them.

A hit's title and date are whatever a provider sent, so each dialect bounds and
collapses them the same way before they go on the wire. A fetch's URL loses its
query string and fragment, which a fetch result never displays.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from gateway.services.web_retrieval_network import truncate_utf8
from gateway.services.web_retrieval_policy import WebURLValidationError, canonicalize_web_url

# A search result's recency, collapsed to one line and length-capped before it goes
# on the wire. Same rendering hygiene the text formatter applies: the value is
# whatever a search-API-fronting adapter forwarded, so one overlong or multiline
# entry shouldn't be what makes a citations panel unreadable.
_HIT_DATE_MAX_CHARS = 128

# Backend-supplied titles are unbounded. Generous for a real page title, and small
# enough that a full result set of them cannot approach the tool-result byte limit
# on its own.
_HIT_TITLE_MAX_BYTES = 512


def hit_title(title: object, url: str) -> str:
    """A hit's title, capped, or its URL when it has none."""
    return truncate_utf8(str(title or url).strip(), _HIT_TITLE_MAX_BYTES, suffix="…").text


def hit_date(value: object) -> str | None:
    """A hit's date on one line and capped, ``None`` when it has none."""
    return " ".join(str(value or "").split())[:_HIT_DATE_MAX_CHARS] or None


def search_query(arguments: Mapping[str, Any]) -> str:
    """The query a search call asked for, empty when it named none."""
    return str(arguments.get("query") or "")


def hits_to_show(results: Iterable[Mapping[str, Any]]) -> Iterator[tuple[str, str, str | None]]:
    """Each hit's URL, title and date as a caller is shown them; a hit without a URL has nothing to show."""
    for result in results:
        url = str(result.get("url") or "").strip()
        if url:
            yield url, hit_title(result.get("title"), url), hit_date(result.get("published_date"))


def fetch_display_url(value: object) -> str:
    """The URL without its query or fragment, which a fetch result never displays; empty when unparseable."""
    try:
        return canonicalize_web_url(str(value or "")).display_url
    except WebURLValidationError:
        return ""
