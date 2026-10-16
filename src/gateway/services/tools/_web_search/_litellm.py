"""The direct endpoint's LiteLLM fields, mapped onto the provider that serves a call (design J).

``POST /api/v1/search`` takes LiteLLM's search request. Its fields are not
any-search's shared parameters, so they are mapped per provider: the domain
filter, the country and the page budget become a provider's native options
where it has them, and otherwise the hits are filtered and cut here. The count
is the shared ``max_results``. The old clients left in
``services/search_backend.py`` read the same query and use the same helpers.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from urllib.parse import urlparse

import any_search
from gateway.schemas.tools import SearchResultItem

DIRECT_DEFAULT_MAX_RESULTS = 10
"""How many results a direct search returns when nothing names a count; the in-loop tool's is 5 (decision 11)."""

DEFAULT_MAX_TOKENS_PER_PAGE = 1024
"""The page budget of a request that names no ``max_tokens_per_page``."""

CHARS_PER_TOKEN = 4
"""Perplexity meters page content in tokens and providers in characters; the usual rough ratio."""

# The most page text Exa returns for one hit of a search.
_EXA_MAX_CHARACTERS = 10_000


@dataclass(frozen=True)
class SearchQuery:
    """One direct search in the gateway's provider-neutral terms."""

    query: str
    max_results: int | None = None
    domain_filter: tuple[str, ...] = ()
    country: str | None = None
    max_tokens_per_page: int | None = None


def page_budget_chars(query: SearchQuery) -> int:
    """The page budget in characters: the request's ``max_tokens_per_page``, else the default."""
    return max(1, (query.max_tokens_per_page or DEFAULT_MAX_TOKENS_PER_PAGE) * CHARS_PER_TOKEN)


class _Hit(Protocol):
    @property
    def url(self) -> str: ...


HitT = TypeVar("HitT", bound=_Hit)


def filter_hits_by_domain(hits: Sequence[HitT], domain_filter: Sequence[str]) -> list[HitT]:
    """The hits whose host the domain filter allows, for a provider whose API has no domain filter.

    Perplexity's convention: a leading ``-`` excludes a domain rather than
    restricting to it. A domain matches itself and its subdomains.
    """
    include = tuple(d for d in (_normalize_domain(d) for d in domain_filter if not d.startswith("-")) if d)
    exclude = tuple(d for d in (_normalize_domain(d[1:]) for d in domain_filter if d.startswith("-")) if d)
    if not include and not exclude:
        return list(hits)
    kept: list[HitT] = []
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
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


@dataclass(frozen=True)
class _Mapping:
    """How one provider takes the LiteLLM fields."""

    # The native options the request's fields become, given the call's own options.
    native: Callable[[SearchQuery, Mapping[str, Any]], dict[str, Any]]
    # The provider applies the domain filter itself.
    filters_domains: bool
    # Without a request's page budget, the provider's text is already cut upstream.
    bounds_text: bool
    # Where the provider's item carries its own date string, in order.
    date_fields: tuple[str, ...]


def _exa_options(query: SearchQuery, options: Mapping[str, Any]) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    include = [d for d in query.domain_filter if not d.startswith("-")]
    exclude = [d[1:] for d in query.domain_filter if d.startswith("-") and len(d) > 1]
    if include:
        derived["includeDomains"] = include
    if exclude:
        derived["excludeDomains"] = exclude
    if query.country:
        derived["userLocation"] = query.country
    derived["contents"] = _exa_contents(options.get("contents"), query.max_tokens_per_page)
    return derived


def _exa_contents(pinned: Any, max_tokens_per_page: int | None) -> Any:
    """Exa's ``contents``: page text up to the page budget, unless the options turn the text off.

    Contents an instance or a request pinned are kept, and only the text's size
    is the page budget's to set: by the request's budget when it names one, else
    by the default when nothing set a text size. ``text`` set to null or false
    turns the text off, and with nothing else left no contents are asked for;
    ``false`` for the whole of ``contents`` does the same. A value Exa does not
    take is sent as it is, for Exa to refuse.
    """
    if pinned is False or (pinned is not None and not isinstance(pinned, dict)):
        return pinned
    contents = dict(pinned or {})
    text = contents.get("text")
    if "text" in contents and (text is None or text is False):
        del contents["text"]
    elif max_tokens_per_page is not None or "text" not in contents:
        budget = (max_tokens_per_page or DEFAULT_MAX_TOKENS_PER_PAGE) * CHARS_PER_TOKEN
        text_options = dict(text) if isinstance(text, dict) else {}
        text_options["maxCharacters"] = max(1, min(budget, _EXA_MAX_CHARACTERS))
        contents["text"] = text_options
    return contents or False


def _none(_query: SearchQuery, _options: Mapping[str, Any]) -> dict[str, Any]:
    return {}


_MAPPINGS: dict[str, _Mapping] = {
    "exa": _Mapping(native=_exa_options, filters_domains=True, bounds_text=True, date_fields=("publishedDate",)),
}
# A provider with no native equivalents: the hits are filtered and cut here, and the country is left out.
_NO_EQUIVALENTS = _Mapping(native=_none, filters_domains=False, bounds_text=False, date_fields=())


def _mapping(provider: str) -> _Mapping:
    return _MAPPINGS.get(provider, _NO_EQUIVALENTS)


def litellm_native_options(provider: str, query: SearchQuery, options: Mapping[str, Any]) -> dict[str, Any]:
    """The provider's native options the request's LiteLLM fields become.

    They are gateway-derived, so they win over the same keys in ``options``,
    the call's layered options (design G, item 2).
    """
    return _mapping(provider).native(query, options)


def direct_count(
    provider: str, query: SearchQuery, *, layered: int | None, options: Mapping[str, Any], ceiling: int | None
) -> int:
    """How many results a direct search asks for.

    The request's ``max_results``, else the one the options layered, else an
    Exa ``numResults`` option (design G), else 10; lowered to the
    workspace's ceiling when it sets one (decision 65). The provider's adapter
    lowers it further to the provider's own maximum.
    """
    count = query.max_results or layered or _native_count(provider, options) or DIRECT_DEFAULT_MAX_RESULTS
    return min(count, ceiling) if ceiling is not None else count


def _native_count(provider: str, options: Mapping[str, Any]) -> int | None:
    value = options.get("numResults") if provider == "exa" else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def litellm_results(provider: str, query: SearchQuery, hits: Sequence[any_search.SearchHit]) -> list[SearchResultItem]:
    """The hits as LiteLLM's results.

    ``snippet`` is the hit's page text when the provider returned it, else its
    snippet, cut to the page budget: the request's for every provider, else the
    default, except where the provider already cut its text upstream. ``date``
    is the provider's own date string, else the parsed date in ISO form.
    """
    mapping = _mapping(provider)
    if not mapping.filters_domains:
        hits = filter_hits_by_domain(hits, query.domain_filter)
    limit = None if mapping.bounds_text and query.max_tokens_per_page is None else page_budget_chars(query)
    return [
        SearchResultItem(
            title=hit.title,
            url=hit.url,
            snippet=(hit.text or hit.snippet)[:limit],
            date=_date(hit, mapping.date_fields),
        )
        for hit in hits
    ]


def _date(hit: any_search.SearchHit, fields: tuple[str, ...]) -> str | None:
    for name in fields:
        value = hit.raw.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return hit.published.isoformat() if hit.published is not None else None
