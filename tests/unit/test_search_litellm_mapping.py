"""The direct endpoint's LiteLLM mapping (design J): for Exa, and for a provider without native equivalents.

The request's fields become Exa's native options, and for a provider with none
of its own (``fake`` here) they are applied to the hits instead. The cases on
Exa's options moved here from ``test_search_backend.py`` with the mapping.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from any_search import SearchHit
from gateway.services import search_backend
from gateway.services.tools import SearchQuery, filter_hits_by_domain
from gateway.services.tools._web_search._litellm import direct_count, litellm_native_options, litellm_results

_Q = SearchQuery(query="q")


def _exa(query: SearchQuery = _Q, **options: Any) -> dict[str, Any]:
    return litellm_native_options("exa", query, options)


def _hit(url: str = "https://example.com/a", **fields: Any) -> SearchHit:
    return SearchHit(url=url, raw={"url": url, **fields.pop("raw", {})}, **fields)


# --------------------------------------------------------------------------- #
# Exa
# --------------------------------------------------------------------------- #


def test_exa_asks_for_page_text_within_the_default_budget() -> None:
    assert _exa() == {"contents": {"text": {"maxCharacters": 4096}}}


def test_exa_maps_the_request_fields() -> None:
    query = SearchQuery(query="q", domain_filter=("arxiv.org", "-spam.example"), country="US", max_tokens_per_page=100)
    assert _exa(query) == {
        "includeDomains": ["arxiv.org"],
        "excludeDomains": ["spam.example"],
        "userLocation": "US",
        "contents": {"text": {"maxCharacters": 400}},
    }


def test_exa_page_size_stays_within_exas_ceiling() -> None:
    assert _exa(SearchQuery(query="q", max_tokens_per_page=100_000))["contents"]["text"]["maxCharacters"] == 10_000


def test_pinned_contents_survive_when_the_caller_asks_for_no_page_size() -> None:
    assert _exa(contents={"text": {"maxCharacters": 800}})["contents"] == {"text": {"maxCharacters": 800}}


def test_the_callers_page_size_overrides_pinned_contents() -> None:
    """The caller's size wins; 800 would survive if the override were a no-op."""
    query = SearchQuery(query="q", max_tokens_per_page=50)
    assert _exa(query, contents={"text": {"maxCharacters": 800}})["contents"]["text"]["maxCharacters"] == 200


@pytest.mark.parametrize("pinned", [None, False])
def test_page_text_can_be_turned_off(pinned: object) -> None:
    """Pinning contents.text to null or false spares the per-page content charge."""
    assert _exa(contents={"text": pinned, "highlights": True})["contents"] == {"highlights": True}


@pytest.mark.parametrize("contents", [{"text": None}, False])
def test_turning_off_everything_asks_for_no_contents(contents: object) -> None:
    """False, since any-search would otherwise ask for its default page text."""
    assert _exa(contents=contents)["contents"] is False


def test_turning_page_text_off_survives_a_callers_page_size() -> None:
    query = SearchQuery(query="q", max_tokens_per_page=256)
    assert _exa(query, contents={"text": None, "highlights": True})["contents"] == {"highlights": True}


def test_the_callers_page_size_keeps_the_other_pinned_text_options() -> None:
    """Only the size is the caller's to set; pinned siblings are merged, not replaced."""
    query = SearchQuery(query="q", max_tokens_per_page=50)
    pinned = {"text": {"maxCharacters": 800, "verbosity": "full"}, "highlights": True}
    assert _exa(query, contents=pinned)["contents"] == {
        "text": {"maxCharacters": 200, "verbosity": "full"},
        "highlights": True,
    }


def test_contents_exa_does_not_take_are_sent_for_exa_to_refuse() -> None:
    assert _exa(contents="everything")["contents"] == "everything"


# --------------------------------------------------------------------------- #
# The count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("asked", "layered", "options", "ceiling", "count"),
    [
        pytest.param(None, None, {}, None, 10, id="the-endpoints-default"),
        pytest.param(3, 7, {"numResults": 5}, None, 3, id="the-request-first"),
        pytest.param(None, 7, {"numResults": 5}, None, 7, id="then-the-layered-max-results"),
        pytest.param(None, None, {"numResults": 5}, None, 5, id="then-exas-num-results"),
        pytest.param(None, None, {"numResults": "5"}, None, 10, id="a-num-results-that-is-no-count"),
        pytest.param(500, None, {}, None, 500, id="no-ceiling-of-the-gateways-own"),
        pytest.param(50, None, {}, 30, 30, id="the-workspace-ceiling-lowers-it"),
        pytest.param(None, None, {"numResults": 50}, 30, 30, id="a-native-count-cannot-pass-the-ceiling"),
        pytest.param(5, None, {}, 30, 5, id="the-ceiling-never-raises-it"),
    ],
)
def test_the_count_on_exa(
    asked: int | None, layered: int | None, options: dict[str, Any], ceiling: int | None, count: int
) -> None:
    query = SearchQuery(query="q", max_results=asked)
    assert direct_count("exa", query, layered=layered, options=options, ceiling=ceiling) == count


def test_num_results_is_exas_alone() -> None:
    assert direct_count("fake", _Q, layered=None, options={"numResults": 5}, ceiling=None) == 10


# --------------------------------------------------------------------------- #
# A provider without native equivalents
# --------------------------------------------------------------------------- #


def test_a_provider_without_equivalents_gets_no_native_options() -> None:
    query = SearchQuery(query="q", domain_filter=("arxiv.org",), country="US", max_tokens_per_page=100)
    assert litellm_native_options("fake", query, {"contents": {"text": True}}) == {}


def test_its_hits_are_filtered_by_domain_and_cut_to_the_page_budget() -> None:
    hits = [
        _hit("https://arxiv.org/abs/1", text="x" * 5000),
        _hit("https://spam.example/2", snippet="no"),
        _hit("https://export.arxiv.org/abs/3", snippet="short"),
    ]

    default = litellm_results("fake", SearchQuery(query="q", domain_filter=("arxiv.org",)), hits)
    asked = litellm_results("fake", SearchQuery(query="q", max_tokens_per_page=10), hits)

    assert [result.url for result in default] == ["https://arxiv.org/abs/1", "https://export.arxiv.org/abs/3"]
    # The default budget of 1,024 tokens at four characters each.
    assert len(default[0].snippet) == 4096
    assert default[1].snippet == "short"
    assert [len(result.snippet) for result in asked] == [40, 2, 5]


def test_its_date_is_the_parsed_one_in_iso_form() -> None:
    (result,) = litellm_results("fake", _Q, [_hit(published=datetime(2026, 1, 2, tzinfo=UTC), raw={"date": "x"})])
    assert (result.date, result.last_updated) == ("2026-01-02T00:00:00+00:00", None)


# --------------------------------------------------------------------------- #
# Exa's hits
# --------------------------------------------------------------------------- #


def test_exa_hits_keep_their_text_uncut_unless_the_caller_names_a_budget() -> None:
    hits = [_hit(text="x" * 5000, snippet="highlight"), _hit("https://example.com/b", snippet="only highlights")]

    default = litellm_results("exa", _Q, hits)
    asked = litellm_results("exa", SearchQuery(query="q", max_tokens_per_page=10), hits)

    # Exa's maxCharacters bounded the text upstream.
    assert [len(result.snippet) for result in default] == [5000, len("only highlights")]
    assert [len(result.snippet) for result in asked] == [40, len("only highlights")]


def test_an_exa_date_is_exas_own_string() -> None:
    published = datetime(2025, 10, 1, tzinfo=UTC)
    dated = _hit(published=published, raw={"publishedDate": "2025-10-01T00:00:00.000Z"})
    parsed_only = _hit("https://example.com/b", published=published)
    undated = _hit("https://example.com/c")

    assert [result.date for result in litellm_results("exa", _Q, [dated, parsed_only, undated])] == [
        "2025-10-01T00:00:00.000Z",
        "2025-10-01T00:00:00+00:00",
        None,
    ]


def test_exa_filters_its_own_domains() -> None:
    hits = [_hit("https://spam.example/1")]
    assert len(litellm_results("exa", SearchQuery(query="q", domain_filter=("arxiv.org",)), hits)) == 1


# --------------------------------------------------------------------------- #
# The domain filter
# --------------------------------------------------------------------------- #


def test_the_domain_filter_takes_any_hit_with_a_url() -> None:
    """The library's hits and the old clients' alike."""
    old = [search_backend.SearchHit(url="https://arxiv.org/abs/1"), search_backend.SearchHit(url="https://x.example")]
    new = [_hit("https://arxiv.org/abs/1"), _hit("https://x.example")]

    assert filter_hits_by_domain(old, ("arxiv.org",)) == old[:1]
    assert filter_hits_by_domain(new, ("-x.example",)) == new[:1]


@pytest.mark.parametrize(
    ("domain_filter", "kept"),
    [
        pytest.param((), ["arxiv.org", "export.arxiv.org", "notarxiv.org"], id="no-filter"),
        pytest.param(("arxiv.org",), ["arxiv.org", "export.arxiv.org"], id="a-domain-and-its-subdomains"),
        pytest.param((" .ArXiv.org ",), ["arxiv.org", "export.arxiv.org"], id="normalized"),
        pytest.param(("-export.arxiv.org",), ["arxiv.org", "notarxiv.org"], id="a-leading-dash-excludes"),
        pytest.param(("arxiv.org", "-export.arxiv.org"), ["arxiv.org"], id="both"),
    ],
)
def test_the_domain_filter(domain_filter: tuple[str, ...], kept: list[str]) -> None:
    hits = [_hit(f"https://{host}/x") for host in ("arxiv.org", "export.arxiv.org", "notarxiv.org")]
    assert [hit.url for hit in filter_hits_by_domain(hits, domain_filter)] == [f"https://{host}/x" for host in kept]
