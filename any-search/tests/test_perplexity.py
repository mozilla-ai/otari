"""The Perplexity adapter: the request it sends and how it reads Perplexity's recorded answers."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import (
    AnySearch,
    MissingCredentialError,
    ProviderError,
    SearchResult,
    UnsupportedParameterError,
    UnsupportedValueError,
)
from any_search._http import USER_AGENT
from any_search.providers.perplexity import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "perplexity"
KEY = "test-key"
QUERY = "latest stable python release"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    return httpx.Response(recorded["status"], json=recorded["body"])


async def _search(
    answer: httpx.Response | None = None, *, api_base: str | None = None, **kwargs: Any
) -> tuple[SearchResult, httpx.Request]:
    """Search through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnySearch.create("perplexity", api_key=KEY, api_base=api_base, client=client) as perplexity:
            result = await perplexity.search(QUERY, **kwargs)
    return result, sent[0]


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("perplexity")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "PERPLEXITY_API_KEY", True)
    assert (metadata.env_api_base, metadata.default_api_base) == ("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (False, False, False)
    assert metadata.max_results == MAX_RESULTS == 20
    tiers = [option for option in metadata.options if option.cost_tier]
    assert [(option.name, option.enum, option.default) for option in tiers] == [("search_type", ["web", "fast"], "web")]
    assert not any(option.operator_only for option in metadata.options)


@pytest.mark.asyncio
async def test_posts_the_query_with_the_key_as_a_bearer_token() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://api.perplexity.ai/search")
    assert json.loads(request.content) == {"query": QUERY}
    assert (request.headers["Authorization"], request.headers["User-Agent"]) == (f"Bearer {KEY}", USER_AGENT)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/perplexity/")
    assert str(request.url) == "https://proxy.example/perplexity/search"


def test_the_base_comes_from_the_sdks_variable_unless_one_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_BASE_URL", "https://env.example")
    assert AnySearch.create("perplexity", api_key=KEY)._api_base == "https://env.example"
    assert AnySearch.create("perplexity", api_key=KEY, api_base="https://passed.example")._api_base == (
        "https://passed.example"
    )


@pytest.mark.asyncio
async def test_max_results_is_sent_and_capped_at_twenty() -> None:
    _, request = await _search(max_results=3)
    assert json.loads(request.content)["max_results"] == 3
    _, request = await _search(max_results=50)
    assert json.loads(request.content)["max_results"] == MAX_RESULTS


@pytest.mark.asyncio
async def test_native_options_are_sent_as_given() -> None:
    options: dict[str, Any] = {
        "search_type": "fast",
        "search_recency_filter": "hour",
        "search_domain_filter": ["python.org", "-reddit.com"],
        "country": "DE",
        "search_language_filter": ["de"],
        "max_tokens_per_page": 256,
        "max_tokens": 2048,
        "last_updated_after_filter": "3/1/2025",
        "last_updated_before_filter": "3/1/2026",
    }
    _, request = await _search(**options)
    assert json.loads(request.content) == {"query": QUERY, **options}


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", ["people", "Fast"])
async def test_a_search_type_the_adapter_does_not_offer_is_refused_before_the_request(search_type: str) -> None:
    with pytest.raises(UnsupportedValueError, match="^perplexity takes search_type web or fast$") as refused:
        await _search(search_type=search_type)
    # A ValueError too, as a bad value is.
    assert isinstance(refused.value, ValueError)
    # The allowed values are a copy: emptying them leaves fast accepted.
    refused.value.allowed.clear()
    _, request = await _search(search_type="fast")
    assert json.loads(request.content)["search_type"] == "fast"


@pytest.mark.asyncio
async def test_time_range_becomes_the_recency_filter_and_wins_over_it() -> None:
    _, request = await _search(time_range="week")
    assert json.loads(request.content) == {"query": QUERY, "search_recency_filter": "week"}
    _, request = await _search(time_range="year", search_recency_filter="hour")
    assert json.loads(request.content)["search_recency_filter"] == "year"


@pytest.mark.asyncio
async def test_a_date_filter_is_sent_beside_time_range_for_perplexity_to_refuse() -> None:
    _, request = await _search(time_range="week", search_after_date_filter="1/1/2025")
    assert json.loads(request.content) == {
        "query": QUERY,
        "search_recency_filter": "week",
        "search_after_date_filter": "1/1/2025",
    }


@pytest.mark.asyncio
async def test_an_option_perplexity_does_not_take_is_refused() -> None:
    options: dict[str, Any] = {"search_mode": "academic"}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


SENTINEL = "sentinel-body-text"


def _item(name: str, date: str | None = "2026-11-01") -> dict[str, Any]:
    item: dict[str, Any] = {
        "url": f"https://example.com/{name}",
        "title": f"{name.upper()} {SENTINEL}",
        "snippet": f"about {name.upper()} {SENTINEL}",
        "last_updated": "2026-11-08",
    }
    if date is not None:
        item["date"] = date
    return item


def _answer() -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {"id": "an-id", "results": [_item("a")]}


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    undated, dated = result.hits
    assert (undated.title, undated.snippet, undated.text) == (recorded[0]["title"], "", recorded[0]["snippet"])
    assert (undated.published, dated.published) == (
        None,
        datetime.fromisoformat(recorded[1]["date"]).replace(tzinfo=UTC),
    )
    assert undated.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_max_results_cuts_the_hits() -> None:
    answer = httpx.Response(200, json={"results": [_item(str(i)) for i in range(MAX_RESULTS + 1)]})
    result, _ = await _search(answer, max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/0", "https://example.com/1"]


@pytest.mark.asyncio
async def test_an_empty_snippet_leaves_the_hit_without_text() -> None:
    body = {"results": [{**_item("a"), "snippet": ""}]}
    result, _ = await _search(httpx.Response(200, json=body))
    assert (result.hits[0].snippet, result.hits[0].text) == ("", None)


@pytest.mark.asyncio
async def test_a_date_may_be_absent_null_or_a_date_time() -> None:
    body = {"results": [_item("absent", None), {**_item("null"), "date": None}, _item("time", "2026-11-01T10:00:00")]}
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.published for hit in result.hits] == [None, None, datetime(2026, 11, 1, 10, tzinfo=UTC)]


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_refused_key_raises_with_its_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (401, "invalid_api_key")
    assert "Invalid API key" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (
            httpx.Response(
                400, json={"error": {"message": f"got {SENTINEL}", "type": "invalid_country_code", "code": 400}}
            ),
            "invalid_country_code",
        ),
        (
            httpx.Response(400, json={"error": {"message": SENTINEL, "type": "invalid_request", "code": 400}}),
            "invalid_request",
        ),
        (httpx.Response(429, json={"error": {"message": SENTINEL, "type": f"has spaces {SENTINEL}"}}), "http_error"),
        (
            httpx.Response(422, json={"detail": [{"loc": ["body", "query"], "msg": SENTINEL, "type": "missing"}]}),
            "http_error",
        ),
        (httpx.Response(500, json={"error": SENTINEL}), "http_error"),
        (httpx.Response(502, text=f"<html>{SENTINEL}</html>"), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_its_error_type_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer()
    body["results"][0].update(last_updated=7, newField={"any": "thing"})
    body.update(server_time=None, newTopLevel=[])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    body = _answer()
    body["results"].append({"url": 7})
    result, _ = await _search(httpx.Response(200, json=body), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


def _without(*path: str | int) -> Any:
    """A change to an answer that deletes the field at ``path``."""

    def change(body: dict[str, Any]) -> None:
        parent: Any = body
        for step in path[:-1]:
            parent = parent[step]
        del parent[path[-1]]

    return change


def _setting(value: Any, *path: str | int) -> Any:
    """A change to an answer that sets the field at ``path`` to ``value``."""

    def change(body: dict[str, Any]) -> None:
        parent: Any = body
        for step in path[:-1]:
            parent = parent[step]
        parent[path[-1]] = value

    return change


ITEM = ("results", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("results"), id="results-missing"),
        pytest.param(_setting({}, "results"), id="results-not-a-list"),
        pytest.param(_setting(None, "results"), id="results-null"),
        pytest.param(_setting(["an item"], "results"), id="item-not-an-object"),
        pytest.param(_without(*ITEM, "url"), id="url-missing"),
        pytest.param(_setting("", *ITEM, "url"), id="url-empty"),
        pytest.param(_setting(7, *ITEM, "url"), id="url-not-a-string"),
        pytest.param(_without(*ITEM, "title"), id="title-missing"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_without(*ITEM, "snippet"), id="snippet-missing"),
        pytest.param(_setting(None, *ITEM, "snippet"), id="snippet-null"),
        pytest.param(_setting(["text"], *ITEM, "snippet"), id="snippet-not-a-string"),
        pytest.param(_setting(20261101, *ITEM, "date"), id="date-not-a-string"),
        pytest.param(_setting("last Tuesday", *ITEM, "date"), id="date-unparsable"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(change: Any) -> None:
    body = _answer()
    change(body)
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="<html>results</html>"), httpx.Response(200, json=[])],
)
async def test_a_body_that_is_not_a_search_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="PERPLEXITY_API_KEY"):
        AnySearch.create("perplexity")
