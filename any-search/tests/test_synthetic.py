"""The Synthetic adapter: the request it sends and how it reads Synthetic's recorded answers."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search._http import USER_AGENT
from any_search.providers.synthetic import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
KEY = "test-key"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    body = recorded["body"]
    if isinstance(body, str):
        return httpx.Response(recorded["status"], text=body)
    return httpx.Response(recorded["status"], json=body)


async def _search(
    answer: httpx.Response | None = None, *, api_base: str | None = None, **kwargs: Any
) -> tuple[SearchResult, httpx.Request]:
    """Search through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnySearch.create("synthetic", api_key=KEY, api_base=api_base, client=client) as synthetic:
            result = await synthetic.search("latest stable python release", **kwargs)
    return result, sent[0]


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("synthetic")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "SYNTHETIC_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url) == (False, False)
    assert metadata.max_results == MAX_RESULTS
    assert metadata.options == []


@pytest.mark.asyncio
async def test_posts_the_query_alone_with_the_key_as_a_bearer_token() -> None:
    _, request = await _search(max_results=2, time_range="week")
    assert (request.method, str(request.url)) == ("POST", "https://api.synthetic.new/v2/search")
    assert json.loads(request.content) == {"query": "latest stable python release"}
    assert (request.headers["Authorization"], request.headers["User-Agent"]) == (f"Bearer {KEY}", USER_AGENT)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/synthetic/")
    assert str(request.url) == "https://proxy.example/synthetic/v2/search"


@pytest.mark.asyncio
async def test_no_option_is_taken() -> None:
    options: dict[str, Any] = {"num": 5}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


SENTINEL = "sentinel-body-text"


def _item(name: str, published: str | None = "2026-11-01T00:00:00.000Z") -> dict[str, Any]:
    return {
        "url": f"https://example.com/{name}",
        "title": f"{name.upper()} {SENTINEL}",
        "text": f"about {name.upper()} {SENTINEL}",
        "published": published,
    }


def _answer() -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {"results": [_item("a")]}


@pytest.mark.asyncio
async def test_max_results_cuts_the_hits() -> None:
    answer = httpx.Response(200, json={"results": [_item(str(i)) for i in range(MAX_RESULTS + 1)]})
    result, _ = await _search(answer, max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/0", "https://example.com/1"]
    result, _ = await _search(answer, max_results=50)
    assert len(result.hits) == MAX_RESULTS


@pytest.mark.asyncio
async def test_time_range_drops_the_hits_dated_before_its_first_day_and_keeps_the_undated() -> None:
    now = datetime.now(UTC)
    body = {
        "results": [
            _item("old", (now - timedelta(days=9)).isoformat()),
            _item("first-day", (now - timedelta(days=7)).replace(hour=0, minute=0, second=0).isoformat()),
            _item("undated", None),
            _item("recent", now.isoformat()),
        ]
    }
    result, _ = await _search(httpx.Response(200, json=body), time_range="week")
    assert [hit.url.rsplit("/", 1)[1] for hit in result.hits] == ["first-day", "undated", "recent"]


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded[:MAX_RESULTS]]
    first = result.hits[0]
    assert (first.title, first.snippet, first.text) == (recorded[0]["title"], "", recorded[0]["text"] or None)
    assert first.published == datetime.fromisoformat(recorded[0]["published"])
    assert first.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_a_date_without_a_zone_is_utc() -> None:
    body = {"results": [_item("a", "2026-11-01T10:00:00")]}
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.hits[0].published == datetime(2026, 11, 1, 10, tzinfo=UTC)


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_refused_key_raises_with_its_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (401, "invalid_api_key")
    assert "Invalid API Key" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (httpx.Response(401, json={"error": "Authorization header required."}), "missing_api_key"),
        (
            httpx.Response(
                401, json={"error": "The search API is subscription-only: go to https://synthetic.new/billing"}
            ),
            "subscription_required",
        ),
        (httpx.Response(429, json={"error": SENTINEL}), "http_error"),
        (httpx.Response(500, json={"error": 7}), "http_error"),
        (httpx.Response(502, text=f"<html>{SENTINEL}</html>"), "http_error"),
        (httpx.Response(503, json=[SENTINEL]), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_its_error_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer()
    body["results"][0].update(score="unread", newField={"any": "thing"})
    body.update(newTopLevel=[])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_text_and_published_may_be_absent_or_null() -> None:
    item = {"url": "https://example.com/a", "title": "A"}
    body = {"results": [item, {**item, "text": None, "published": None}]}
    result, _ = await _search(httpx.Response(200, json=body))
    assert [(hit.text, hit.published) for hit in result.hits] == [(None, None), (None, None)]


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
        pytest.param(_setting(["text"], *ITEM, "text"), id="text-not-a-string"),
        pytest.param(_setting(1730419200, *ITEM, "published"), id="published-not-a-string"),
        pytest.param(_setting("last Tuesday", *ITEM, "published"), id="published-unparsable"),
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
    monkeypatch.delenv("SYNTHETIC_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="SYNTHETIC_API_KEY"):
        AnySearch.create("synthetic")
