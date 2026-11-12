"""The SpecStream adapter: the request it sends and how it reads SpecStream's answers."""

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search._http import USER_AGENT
from any_search.providers.specstream import DEFAULT_RESULTS, MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "specstream"
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
        async with AnySearch.create("specstream", api_key=KEY, api_base=api_base, client=client) as specstream:
            result = await specstream.search(QUERY, **kwargs)
    return result, sent[0]


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("specstream")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "SPECSTREAM_API_KEY", True)
    assert (metadata.env_api_base, metadata.default_api_base) == (None, "https://api.specstream.ai")
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (False, False, True)
    assert metadata.max_results == MAX_RESULTS == 100
    tiers = [option for option in metadata.options if option.cost_tier]
    assert [(option.name, option.enum, option.default) for option in tiers] == [
        ("speed", ["default", "fast"], "default")
    ]
    assert not any(option.operator_only for option in metadata.options)


@pytest.mark.asyncio
async def test_posts_the_query_with_the_key_as_a_bearer_token() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://api.specstream.ai/v1/web-search")
    assert json.loads(request.content) == {"q": QUERY, "num": DEFAULT_RESULTS}
    assert (request.headers["Authorization"], request.headers["User-Agent"]) == (f"Bearer {KEY}", USER_AGENT)
    assert QUERY.split()[0] not in str(request.url)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/specstream/")
    assert str(request.url) == "https://proxy.example/specstream/v1/web-search"


@pytest.mark.asyncio
async def test_max_results_is_sent_and_capped_at_one_hundred() -> None:
    _, request = await _search(max_results=3)
    assert json.loads(request.content)["num"] == 3
    _, request = await _search(max_results=500)
    assert json.loads(request.content)["num"] == MAX_RESULTS


@pytest.mark.asyncio
async def test_native_options_are_sent_as_given() -> None:
    options: dict[str, Any] = {
        "speed": "fast",
        "news": True,
        "passages": True,
        "pages_wait_ms": 1500,
        "language": "de",
        "country": "DE",
        "safe": "strict",
        "engines": "-CN",
        "page": 2,
    }
    _, request = await _search(**options)
    assert json.loads(request.content) == {"q": QUERY, "num": DEFAULT_RESULTS, **options}


@pytest.mark.asyncio
async def test_an_option_set_to_null_is_not_sent() -> None:
    _, request = await _search(speed=None)
    assert "speed" not in json.loads(request.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(("time_range", "days"), [("day", 1), ("week", 7), ("month", 30), ("year", 365)])
async def test_time_range_becomes_the_after_operator(time_range: str, days: int) -> None:
    before = (datetime.now(UTC) - timedelta(days=days)).date()
    _, request = await _search(time_range=time_range)
    after = (datetime.now(UTC) - timedelta(days=days)).date()
    query, operator = json.loads(request.content)["q"].rsplit(" ", 1)
    assert query == QUERY and operator.startswith("after:")
    assert before <= date.fromisoformat(operator.removeprefix("after:")) <= after


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["images", "q", "num"])
async def test_an_option_specstream_does_not_take_here_is_refused(option: str) -> None:
    options: dict[str, Any] = {option: True}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


SENTINEL = "sentinel-body-text"


def _item(name: str, **fields: Any) -> dict[str, Any]:
    return {
        "url": f"https://example.com/{name}",
        "title": f"{name.upper()} {SENTINEL}",
        "site": "example.com",
        "snippet": f"about {name.upper()} {SENTINEL}",
        "cited": False,
        **fields,
    }


def _answer(*items: dict[str, Any], price: Any = 0.005) -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {
        "data": {
            "request_id": "req_1",
            "query": QUERY,
            "kind": "fact",
            "columns": [],
            "answer": None,
            "rows": [],
            "sources": list(items) or [_item("a", published="2026-11-01")],
            "handoff": None,
            "took_ms": 900,
            "price_usd": price,
        }
    }


@pytest.mark.asyncio
async def test_reads_the_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["data"]["sources"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    undated, dated = result.hits
    assert (undated.title, undated.snippet, undated.text) == (recorded[0]["title"], recorded[0]["snippet"], None)
    assert (undated.published, dated.published) == (None, datetime(2025, 10, 7, tzinfo=UTC))
    assert undated.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (Decimal("0.005"), "reported", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_max_results_cuts_the_hits() -> None:
    answer = httpx.Response(200, json=_answer(*[_item(str(i)) for i in range(5)]))
    result, _ = await _search(answer, max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/0", "https://example.com/1"]


@pytest.mark.asyncio
async def test_an_excerpt_is_the_hits_text() -> None:
    body = _answer(_item("a", excerpt="The page's text."), _item("b", excerpt=""), _item("c", excerpt=None))
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.text for hit in result.hits] == ["The page's text.", None, None]


@pytest.mark.asyncio
async def test_a_news_time_wins_over_its_day_and_a_date_may_be_absent_or_null() -> None:
    body = _answer(
        _item("news", published="2026-11-10", published_at="2026-11-10T08:30:00Z"),
        _item("day", published="2026-11-01", published_at=None),
        _item("absent"),
        _item("null", published=None),
    )
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.published for hit in result.hits] == [
        datetime(2026, 11, 10, 8, 30, tzinfo=UTC),
        datetime(2026, 11, 1, tzinfo=UTC),
        None,
        None,
    ]


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_no_error_and_costs_nothing() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error, result.cost, result.cost_source) == ([], None, Decimal("0.0"), "reported")


@pytest.mark.asyncio
async def test_a_refused_key_raises_with_its_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (401, "invalid_token")
    assert "accept this API key" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (
            httpx.Response(402, json={"error": {"code": "insufficient_balance", "message": SENTINEL, "details": {}}}),
            "insufficient_balance",
        ),
        (httpx.Response(400, json={"error": {"code": "bad_request", "message": f"got {SENTINEL}"}}), "bad_request"),
        (httpx.Response(429, json={"error": {"code": f"has spaces {SENTINEL}", "message": SENTINEL}}), "http_error"),
        (httpx.Response(500, json={"error": SENTINEL}), "http_error"),
        (httpx.Response(502, text=f"<html>{SENTINEL}</html>"), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_its_error_code_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer()
    body["data"]["sources"][0].update(cited="maybe", newField={"any": "thing"})
    body["data"].update(answer=7, newTopLevel=[])
    body.update(meta={"any": "thing"})
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    body = _answer(_item("a"), {"url": 7})
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


ITEM = ("data", "sources", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("data"), id="data-missing"),
        pytest.param(_setting([], "data"), id="data-not-an-object"),
        pytest.param(_without("data", "sources"), id="sources-missing"),
        pytest.param(_setting({}, "data", "sources"), id="sources-not-a-list"),
        pytest.param(_setting(None, "data", "sources"), id="sources-null"),
        pytest.param(_without("data", "price_usd"), id="price-missing"),
        pytest.param(_setting(None, "data", "price_usd"), id="price-null"),
        pytest.param(_setting("0.005", "data", "price_usd"), id="price-not-a-number"),
        pytest.param(_setting(True, "data", "price_usd"), id="price-a-boolean"),
        pytest.param(_setting(-0.005, "data", "price_usd"), id="price-negative"),
        pytest.param(_setting(["an item"], "data", "sources"), id="item-not-an-object"),
        pytest.param(_without(*ITEM, "url"), id="url-missing"),
        pytest.param(_setting("", *ITEM, "url"), id="url-empty"),
        pytest.param(_setting(7, *ITEM, "url"), id="url-not-a-string"),
        pytest.param(_without(*ITEM, "title"), id="title-missing"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_without(*ITEM, "snippet"), id="snippet-missing"),
        pytest.param(_setting(["text"], *ITEM, "snippet"), id="snippet-not-a-string"),
        pytest.param(_setting(["text"], *ITEM, "excerpt"), id="excerpt-not-a-string"),
        pytest.param(_setting(20261101, *ITEM, "published"), id="published-not-a-string"),
        pytest.param(_setting("last Tuesday", *ITEM, "published"), id="published-unparsable"),
        pytest.param(_setting("2026-13-40T00:00:00Z", *ITEM, "published_at"), id="published-at-unparsable"),
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
    monkeypatch.delenv("SPECSTREAM_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="SPECSTREAM_API_KEY"):
        AnySearch.create("specstream")
