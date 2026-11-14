"""The You.com adapter: the request it sends and how it reads You.com's recorded answers."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search._http import USER_AGENT
from any_search.providers.youdotcom import DEFAULT_COUNT, HIGHLIGHTS_MAX_RESULTS, MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "youdotcom"
KEY = "test-key"
QUERY = "latest stable python release"
DEFAULTS = {"language": "EN", "knowledge": "core", "extraction": {"extraction_mode": "highlights"}}


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
        async with AnySearch.create("youdotcom", api_key=KEY, api_base=api_base, client=client) as youdotcom:
            result = await youdotcom.search(QUERY, **kwargs)
    return result, sent[0]


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("youdotcom")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "YDC_API_KEY", True)
    assert (metadata.env_api_base, metadata.default_api_base) == (None, "https://ydc-index.io")
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (False, False, False)
    assert metadata.max_results == MAX_RESULTS == 100
    assert not any(option.cost_tier or option.operator_only for option in metadata.options)
    options = {option.name: option for option in metadata.options}
    assert {name: options[name].default for name in DEFAULTS} == DEFAULTS
    # null turns these defaults off, and a host refuses any value an enum does not list.
    assert all(options[name].enum is None for name in DEFAULTS)


@pytest.mark.asyncio
async def test_posts_the_sdks_defaults_with_the_key_in_a_header() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://ydc-index.io/v1/search")
    assert json.loads(request.content) == {"query": QUERY, "count": DEFAULT_COUNT, **DEFAULTS}
    assert (request.headers["X-API-Key"], request.headers["User-Agent"]) == (KEY, USER_AGENT)
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/youdotcom/")
    assert str(request.url) == "https://proxy.example/youdotcom/v1/search"


@pytest.mark.asyncio
async def test_a_default_passed_as_none_is_not_sent() -> None:
    _, request = await _search(language=None, knowledge=None, extraction=None, country=None)
    assert json.loads(request.content) == {"query": QUERY, "count": DEFAULT_COUNT}


@pytest.mark.asyncio
async def test_native_options_are_sent_as_given() -> None:
    options: dict[str, Any] = {
        "freshness": "2026-01-01to2026-11-01",
        "offset": 2,
        "country": "DE",
        "language": "DE",
        "safesearch": "strict",
        "knowledge": "core",
        "exclude_domains": ["reddit.com"],
        "boost_domains": ["python.org"],
        "strict_operators": True,
        "extraction": {"extraction_mode": "full_page", "extraction_source": "cache"},
        "crawl_timeout": 30,
    }
    _, request = await _search(**options)
    assert json.loads(request.content) == {"query": QUERY, "count": DEFAULT_COUNT, **options}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extraction", "sent"),
    [
        (
            {"extraction_mode": "full_page", "full_page": {"extraction_formats": ["html"]}},
            {"extraction_mode": "full_page", "full_page": {"extraction_formats": ["markdown", "html"]}},
        ),
        (
            {"extraction_mode": "full_page", "full_page": {"extraction_formats": ["html", "markdown"]}},
            {"extraction_mode": "full_page", "full_page": {"extraction_formats": ["html", "markdown"]}},
        ),
        # An empty list and no list at all are You.com's ["markdown"].
        (
            {"extraction_mode": "full_page", "full_page": {"extraction_formats": []}},
            {"extraction_mode": "full_page", "full_page": {"extraction_formats": []}},
        ),
        ({"extraction_mode": "full_page"}, {"extraction_mode": "full_page"}),
        # Formats mean nothing under highlights, so they are left as given.
        (
            {"extraction_mode": "highlights", "full_page": {"extraction_formats": ["html"]}},
            {"extraction_mode": "highlights", "full_page": {"extraction_formats": ["html"]}},
        ),
    ],
)
async def test_full_page_always_asks_for_markdown(extraction: dict[str, Any], sent: dict[str, Any]) -> None:
    given = json.loads(json.dumps(extraction))
    _, request = await _search(extraction=extraction)
    assert json.loads(request.content)["extraction"] == sent
    assert extraction == given


@pytest.mark.asyncio
async def test_include_domains_is_sent_as_given() -> None:
    _, request = await _search(include_domains=["python.org"])
    assert json.loads(request.content)["include_domains"] == ["python.org"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_results", "options", "count"),
    [
        (3, {}, 3),
        (80, {}, HIGHLIGHTS_MAX_RESULTS),
        (
            80,
            {"extraction": {"extraction_mode": "highlights", "highlights": {"total_max_tokens": 1024}}},
            HIGHLIGHTS_MAX_RESULTS,
        ),
        (80, {"extraction": {"extraction_mode": "full_page"}}, 80),
        (150, {"extraction": {"extraction_mode": "full_page"}}, MAX_RESULTS),
        (150, {"extraction": None}, MAX_RESULTS),
    ],
)
async def test_max_results_is_the_count_capped_lower_under_highlights(
    max_results: int, options: dict[str, Any], count: int
) -> None:
    _, request = await _search(max_results=max_results, **options)
    assert json.loads(request.content)["count"] == count


@pytest.mark.asyncio
async def test_time_range_becomes_freshness_and_wins_over_it() -> None:
    _, request = await _search(time_range="week")
    assert json.loads(request.content)["freshness"] == "week"
    _, request = await _search(time_range="year", freshness="day")
    assert json.loads(request.content)["freshness"] == "year"


@pytest.mark.asyncio
async def test_an_option_youdotcom_does_not_take_is_refused() -> None:
    options: dict[str, Any] = {"livecrawl": "web"}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


SENTINEL = "sentinel-body-text"


def _item(name: str, **fields: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "url": f"https://example.com/{name}",
        "title": f"{name.upper()} {SENTINEL}",
        "description": f"about {name.upper()} {SENTINEL}",
        "contents": {"highlights": [f"{name} passage {SENTINEL}"]},
        "page_age": "2026-11-01T10:00:00",
    }
    item.update(fields)
    return item


def _answer() -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {"results": {"web": [_item("a")]}, "metadata": {"query": SENTINEL}}


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]["web"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    for hit, item in zip(result.hits, recorded, strict=True):
        assert (hit.title, hit.text, hit.raw) == (item["title"], None, item)
        assert hit.snippet == " ".join(part.strip() for part in item["contents"]["highlights"])
        assert hit.published == datetime.fromisoformat(item["page_age"]).replace(tzinfo=UTC)
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_the_full_page_is_the_text_and_the_description_the_snippet_without_highlights() -> None:
    without_contents = _item("plain", snippets=["kept in raw"])
    del without_contents["contents"]
    body = {
        "results": {
            "web": [
                _item("page", contents={"markdown": "# The page"}),
                _item("blank", contents={"highlights": [" ", ""]}),
                without_contents,
            ]
        }
    }
    result, _ = await _search(httpx.Response(200, json=body))
    page, blank, plain = result.hits
    assert (page.snippet, page.text) == ("about PAGE " + SENTINEL, "# The page")
    assert (blank.snippet, blank.text) == ("about BLANK " + SENTINEL, None)
    assert (plain.snippet, plain.raw["snippets"]) == ("about PLAIN " + SENTINEL, ["kept in raw"])


@pytest.mark.asyncio
async def test_a_result_with_only_a_url_is_a_hit() -> None:
    body = {"results": {"web": [{"url": "https://example.com/bare"}]}}
    result, _ = await _search(httpx.Response(200, json=body))
    hit = result.hits[0]
    assert hit.url == "https://example.com/bare"
    assert (hit.title, hit.snippet, hit.text, hit.published) == ("", "", None, None)


@pytest.mark.asyncio
async def test_a_result_without_a_web_url_is_dropped_and_the_count_filled_after_it() -> None:
    body = {
        "results": {
            "web": [
                _item("none", url=None),
                {key: value for key, value in _item("absent").items() if key != "url"},
                _item("ftp", url="ftp://example.com/file"),
                _item("relative", url="/a/path"),
                _item("a"),
                _item("b"),
                _item("c"),
            ],
            "news": [_item("news")],
        }
    }
    result, _ = await _search(httpx.Response(200, json=body), max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/a", "https://example.com/b"]


@pytest.mark.asyncio
async def test_a_page_age_may_be_absent_null_zoned_or_in_another_form() -> None:
    plus_two = timezone(timedelta(hours=2))
    ages: list[Any] = [None, "2026-11-01", "2026-11-01T10:00:00+02:00", "7/29/2024 10:38:56 AM", "171 days ago"]
    body = {"results": {"web": [_item("absent"), *(_item(str(i), page_age=age) for i, age in enumerate(ages))]}}
    del body["results"]["web"][0]["page_age"]
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.published for hit in result.hits] == [
        None,
        None,
        datetime(2026, 11, 1, tzinfo=UTC),
        datetime(2026, 11, 1, 10, tzinfo=plus_two),
        None,
        None,
    ]
    assert result.hits[-1].raw["page_age"] == "171 days ago"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [_recorded("empty").json(), {"results": {"web": []}}, {"results": {"news": [_item("news")]}}]
)
async def test_an_empty_answer_has_no_hits_and_no_error(body: dict[str, Any]) -> None:
    result, _ = await _search(httpx.Response(200, json=body))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_refused_key_raises_with_its_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (403, "forbidden")
    assert "Forbidden" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (httpx.Response(401, json={"error": SENTINEL}), "unauthorized"),
        (httpx.Response(403, json={"message": SENTINEL}), "forbidden"),
        (httpx.Response(422, json={"error": f"highlights supports count up to 50, got {SENTINEL}"}), "invalid_request"),
        (httpx.Response(422, json={"error": "invalid_request_parameter"}), "invalid_request"),
        (httpx.Response(402, json={"error": "payment_required", "detail": SENTINEL}), "payment_required"),
        (httpx.Response(402, json={"error": "insufficient_credits"}), "insufficient_credits"),
        (httpx.Response(402, json={"error": f"has spaces {SENTINEL}"}), "payment_required"),
        (httpx.Response(402, text=SENTINEL), "payment_required"),
        (httpx.Response(429, json={"error": "rate_limited"}), "http_error"),
        (httpx.Response(500, json={"error": SENTINEL}), "http_error"),
        (httpx.Response(502, text=f"<html>{SENTINEL}</html>"), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_its_status_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer()
    body["results"]["web"][0].update(thumbnail_url=7, favicon_url=None, newField={"any": "thing"})
    body["results"]["web"][0]["contents"]["html"] = 7
    body["results"].update(news=None, knowledge="anything")
    body.update(metadata=None, newTopLevel=[])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    body = _answer()
    body["results"]["web"].append({"url": 7})
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


ITEM = ("results", "web", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("results"), id="results-missing"),
        pytest.param(_setting([], "results"), id="results-not-an-object"),
        pytest.param(_setting(None, "results"), id="results-null"),
        pytest.param(_setting({}, "results", "web"), id="web-not-a-list"),
        pytest.param(_setting(None, "results", "web"), id="web-null"),
        pytest.param(_setting(["an item"], "results", "web"), id="item-not-an-object"),
        pytest.param(_setting(7, *ITEM, "url"), id="url-not-a-string"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_setting(["text"], *ITEM, "description"), id="description-not-a-string"),
        pytest.param(_setting(None, *ITEM, "contents"), id="contents-null"),
        pytest.param(_setting("text", *ITEM, "contents"), id="contents-not-an-object"),
        pytest.param(_setting("text", *ITEM, "contents", "highlights"), id="highlights-not-a-list"),
        pytest.param(_setting([7], *ITEM, "contents", "highlights"), id="highlight-not-a-string"),
        pytest.param(_setting(None, *ITEM, "contents", "markdown"), id="markdown-null"),
        pytest.param(_setting(20261101, *ITEM, "page_age"), id="page-age-not-a-string"),
        pytest.param(_setting("2026-13-45T00:00:00", *ITEM, "page_age"), id="page-age-unparsable"),
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
async def test_a_result_dropped_for_its_url_is_still_checked() -> None:
    body = _answer()
    body["results"]["web"].insert(0, _item("none", url=None, title=7))
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert raised.value.tag == "unexpected_response"


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
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="YDC_API_KEY"):
        AnySearch.create("youdotcom")
