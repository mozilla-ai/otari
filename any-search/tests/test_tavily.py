"""The Tavily adapter: the request it sends and how it reads Tavily's recorded answers."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, ProviderError, SearchResult
from any_search._http import USER_AGENT
from any_search.providers.tavily import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tavily"
KEY = "test-key"


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
        async with AnySearch.create("tavily", api_key=KEY, api_base=api_base, client=client) as tavily:
            result = await tavily.search("technology news", **kwargs)
    return result, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("tavily")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "TAVILY_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (False, False, False)
    assert metadata.max_results == MAX_RESULTS == 20
    (tier,) = [option for option in metadata.options if option.cost_tier]
    assert (tier.name, tier.default) == ("search_depth", "basic")
    names = {option.name for option in metadata.options}
    # The query and the count are the adapter's to set; auto_parameters can pick a more expensive depth.
    assert not {"query", "max_results", "time_range", "auto_parameters"} & names


@pytest.mark.asyncio
async def test_sends_the_query_in_a_post_and_the_key_as_a_bearer_token() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://api.tavily.com/search")
    assert _payload(request) == {"query": "technology news"}
    assert (request.headers["Authorization"], request.headers["User-Agent"]) == (f"Bearer {KEY}", USER_AGENT)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/tavily/")
    assert str(request.url) == "https://proxy.example/tavily/search"


@pytest.mark.asyncio
async def test_max_results_is_sent_at_most_20_and_cuts_the_hits() -> None:
    result, request = await _search(max_results=1)
    assert (_payload(request)["max_results"], len(result.hits)) == (1, 1)
    _, request = await _search(max_results=50)
    assert _payload(request)["max_results"] == 20


@pytest.mark.asyncio
async def test_options_are_sent_unless_null_and_time_range_wins() -> None:
    options: dict[str, Any] = {"search_depth": "advanced", "include_raw_content": "text", "topic": None}
    _, request = await _search(time_range="week", **options)
    assert _payload(request) == {
        "query": "technology news",
        "search_depth": "advanced",
        "include_raw_content": "text",
        "time_range": "week",
    }


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    body = _recorded("normal").json()
    recorded = body["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    first = result.hits[0]
    assert (first.title, first.snippet, first.text) == (recorded[0]["title"], recorded[0]["content"], None)
    assert first.published == datetime(2026, 11, 16, 1, 30, 48, tzinfo=UTC)
    assert first.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == body


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


SENTINEL = "Some text of the body."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (_recorded("error"), "unauthorized"),
        (httpx.Response(429, json={"detail": {"error": SENTINEL}}), "rate_limited"),
        (httpx.Response(432, json={"detail": {"error": SENTINEL}}), "plan_limit_exceeded"),
        (httpx.Response(433, json={"detail": {"error": SENTINEL}}), "pay_as_you_go_limit_exceeded"),
        (httpx.Response(400, json={"detail": {"error": SENTINEL}}), "http_error"),
        (httpx.Response(502, text=f"<html>{SENTINEL}</html>"), "http_error"),
    ],
)
async def test_a_failure_is_tagged_by_its_status_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value) and "Unauthorized" not in str(raised.value)


def _item(**changes: Any) -> dict[str, Any]:
    """A result with every field the adapter reads, changed by ``changes``."""
    item = {
        "url": "https://example.com/a",
        "title": SENTINEL,
        "content": SENTINEL,
        "raw_content": None,
        "published_date": "Thu, 15 Oct 2026 00:00:00 GMT",
    }
    return {**item, **changes}


def _answer(*items: Any) -> httpx.Response:
    return httpx.Response(200, json={"results": list(items)})


@pytest.mark.asyncio
async def test_raw_content_is_the_hits_text() -> None:
    result, _ = await _search(_answer(_item(raw_content="The page."), _item(raw_content="")))
    assert [hit.text for hit in result.hits] == ["The page.", None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("published_date", "published"),
    [
        ("Thu, 15 Oct 2026 00:00:00 GMT", datetime(2026, 10, 15, tzinfo=UTC)),
        (None, None),
        ("2 days ago", None),
    ],
)
async def test_published_date_is_read_as_rfc_1123_and_left_out_when_it_does_not_parse(
    published_date: str | None, published: datetime | None
) -> None:
    result, _ = await _search(_answer(_item(published_date=published_date)))
    assert result.hits[0].published == published


@pytest.mark.asyncio
async def test_only_the_results_within_the_count_are_checked() -> None:
    result, _ = await _search(_answer(_item(), {"not": "a result"}), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="results-missing"),
        pytest.param({"results": {}}, id="results-not-a-list"),
        pytest.param({"results": [SENTINEL]}, id="result-not-an-object"),
        pytest.param({"results": [_item(url="")]}, id="url-empty"),
        pytest.param({"results": [_item(url=7)]}, id="url-not-a-string"),
        pytest.param({"results": [_item(title=None)]}, id="title-null"),
        pytest.param({"results": [_item(content=None)]}, id="content-null"),
        pytest.param({"results": [_item(raw_content=7)]}, id="raw-content-not-a-string"),
        pytest.param({"results": [_item(published_date=7)]}, id="published-date-not-a-string"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(body: Any) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [httpx.Response(200, text="<html>results</html>"), httpx.Response(200, json=[])])
async def test_a_body_that_is_not_a_search_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")
