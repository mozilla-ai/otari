"""The Firecrawl adapter: the request it sends, how it reads recorded answers, and the fields it checks."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search._http import USER_AGENT
from any_search.providers.firecrawl import DEFAULT_LIMIT, MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "firecrawl"
KEY = "test-key"
QUERY = "python release schedule"
SENTINEL = "sentinel-body-text"


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
        async with AnySearch.create("firecrawl", api_key=KEY, api_base=api_base, client=client) as firecrawl:
            result = await firecrawl.search(QUERY, **kwargs)
    return result, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


def _item(**fields: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "url": "https://example.com/page",
        "title": f"Title {SENTINEL}",
        "description": f"Description {SENTINEL}",
        "position": 1,
    }
    item.update(fields)
    return item


def _answer(*items: dict[str, Any]) -> dict[str, Any]:
    """A successful answer holding ``items`` as its web results."""
    return {"success": True, "data": {"web": list(items)}, "creditsUsed": 2, "id": "01a21281"}


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("firecrawl")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "FIRECRAWL_API_KEY", True)
    assert (metadata.env_api_base, metadata.default_api_base) == (None, "https://api.firecrawl.dev")
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (False, False, False)
    assert metadata.max_results == MAX_RESULTS == 100
    assert not any(option.cost_tier for option in metadata.options)
    assert not any(option.operator_only for option in metadata.options)
    names = {option.name for option in metadata.options}
    # The query, the count, the origin and the sources the envelope cannot read are not options.
    assert not names & {"query", "limit", "origin", "integration", "sources"}


@pytest.mark.asyncio
async def test_posts_the_query_with_a_bearer_key_and_no_limit_unless_asked() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://api.firecrawl.dev/v2/search")
    assert _payload(request) == {"query": QUERY}
    assert (request.headers["Authorization"], request.headers["User-Agent"]) == (f"Bearer {KEY}", USER_AGENT)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://firecrawl.internal.example/")
    assert str(request.url) == "https://firecrawl.internal.example/v2/search"


@pytest.mark.asyncio
async def test_native_options_are_sent_as_given_and_none_is_not_sent() -> None:
    options: dict[str, Any] = {
        "tbs": "sbd:1,qdr:w",
        "categories": ["research"],
        "includeDomains": ["python.org"],
        "location": "Berlin,Germany",
        "country": "DE",
        "safe": True,
        "highlights": False,
        "ignoreInvalidURLs": True,
        "timeout": 30000,
    }
    _, request = await _search(**options, excludeDomains=None)
    assert _payload(request) == {"query": QUERY, **options}


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_results", "limit"), [(1, 1), (100, 100), (250, 100)])
async def test_max_results_is_the_limit_at_most_100(max_results: int, limit: int) -> None:
    _, request = await _search(max_results=max_results)
    assert _payload(request)["limit"] == limit


@pytest.mark.asyncio
async def test_without_max_results_at_most_firecrawls_default_of_10_are_kept() -> None:
    answer = _answer(*(_item(url=f"https://example.com/{n}") for n in range(DEFAULT_LIMIT + 2)))
    result, _ = await _search(httpx.Response(200, json=answer))
    assert DEFAULT_LIMIT == 10
    assert [hit.url for hit in result.hits] == [f"https://example.com/{n}" for n in range(DEFAULT_LIMIT)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_range", "tbs"), [("day", "qdr:d"), ("week", "qdr:w"), ("month", "qdr:m"), ("year", "qdr:y")]
)
async def test_time_range_becomes_tbs_and_wins_over_the_option(time_range: str, tbs: str) -> None:
    _, request = await _search(time_range=time_range, tbs="qdr:h")
    assert _payload(request)["tbs"] == tbs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scrape_options", "sent"),
    [
        ({"formats": ["html"]}, {"formats": ["markdown", "html"]}),
        ({"formats": ["markdown", "links"]}, {"formats": ["markdown", "links"]}),
        ({"formats": [{"type": "markdown"}]}, {"formats": [{"type": "markdown"}]}),
        ({"onlyMainContent": False}, {"onlyMainContent": False, "formats": ["markdown"]}),
        ({"formats": []}, {"formats": ["markdown"]}),
        ({}, {"formats": ["markdown"]}),
    ],
)
async def test_scrape_options_always_ask_for_markdown(scrape_options: dict[str, Any], sent: dict[str, Any]) -> None:
    given = json.loads(json.dumps(scrape_options))
    _, request = await _search(scrapeOptions=scrape_options)
    assert _payload(request)["scrapeOptions"] == sent
    assert scrape_options == given


@pytest.mark.asyncio
async def test_scrape_options_that_are_not_an_object_are_refused_before_the_request() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(scrapeOptions=["markdown"])
    assert (raised.value.status, raised.value.tag) == (None, "invalid_option")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["sources", "limit", "origin", "integration"])
async def test_a_reserved_or_unoffered_option_is_refused(name: str) -> None:
    with pytest.raises(UnsupportedParameterError, match=name):
        await _search(None, **{name: "x"})


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()
    web = recorded["data"]["web"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in web]
    assert [hit.title for hit in result.hits] == [item["title"] for item in web]
    assert [hit.snippet for hit in result.hits] == [item["description"] for item in web]
    assert all(hit.text is None and hit.published is None for hit in result.hits)
    assert [hit.raw for hit in result.hits] == web
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == recorded


@pytest.mark.asyncio
async def test_the_recorded_empty_answer_has_no_hits() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_markdown_is_the_text() -> None:
    answer = _answer(_item(markdown="# Page"), _item(url="https://example.com/b", markdown=None), _item(markdown=""))
    result, _ = await _search(httpx.Response(200, json=answer), max_results=3)
    assert [hit.text for hit in result.hits] == ["# Page", None, None]


@pytest.mark.asyncio
async def test_an_answer_without_web_results_has_no_hits() -> None:
    result, _ = await _search(httpx.Response(200, json={"success": True, "data": {}}))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_hits_are_cut_to_max_results() -> None:
    answer = _answer(*(_item(url=f"https://example.com/{index}") for index in range(4)))
    result, _ = await _search(httpx.Response(200, json=answer), max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/0", "https://example.com/1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["", "example.com/page", "ftp://example.com/file", "https://", "https://exa mple.com/"])
async def test_a_result_without_a_web_url_is_dropped_and_the_count_filled_after_it(url: str) -> None:
    answer = _answer(_item(url=url), _item(url="https://example.com/a"), _item(url="https://example.com/b"))
    result, _ = await _search(httpx.Response(200, json=answer), max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/a", "https://example.com/b"]


@pytest.mark.asyncio
async def test_a_result_without_a_title_or_description_has_empty_ones() -> None:
    result, _ = await _search(httpx.Response(200, json=_answer({"url": "https://example.com/"})))
    assert (result.hits[0].title, result.hits[0].snippet) == ("", "")


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    answer = _answer(_item(position="first", metadata=7, newField={}))
    answer["data"]["news"] = "not a list"
    answer["warning"] = 3
    result, _ = await _search(httpx.Response(200, json=answer))
    assert [hit.url for hit in result.hits] == ["https://example.com/page"]


@pytest.mark.asyncio
async def test_success_false_inside_a_200_is_the_results_error() -> None:
    body = {"success": False, "code": "SEARCH_FAILED", "error": f"failed for {QUERY} {SENTINEL}"}
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.error is not None
    assert (result.error.tag, result.error.status, result.hits) == ("SEARCH_FAILED", None, [])
    body = {"success": False, "error": SENTINEL}
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.error is not None
    assert result.error.tag == "unsuccessful"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"success": "yes", "data": {"web": []}}, id="success-not-a-boolean"),
        pytest.param({"data": {"web": []}}, id="success-missing"),
        pytest.param({"success": True}, id="data-missing"),
        pytest.param({"success": True, "data": None}, id="data-null"),
        pytest.param({"success": True, "data": [SENTINEL]}, id="data-not-an-object"),
        pytest.param({"success": True, "data": {"web": None}}, id="web-null"),
        pytest.param({"success": True, "data": {"web": {"0": SENTINEL}}}, id="web-not-a-list"),
        pytest.param(_answer(SENTINEL), id="item-not-an-object"),  # type: ignore[arg-type]
        pytest.param(_answer({"title": SENTINEL}), id="url-missing"),
        pytest.param(_answer(_item(url=None)), id="url-null"),
        pytest.param(_answer(_item(url=[SENTINEL])), id="url-not-a-string"),
        pytest.param(_answer(_item(title=None)), id="title-null"),
        pytest.param(_answer(_item(title=[SENTINEL])), id="title-not-a-string"),
        pytest.param(_answer(_item(description=None)), id="description-null"),
        pytest.param(_answer(_item(description={"text": SENTINEL})), id="description-not-a-string"),
        pytest.param(_answer(_item(markdown=[SENTINEL])), id="markdown-not-a-string"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(body: dict[str, Any]) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
async def test_the_recorded_refused_key_is_tagged_unauthorized() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.provider, raised.value.status, raised.value.tag) == ("firecrawl", 401, "unauthorized")
    assert "Invalid token" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "tag"),
    [
        (401, {"success": False, "error": f"Unauthorized: {SENTINEL}"}, "unauthorized"),
        (402, {"success": False, "error": SENTINEL, "code": "insufficient_credits"}, "payment_required"),
        (429, {"error": f"Request rate limit exceeded {SENTINEL}"}, "rate_limited"),
        (400, {"success": False, "code": "CATEGORY_REMOVED", "error": f"{QUERY} {SENTINEL}"}, "CATEGORY_REMOVED"),
        (400, {"success": False, "error": "Invalid request body", "details": [{"message": SENTINEL}]}, "http_error"),
        (403, {"success": False, "error": SENTINEL, "signup_url": "https://firecrawl.dev/k/x"}, "http_error"),
        (408, {"success": False, "code": "SCRAPE_TIMEOUT", "error": SENTINEL}, "SCRAPE_TIMEOUT"),
        (500, {"success": False, "code": f"not a code {SENTINEL}", "error": SENTINEL}, "http_error"),
        (502, {"success": False, "code": None}, "http_error"),
    ],
)
async def test_a_failure_is_tagged_by_status_or_code_and_carries_no_body_text(
    status: int, body: dict[str, Any], tag: str
) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(status, json=body))
    assert (raised.value.status, raised.value.tag) == (status, tag)
    assert SENTINEL not in str(raised.value) and QUERY not in str(raised.value)


@pytest.mark.asyncio
async def test_a_failure_with_a_plain_body_is_tagged_http_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(502, text=f"Bad Gateway for {QUERY}"))
    assert (raised.value.status, raised.value.tag) == (502, "http_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [httpx.Response(200, text="not json"), httpx.Response(200, json=[])])
async def test_a_body_that_is_not_an_object_is_an_invalid_response(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="FIRECRAWL_API_KEY"):
        AnySearch.create("firecrawl")
