"""The Brave adapter: the request it sends, how it reads Brave's recorded answers, and what it logs."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, ProviderError, SearchResult
from any_search._http import USER_AGENT
from any_search.providers.brave import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "brave"
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
        async with AnySearch.create("brave", api_key=KEY, api_base=api_base, client=client) as brave:
            result = await brave.search("rust release notes", **kwargs)
    return result, sent[0]


def _params(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("brave")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "BRAVE_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (True, False, False)
    assert metadata.max_results == MAX_RESULTS == 20
    # The query and the count are the adapter's to set.
    assert not {"q", "count"} & {option.name for option in metadata.options}


@pytest.mark.asyncio
async def test_sends_the_query_as_parameters_of_a_get_and_the_key_as_a_header() -> None:
    _, request = await _search()
    assert request.method == "GET"
    assert str(request.url.copy_with(query=None)) == "https://api.search.brave.com/res/v1/web/search"
    assert _params(request) == {"q": "rust release notes", "count": "20"}
    assert (request.headers["X-Subscription-Token"], request.headers["User-Agent"]) == (KEY, USER_AGENT)
    assert request.headers["Accept"] == "application/json"


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/brave/")
    assert str(request.url.copy_with(query=None)) == "https://proxy.example/brave/res/v1/web/search"


@pytest.mark.asyncio
async def test_max_results_sets_count_and_cuts_the_hits() -> None:
    result, request = await _search(max_results=1)
    assert _params(request)["count"] == "1"
    assert len(result.hits) == 1


@pytest.mark.asyncio
async def test_max_results_is_capped_at_braves_limit() -> None:
    answer = httpx.Response(200, json={"web": {"results": [_item(url=f"https://example.com/{i}") for i in range(21)]}})
    result, request = await _search(answer, max_results=50)
    assert _params(request)["count"] == "20"
    assert len(result.hits) == 20


@pytest.mark.asyncio
async def test_options_are_sent_as_parameters_unless_null() -> None:
    options: dict[str, Any] = {
        "country": "DE",
        "offset": 1,
        "extra_snippets": True,
        "spellcheck": False,
        "goggles": None,
    }
    _, request = await _search(**options)
    assert _params(request) == {
        "q": "rust release notes",
        "count": "20",
        "country": "DE",
        "offset": "1",
        "extra_snippets": "true",
        "spellcheck": "false",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("time_range", "freshness"), [("day", "pd"), ("week", "pw"), ("month", "pm"), ("year", "py")])
async def test_time_range_is_braves_freshness_and_wins_over_the_option(time_range: str, freshness: str) -> None:
    _, request = await _search(time_range=time_range, freshness="2020-01-01to2020-12-31")
    assert _params(request)["freshness"] == freshness


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    body = _recorded("normal").json()
    recorded = body["web"]["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    first = result.hits[0]
    assert (first.title, first.snippet, first.text) == (recorded[0]["title"], recorded[0]["description"], None)
    assert first.published == datetime.fromisoformat(recorded[0]["page_age"]).replace(tzinfo=UTC)
    assert first.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == body


@pytest.mark.asyncio
async def test_an_answer_without_web_results_has_no_hits_and_no_error() -> None:
    for answer in (_recorded("empty"), httpx.Response(200, json={"web": None})):
        result, _ = await _search(answer)
        assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_refused_key_raises_with_its_status_and_code_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (422, "subscription_token_invalid")
    assert "The provided subscription token" not in str(raised.value)


SENTINEL = "Some text of the body."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (httpx.Response(422, json={"error": {"code": "VALIDATION", "detail": SENTINEL}}), "validation"),
        (httpx.Response(429, json={"error": {"code": "RATE_LIMITED", "meta": SENTINEL}}), "rate_limited"),
        (httpx.Response(500, json={"error": {"code": SENTINEL}}), "http_error"),
        (httpx.Response(502, text=f"<html>{SENTINEL}</html>"), "http_error"),
        (httpx.Response(503, json={"error": SENTINEL}), "http_error"),
    ],
)
async def test_a_failure_is_tagged_by_its_code_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value)


def _item(**changes: Any) -> dict[str, Any]:
    """A web result with every field the adapter reads, changed by ``changes``."""
    item = {
        "title": SENTINEL,
        "url": "https://example.com/a",
        "description": SENTINEL,
        "page_age": "2026-11-12T22:37:39",
    }
    return {**item, **changes}


def _answer(*items: Any) -> httpx.Response:
    return httpx.Response(200, json={"web": {"results": list(items)}})


@pytest.mark.asyncio
async def test_only_the_results_within_the_count_are_checked() -> None:
    result, _ = await _search(_answer(_item(), {"not": "a result"}), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [{}, {"description": None, "page_age": None}], ids=["absent", "null"])
async def test_a_result_without_a_description_or_a_page_age_has_an_empty_snippet_and_no_date(
    result: dict[str, Any],
) -> None:
    answer = await _search(_answer({"title": "A", "url": "https://example.com/a", **result}))
    hit = answer[0].hits[0]
    assert (hit.snippet, hit.published) == ("", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page_age", "published"),
    [
        ("2026-11-12T22:37:39", datetime(2026, 11, 12, 22, 37, 39, tzinfo=UTC)),
        ("2026-11-12T22:37:39+02:00", datetime(2026, 11, 12, 20, 37, 39, tzinfo=UTC)),
        ("2 days ago", None),
        ("2026-13-40T00:00:00", None),
    ],
)
async def test_page_age_is_read_as_utc_and_left_out_when_it_does_not_parse(
    page_age: str, published: datetime | None
) -> None:
    result, _ = await _search(_answer(_item(page_age=page_age)))
    assert result.hits[0].published == published


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "web",
    [
        pytest.param([], id="web-not-an-object"),
        pytest.param({}, id="results-missing"),
        pytest.param({"results": {}}, id="results-not-a-list"),
        pytest.param({"results": [SENTINEL]}, id="result-not-an-object"),
        pytest.param({"results": [{"title": SENTINEL}]}, id="url-missing"),
        pytest.param({"results": [_item(url="")]}, id="url-empty"),
        pytest.param({"results": [_item(url=7)]}, id="url-not-a-string"),
        pytest.param({"results": [{"url": "https://example.com/a"}]}, id="title-missing"),
        pytest.param({"results": [_item(title=7)]}, id="title-not-a-string"),
        pytest.param({"results": [_item(page_age=7)]}, id="page-age-not-a-string"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(web: Any) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json={"web": web}))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [httpx.Response(200, text="<html>results</html>"), httpx.Response(200, json=[])])
async def test_a_body_that_is_not_a_search_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


SENTINEL_KEY = "sentinel-key-3b8d20"
SENTINEL_QUERY = "sentinel-query-7c2e51"


@asynccontextmanager
async def _local_brave(*, answers: bool = True) -> AsyncIterator[str]:
    """A local server that answers every request with the recorded search over a real socket, or hangs up."""
    body = json.dumps(_recorded("normal").json()).encode()

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        if answers:
            head = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
            writer.write(head.encode() + body)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        yield f"http://127.0.0.1:{port}"


def _texts(exc: BaseException | None) -> list[str]:
    texts: list[str] = []
    while exc is not None:
        texts += [str(exc), repr(exc)]
        exc = exc.__cause__ or exc.__context__
    return texts


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_no_log_record_from_httpx_or_httpcore_carries_the_key_or_the_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Through a real transport, since ``MockTransport`` skips httpcore, which logs below httpx at DEBUG."""
    caplog.set_level(logging.DEBUG)
    async with _local_brave() as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("brave", api_key=SENTINEL_KEY, api_base=api_base, client=client) as brave:
            result = await brave.search(SENTINEL_QUERY)
    assert result.hits
    messages = [record.getMessage() for record in caplog.records]
    assert {"httpx", "httpcore.http11"} <= {record.name for record in caplog.records}
    assert [message for message in messages if SENTINEL_KEY in message or SENTINEL_QUERY in message] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_a_transport_failure_carries_neither_the_key_nor_the_query(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    # The server hangs up without an answer, which httpx raises as a transport error.
    async with _local_brave(answers=False) as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("brave", api_key=SENTINEL_KEY, api_base=api_base, client=client) as brave:
            with pytest.raises(ProviderError) as raised:
                await brave.search(SENTINEL_QUERY)
    assert raised.value.tag == "network"
    texts = _texts(raised.value) + [record.getMessage() for record in caplog.records]
    assert [text for text in texts if SENTINEL_KEY in text or SENTINEL_QUERY in text] == []
