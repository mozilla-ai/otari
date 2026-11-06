"""The SearXNG adapter: the request it sends, how it reads SearXNG's recorded answers, and what it logs."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from any_search import MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search._http import USER_AGENT
from any_search.providers.searxng import MAX_RESULTS, SearxngProvider

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "searxng"
BASE = "https://searx.example"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    body = recorded["body"]
    if isinstance(body, str):
        return httpx.Response(recorded["status"], text=body)
    return httpx.Response(recorded["status"], json=body)


async def _search(
    answer: httpx.Response | None = None, *, api_base: str = BASE, **kwargs: Any
) -> tuple[SearchResult, httpx.Request]:
    """Search through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with SearxngProvider(api_base=api_base, client=client) as searxng:
            result = await searxng.search("python release", **kwargs)
    return result, sent[0]


def _params(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


def _form(request: httpx.Request) -> dict[str, str]:
    return {name: values[0] for name, values in parse_qs(request.content.decode()).items()}


def test_metadata() -> None:
    metadata = SearxngProvider.METADATA
    assert (metadata.tier, metadata.env_key, metadata.env_api_base) == ("production", None, "SEARXNG_BASE_URL")
    assert (metadata.requires_api_key, metadata.requires_api_base, metadata.default_api_base) == (False, True, None)
    assert (metadata.query_in_url, metadata.key_in_url) == (True, False)
    assert metadata.max_results == MAX_RESULTS == 100
    assert {option.name for option in metadata.options if option.operator_only} == {"engines", "method"}
    # What the adapter sets itself, or a shared parameter maps to, is not an option.
    names = {option.name for option in metadata.options}
    assert not names & {"q", "format", "time_range", "User-Agent"}


@pytest.mark.asyncio
async def test_sends_the_query_as_parameters_of_a_get() -> None:
    _, request = await _search()
    assert (request.method, str(request.url.copy_with(query=None))) == ("GET", f"{BASE}/search")
    assert _params(request) == {"q": "python release", "format": "json"}
    assert request.headers["User-Agent"] == USER_AGENT
    assert request.content == b""


@pytest.mark.asyncio
async def test_method_post_sends_the_same_fields_as_form_data() -> None:
    _, request = await _search(method="post", engines="wikinews")
    assert (request.method, str(request.url)) == ("POST", f"{BASE}/search")
    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert _form(request) == {"q": "python release", "format": "json", "engines": "wikinews"}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "form", "post ", ""])
async def test_a_method_other_than_get_or_post_is_refused_before_any_request(method: str) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with SearxngProvider(api_base=BASE, client=client) as searxng:
            with pytest.raises(ValueError, match="method must be one of get, post"):
                await searxng.search("python release", method=method)
    assert sent == []


@pytest.mark.asyncio
async def test_method_get_is_the_default_and_never_sent() -> None:
    _, request = await _search(method="get")
    assert request.method == "GET"
    assert "method" not in _params(request)


@pytest.mark.asyncio
async def test_api_base_is_the_instance_root() -> None:
    _, request = await _search(api_base="https://proxy.example/searx/")
    assert str(request.url.copy_with(query=None)) == "https://proxy.example/searx/search"


def test_an_instance_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    with pytest.raises(MissingCredentialError, match="SEARXNG_BASE_URL"):
        SearxngProvider()


@pytest.mark.asyncio
async def test_the_instance_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://env.example")
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with SearxngProvider(client=client) as searxng:
            await searxng.search("python release")
    assert str(sent[0].url.copy_with(query=None)) == "https://env.example/search"


@pytest.mark.asyncio
async def test_native_options_pass_through_under_searxngs_names() -> None:
    options: dict[str, Any] = {
        "engines": "wikinews,arxiv",
        "categories": "news",
        "language": "de-CH",
        "safesearch": 2,
        "pageno": 3,
    }
    _, request = await _search(**options)
    assert _params(request) == {"q": "python release", "format": "json"} | {
        name: str(value) for name, value in options.items()
    }


@pytest.mark.asyncio
async def test_an_option_set_to_null_is_not_sent() -> None:
    options: dict[str, Any] = {"engines": None, "language": None, "method": None}
    _, request = await _search(**options)
    assert (request.method, _params(request)) == ("GET", {"q": "python release", "format": "json"})


@pytest.mark.asyncio
@pytest.mark.parametrize("time_range", ["day", "week", "month", "year"])
async def test_time_range_is_searxngs_own(time_range: str) -> None:
    _, request = await _search(time_range=time_range)
    assert _params(request)["time_range"] == time_range


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", ["q", "format", "extracted_content"])
async def test_what_the_adapter_sets_or_does_not_know_is_not_an_option(reserved: str) -> None:
    options: dict[str, Any] = {reserved: "1"}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    first = result.hits[0]
    assert (first.title, first.snippet, first.text) == (recorded[0]["title"], recorded[0]["content"], None)
    # Recorded as "2024-12-18T10:54:41", with no zone, so UTC.
    assert first.published == datetime(2024, 12, 18, 10, 54, 41, tzinfo=UTC)
    assert first.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_max_results_cuts_the_hits() -> None:
    result, request = await _search(max_results=2)
    assert len(result.hits) == 2
    assert "max_results" not in _params(request)


@pytest.mark.asyncio
async def test_without_max_results_a_page_is_cut_at_the_adapters_limit() -> None:
    answer = _answer([_item(str(i)) for i in range(MAX_RESULTS + 1)])
    result, _ = await _search(answer)
    assert len(result.hits) == MAX_RESULTS


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_refused_parameter_raises_with_its_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"), pageno="abc")
    assert (raised.value.status, raised.value.tag) == (400, "invalid_parameter")
    assert "pageno" not in str(raised.value) and "abc" not in str(raised.value)


@pytest.mark.asyncio
async def test_engines_that_all_failed_are_an_error_inside_the_answer() -> None:
    result, _ = await _search(_recorded("in_body_error"))
    assert result.hits == []
    assert result.error is not None
    assert (result.error.tag, result.error.status) == ("engines_unresponsive", None)
    assert "access denied" not in result.error.model_dump_json()


@pytest.mark.asyncio
async def test_engines_that_failed_beside_only_unusable_results_are_an_error() -> None:
    answer = _answer([_item("torrent", url="magnet:?xt=urn:btih:0123")], unresponsive=[["duckduckgo", "CAPTCHA"]])
    result, _ = await _search(answer)
    assert result.hits == []
    assert result.error is not None
    assert result.error.tag == "engines_unresponsive"


@pytest.mark.asyncio
async def test_an_engine_that_failed_beside_hits_is_no_error() -> None:
    answer = _answer([_item("a")], unresponsive=[["duckduckgo", "CAPTCHA"]])
    result, _ = await _search(answer)
    assert ([hit.url for hit in result.hits], result.error) == (["https://example.com/a"], None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (
            httpx.Response(403, text="<html><title>403 Forbidden</title>sentinel-body-text</html>"),
            "forbidden",
        ),
        (httpx.Response(400, json={"error": "Empty pageno parameter"}), "invalid_parameter"),
        (httpx.Response(400, json={"error": "No query"}), "http_error"),
        (httpx.Response(500, json={"error": "Empty pageno parameter"}), "http_error"),
        (httpx.Response(400, json={"error": 7}), "http_error"),
        (httpx.Response(429, text="sentinel-body-text"), "http_error"),
        (httpx.Response(500, json={"error": "sentinel-body-text"}), "http_error"),
        (httpx.Response(502, json=["sentinel-body-text"]), "http_error"),
    ],
)
async def test_any_failure_is_tagged_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert "sentinel-body-text" not in str(raised.value) and "sentinel-body-text" not in repr(raised.value)


SENTINEL = "sentinel-body-text"


def _item(name: str, **fields: Any) -> dict[str, Any]:
    return {
        "url": f"https://example.com/{name}",
        "title": f"{name.upper()} {SENTINEL}",
        "content": f"about {name.upper()} {SENTINEL}",
        "publishedDate": None,
        "engine": "wikinews",
    } | fields


def _answer(results: list[Any], *, unresponsive: list[Any] | None = None) -> httpx.Response:
    """A successful answer with every field the adapter reads."""
    body = {"query": "python release", "results": results, "unresponsive_engines": unresponsive or []}
    return httpx.Response(200, json=body)


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    item = _item("a", score=7, positions=[1], parsed_url=["https"], newField={"any": "thing"})
    answer = httpx.Response(
        200,
        json={
            "results": [item],
            "unresponsive_engines": [],
            "infoboxes": [{"id": 7}],
            "answers": "unread",
            "newTopLevel": [],
        },
    )
    result, _ = await _search(answer)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    answer = _answer([_item("a"), {"url": "https://example.com/past-the-count", "title": None}])
    result, _ = await _search(answer, max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "magnet:?xt=urn:btih:0123456789abcdef",
        "/relative",
        "ftp://example.com/a",
        "",
        "https://",
        "http://[::1",
        "https://example.com:99999/a",
        "https://exa mple.com/a",
    ],
)
async def test_a_url_that_is_not_a_web_url_is_dropped_and_the_count_filled_after_it(url: str) -> None:
    answer = _answer([_item("torrent", url=url), _item("b"), _item("c")])
    result, _ = await _search(answer, max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/b", "https://example.com/c"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "published"),
    [
        (None, None),
        ("2026-11-04T09:30:00", datetime(2026, 11, 4, 9, 30, tzinfo=UTC)),
        ("2026-11-04T09:30:00+02:00", datetime(2026, 11, 4, 9, 30, tzinfo=timezone(timedelta(hours=2)))),
        ("2026-11-04", datetime(2026, 11, 4, tzinfo=UTC)),
        ("2 days ago", None),
        ("27.05.2026", None),
    ],
)
async def test_published_date_is_read_as_iso_8601(value: str | None, published: datetime | None) -> None:
    result, _ = await _search(_answer([_item("a", publishedDate=value)]))
    assert result.hits[0].published == published


@pytest.mark.asyncio
async def test_fields_searxng_documents_nowhere_may_be_left_out() -> None:
    item = {"url": "https://example.com/a"}
    result, _ = await _search(httpx.Response(200, json={"results": [item]}))
    assert [(hit.url, hit.title, hit.snippet, hit.published) for hit in result.hits] == [
        ("https://example.com/a", "", "", None)
    ]
    assert result.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [None, "missing"])
async def test_a_result_without_a_url_is_dropped_and_the_count_filled_after_it(url: str | None) -> None:
    first = _item("nothing", url=url)
    if url == "missing":
        del first["url"]
    result, _ = await _search(_answer([first, _item("b"), _item("c")]), max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/b", "https://example.com/c"]


@pytest.mark.asyncio
async def test_published_date_may_be_left_out() -> None:
    item = _item("a")
    del item["publishedDate"]
    result, _ = await _search(_answer([item]))
    assert result.hits[0].published is None


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
        pytest.param(_setting(None, "unresponsive_engines"), id="unresponsive-null"),
        pytest.param(_setting(["an item"], "results"), id="item-not-an-object"),
        pytest.param(_setting(7, *ITEM, "url"), id="url-not-a-string"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_setting(None, *ITEM, "content"), id="content-null"),
        pytest.param(_setting(1730000000, *ITEM, "publishedDate"), id="date-not-a-string"),
        pytest.param(_setting("2026-13-40T09:30:00", *ITEM, "publishedDate"), id="date-iso-shaped-and-impossible"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(change: Any) -> None:
    body = _answer([_item("a")]).json()
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


SENTINEL_QUERY = "sentinel-query-3b8d51"


@asynccontextmanager
async def _local_searxng(*, answers: bool = True) -> AsyncIterator[str]:
    """A local server that answers every request with the recorded search over a real socket, or hangs up."""
    body = json.dumps(_recorded("normal").json()).encode()

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(
            (int(line.split(b":")[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")),
            0,
        )
        await reader.readexactly(length)
        if answers:
            reply = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
            writer.write(reply.encode() + body)
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
@pytest.mark.parametrize("method", ["get", "post"])
async def test_no_log_record_from_httpx_or_httpcore_carries_the_query(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Through a real transport, since ``MockTransport`` skips httpcore, which logs below httpx at DEBUG."""
    caplog.set_level(logging.DEBUG)
    async with _local_searxng() as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with SearxngProvider(api_base=api_base, client=client) as searxng:
            result = await searxng.search(SENTINEL_QUERY, method=method)
    assert result.hits
    messages = [record.getMessage() for record in caplog.records]
    assert {"httpx", "httpcore.http11"} <= {record.name for record in caplog.records}
    assert [message for message in messages if SENTINEL_QUERY in message] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_a_transport_failure_carries_no_query(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    # The server hangs up without an answer, which httpx raises as a transport error.
    async with _local_searxng(answers=False) as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with SearxngProvider(api_base=api_base, client=client) as searxng:
            with pytest.raises(ProviderError) as raised:
                await searxng.search(SENTINEL_QUERY)
    assert raised.value.tag == "network"
    texts = _texts(raised.value) + [record.getMessage() for record in caplog.records]
    assert [text for text in texts if SENTINEL_QUERY in text] == []
