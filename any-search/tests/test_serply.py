"""The Serply adapter: the request it sends, how it reads Serply's recorded answers, and what it logs."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search.providers.serply import MAX_RESULTS, USER_AGENT

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "serply"
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
        async with AnySearch.create("serply", api_key=KEY, api_base=api_base, client=client) as serply:
            result = await serply.search("latest stable python release", **kwargs)
    return result, sent[0]


def _params(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("serply")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "SERPLY_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url) == (True, False)
    assert metadata.max_results == MAX_RESULTS == 10
    assert not [option for option in metadata.options if option.operator_only]
    # What the adapter sets itself is not an option.
    names = {option.name for option in metadata.options}
    assert not names & {"q", "num", "X-Api-Key", "User-Agent"}


@pytest.mark.asyncio
async def test_sends_the_query_as_parameters_of_a_get_and_the_key_as_a_header() -> None:
    _, request = await _search()
    assert (request.method, str(request.url.copy_with(query=None))) == ("GET", "https://api.serply.io/v1/search")
    assert _params(request) == {"q": "latest stable python release", "num": "10"}
    assert (request.headers["X-Api-Key"], request.headers["User-Agent"]) == (KEY, USER_AGENT)
    assert KEY not in str(request.url)
    assert request.content == b""


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/serply/")
    assert str(request.url.copy_with(query=None)) == "https://proxy.example/serply/v1/search"


@pytest.mark.asyncio
async def test_max_results_sets_num_and_cuts_the_hits() -> None:
    result, request = await _search(max_results=2)
    assert _params(request)["num"] == "2"
    assert len(result.hits) == 2


@pytest.mark.asyncio
async def test_max_results_is_capped_at_one_page() -> None:
    _, request = await _search(max_results=50)
    assert _params(request)["num"] == str(MAX_RESULTS)


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", ["q", "num"])
async def test_what_the_adapter_sets_is_not_an_option(reserved: str) -> None:
    options: dict[str, Any] = {reserved: "1"}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


@pytest.mark.asyncio
async def test_query_options_pass_through_as_parameters() -> None:
    options: dict[str, Any] = {"gl": "de", "start": 10, "tbs": "qdr:m", "tbm": "nws"}
    _, request = await _search(**options)
    assert _params(request).items() >= {name: str(value) for name, value in options.items()}.items()


@pytest.mark.asyncio
async def test_header_options_pass_through_as_headers() -> None:
    options: dict[str, Any] = {"X-Proxy-Location": "DE", "X-User-Agent": "mobile"}
    _, request = await _search(**options)
    assert (request.headers["X-Proxy-Location"], request.headers["X-User-Agent"]) == ("DE", "mobile")
    assert "X-Proxy-Location" not in _params(request)


@pytest.mark.asyncio
async def test_an_option_set_to_null_is_not_sent() -> None:
    options: dict[str, Any] = {"gl": None, "start": None, "X-Proxy-Location": None}
    _, request = await _search(**options)
    assert "gl" not in _params(request) and "start" not in _params(request)
    assert "X-Proxy-Location" not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_range", "tbs"), [("day", "qdr:d"), ("week", "qdr:w"), ("month", "qdr:m"), ("year", "qdr:y")]
)
async def test_time_range_is_googles_tbs_and_wins_over_the_option(time_range: str, tbs: str) -> None:
    _, request = await _search(time_range=time_range, tbs="qdr:h")
    assert _params(request)["tbs"] == tbs


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]
    assert [hit.url for hit in result.hits] == [item["link"] for item in recorded]
    first = result.hits[0]
    assert (first.title, first.snippet, first.text, first.published) == (
        recorded[0]["title"],
        recorded[0]["description"],
        None,
        None,
    )
    assert first.raw == recorded[0]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


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
        (httpx.Response(401, json={"detail": "API key missing"}), "missing_api_key"),
        (httpx.Response(429, json={"detail": "sentinel-body-text"}), "http_error"),
        (httpx.Response(500, json={"detail": 7}), "http_error"),
        (httpx.Response(502, text="<html>sentinel-body-text</html>"), "http_error"),
        (httpx.Response(503, json=["sentinel-body-text"]), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_its_detail_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert "sentinel-body-text" not in str(raised.value) and "sentinel-body-text" not in repr(raised.value)


SENTINEL = "sentinel-body-text"


def _item(name: str) -> dict[str, Any]:
    return {
        "title": f"{name.upper()} {SENTINEL}",
        "description": f"about {name.upper()} {SENTINEL}",
        "link": f"https://example.com/{name}",
        "position": 1,
    }


def _answer() -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {"results": [_item("a")]}


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer()
    body["results"][0].update(realPosition="unread", metadata=[7], newField={"any": "thing"})
    body.update(ads=[{"link": 7}], related_searches="unread", newTopLevel=[])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    body = _answer()
    body["results"].append({"link": "https://example.com/past-the-count"})
    result, _ = await _search(httpx.Response(200, json=body), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("link", ["CAESGQoXaHR0cHM6Ly9leGFtcGxlLmNvbS90b2tlbg", "/url?q=x", "ftp://example.com/a", ""])
async def test_a_link_that_is_not_a_web_url_is_dropped_and_the_count_filled_after_it(link: str) -> None:
    body = {"results": [{**_item("token"), "link": link}, _item("b"), _item("c")]}
    result, _ = await _search(httpx.Response(200, json=body), max_results=2)
    assert [hit.url for hit in result.hits] == ["https://example.com/b", "https://example.com/c"]


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
        pytest.param(_without(*ITEM, "link"), id="link-missing"),
        pytest.param(_setting(7, *ITEM, "link"), id="link-not-a-string"),
        pytest.param(_without(*ITEM, "title"), id="title-missing"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_without(*ITEM, "description"), id="description-missing"),
        pytest.param(_setting(None, *ITEM, "description"), id="description-null"),
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
    monkeypatch.delenv("SERPLY_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="SERPLY_API_KEY"):
        AnySearch.create("serply")


SENTINEL_KEY = "sentinel-key-5a1f62"
SENTINEL_QUERY = "sentinel-query-9e4c17"


@asynccontextmanager
async def _local_serply(*, answers: bool = True) -> AsyncIterator[str]:
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
    async with _local_serply() as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("serply", api_key=SENTINEL_KEY, api_base=api_base, client=client) as serply:
            result = await serply.search(SENTINEL_QUERY)
    assert result.hits
    messages = [record.getMessage() for record in caplog.records]
    assert {"httpx", "httpcore.http11"} <= {record.name for record in caplog.records}
    assert [message for message in messages if SENTINEL_KEY in message or SENTINEL_QUERY in message] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_a_transport_failure_carries_neither_the_key_nor_the_query(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    # The server hangs up without an answer, which httpx raises as a transport error.
    async with _local_serply(answers=False) as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("serply", api_key=SENTINEL_KEY, api_base=api_base, client=client) as serply:
            with pytest.raises(ProviderError) as raised:
                await serply.search(SENTINEL_QUERY)
    assert raised.value.tag == "network"
    texts = _texts(raised.value) + [record.getMessage() for record in caplog.records]
    assert [text for text in texts if SENTINEL_KEY in text or SENTINEL_QUERY in text] == []
