"""The SearchApi adapter: the request it sends, how it reads SearchApi's recorded answers, and what it logs."""

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
from any_search._http import USER_AGENT
from any_search._types import SearchError
from any_search.providers.searchapi import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "searchapi"
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
        async with AnySearch.create("searchapi", api_key=KEY, api_base=api_base, client=client) as searchapi:
            result = await searchapi.search("latest stable python release", **kwargs)
    return result, sent[0]


def _params(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("searchapi")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "SEARCHAPI_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (True, False, False)
    assert metadata.max_results == MAX_RESULTS == 10
    assert not [option for option in metadata.options if option.operator_only or option.cost_tier]
    # What the adapter, or the library, sets itself is not an option.
    names = {option.name for option in metadata.options}
    assert not names & {"engine", "q", "num", "api_key", "Authorization", "User-Agent"}


@pytest.mark.asyncio
async def test_sends_the_query_as_parameters_of_a_get_and_the_key_as_a_bearer_token() -> None:
    _, request = await _search()
    assert (request.method, str(request.url.copy_with(query=None))) == ("GET", "https://www.searchapi.io/api/v1/search")
    assert _params(request) == {"engine": "google", "q": "latest stable python release", "num": "10"}
    assert (request.headers["Authorization"], request.headers["User-Agent"]) == (f"Bearer {KEY}", USER_AGENT)
    assert KEY not in str(request.url)
    assert request.content == b""


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/searchapi/")
    assert str(request.url.copy_with(query=None)) == "https://proxy.example/searchapi/api/v1/search"


@pytest.mark.asyncio
async def test_max_results_sets_num_and_cuts_the_hits() -> None:
    answer = httpx.Response(200, json=_answer([_item("a"), _item("b"), _item("c")]))
    result, request = await _search(answer, max_results=2)
    assert _params(request)["num"] == "2"
    assert [hit.url for hit in result.hits] == ["https://example.com/a", "https://example.com/b"]


@pytest.mark.asyncio
async def test_max_results_is_capped_at_one_page() -> None:
    answer = httpx.Response(200, json=_answer([_item(str(i)) for i in range(MAX_RESULTS + 1)]))
    result, request = await _search(answer, max_results=50)
    assert _params(request)["num"] == str(MAX_RESULTS)
    assert len(result.hits) == MAX_RESULTS


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", ["engine", "q", "num", "api_key"])
async def test_what_the_adapter_sets_is_not_an_option(reserved: str) -> None:
    options: dict[str, Any] = {reserved: "1"}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


@pytest.mark.asyncio
async def test_query_options_pass_through_as_parameters() -> None:
    options: dict[str, Any] = {
        "gl": "de",
        "hl": "de",
        "location": "Berlin, Germany",
        "device": "mobile",
        "safe": "active",
        "page": 2,
        "time_period": "last_hour",
        "time_period_min": "01/01/2026",
        "time_period_max": "06/30/2026",
        "lr": "lang_de",
        "cr": "countryDE",
        "nfpr": 1,
        "filter": 0,
    }
    _, request = await _search(**options)
    assert _params(request).items() >= {name: str(value) for name, value in options.items()}.items()
    assert _params(request)["engine"] == "google"


@pytest.mark.asyncio
async def test_an_option_set_to_null_is_not_sent() -> None:
    options: dict[str, Any] = {"gl": None, "page": None, "time_period": None}
    _, request = await _search(**options)
    assert not {"gl", "page", "time_period"} & _params(request).keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_range", "time_period"),
    [("day", "last_day"), ("week", "last_week"), ("month", "last_month"), ("year", "last_year")],
)
async def test_time_range_is_a_time_period_and_wins_over_the_option(time_range: str, time_period: str) -> None:
    _, request = await _search(time_range=time_range, time_period="last_hour")
    assert _params(request)["time_period"] == time_period


@pytest.mark.asyncio
async def test_time_range_drops_the_min_and_max_options() -> None:
    _, request = await _search(time_range="week", time_period_min="01/01/2024", time_period_max="12/31/2024")
    params = _params(request)
    assert params["time_period"] == "last_week"
    assert "time_period_min" not in params and "time_period_max" not in params


@pytest.mark.asyncio
async def test_without_time_range_the_min_and_max_options_are_sent() -> None:
    _, request = await _search(time_period_min="01/01/2024", time_period_max="12/31/2024")
    params = _params(request)
    assert (params["time_period_min"], params["time_period_max"]) == ("01/01/2024", "12/31/2024")


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["organic_results"]
    assert [hit.url for hit in result.hits] == [item["link"] for item in recorded]
    first = result.hits[0]
    assert (first.title, first.snippet, first.text, first.published) == (
        recorded[0]["title"],
        recorded[0]["snippet"],
        None,
        None,
    )
    assert first.raw == recorded[0]
    # A result's date is display text, such as "7 days ago", and stays in raw.
    assert [hit.published for hit in result.hits] == [None] * len(recorded)
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_a_google_redirect_with_a_token_is_dropped_and_the_count_filled() -> None:
    # The recorded shape: ``url`` holds a session token, not an address, so no caller can fetch it.
    redirect = "https://www.google.com/goto?url=CAESYQHEIkXXlH8uFHDbd9xJPhZVAePTbLr4rkOYo6nG4kNO&gsessionid=abc"
    body = _answer([{**_item("a"), "link": redirect, "link_token": "ABC123"}, _item("b")])
    result, _ = await _search(httpx.Response(200, json=body), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/b"]


@pytest.mark.asyncio
async def test_a_google_redirect_to_a_url_is_unwrapped() -> None:
    redirect = "https://www.google.com/goto?url=https%3A%2F%2Fexample.com%2Fa"
    body = _answer([{**_item("a"), "link": redirect, "link_token": "ABC123"}])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]
    assert (result.hits[0].raw["link"], result.hits[0].raw["link_token"]) == (redirect, "ABC123")


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)
    # SearchApi's sentence for a search that found nothing stays in raw.
    assert result.raw["error"] == "Google didn't return any results."


@pytest.mark.asyncio
async def test_another_error_sentence_in_a_200_without_results_is_the_result_error() -> None:
    body = {"search_metadata": {"status": "Success"}, "error": f"Something else went wrong {SENTINEL}"}
    result, _ = await _search(httpx.Response(200, json=body))
    assert (result.hits, result.error) == ([], SearchError(tag="provider_error"))
    assert result.raw["error"].endswith(SENTINEL)


@pytest.mark.asyncio
async def test_an_answer_with_other_sections_and_no_organic_results_has_no_hits() -> None:
    body = {"search_metadata": {"status": "Success"}, "top_stories": [{"title": SENTINEL, "link": "https://a.example"}]}
    result, _ = await _search(httpx.Response(200, json=body))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_status_other_than_success_is_the_results_error_without_its_text() -> None:
    body = _answer([_item("a")])
    body["search_metadata"]["status"] = f"Error: {SENTINEL}"
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.hits == []
    assert result.error is not None
    assert (result.error.tag, result.error.status) == ("error", None)
    assert SENTINEL not in repr(result.error)
    body["search_metadata"]["status"] = "Processing"
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.error is not None
    assert result.error.tag == "processing"


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
            httpx.Response(400, json={"error": "Unsupported value `sentinel-body-text` in time_period parameter."}),
            "invalid_request",
        ),
        (httpx.Response(402, json={"error": "sentinel-body-text"}), "payment_required"),
        (httpx.Response(429, json={"error": "sentinel-body-text"}), "rate_limited"),
        (httpx.Response(500, json={"error": 7}), "http_error"),
        (httpx.Response(502, text="<html>sentinel-body-text</html>"), "http_error"),
        (httpx.Response(503, json=["sentinel-body-text"]), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_its_status_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert "sentinel-body-text" not in str(raised.value) and "sentinel-body-text" not in repr(raised.value)


SENTINEL = "sentinel-body-text"


def _item(name: str) -> dict[str, Any]:
    return {
        "position": 1,
        "title": f"{name.upper()} {SENTINEL}",
        "link": f"https://example.com/{name}",
        "source": name,
        "domain": "example.com",
        "displayed_link": f"https://example.com › {name}",
        "snippet": f"about {name.upper()} {SENTINEL}",
    }


def _answer(items: list[dict[str, Any]]) -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {"search_metadata": {"id": "search_sentinel", "status": "Success"}, "organic_results": items}


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer([_item("a")])
    body["organic_results"][0].update(date="7 days ago", favicon="unread", rich_snippet={"any": 7}, newField=[1])
    body.update(ai_overview={"text": 7}, knowledge_graph="unread", top_stories=[{"link": 7}], newTopLevel=[])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_an_absent_snippet_is_empty() -> None:
    without = _item("a")
    del without["snippet"]
    result, _ = await _search(httpx.Response(200, json=_answer([without])))
    assert [hit.snippet for hit in result.hits] == [""]


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    body = _answer([_item("a"), {"link": "https://example.com/past-the-count"}])
    result, _ = await _search(httpx.Response(200, json=body), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link",
    [
        "/url?q=x",
        "ftp://example.com/a",
        "",
        "https://",
        "http:///a",
        "http://[::1",
        "https://example.com:not-a-port/a",
        "https://example.com:99999/a",
        "https://exa mple.com/a",
    ],
)
async def test_a_link_that_is_not_a_web_url_is_dropped_and_the_count_filled_after_it(link: str) -> None:
    body = _answer([{**_item("token"), "link": link}, _item("b"), _item("c")])
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


ITEM = ("organic_results", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("search_metadata"), id="metadata-missing"),
        pytest.param(_setting("Success", "search_metadata"), id="metadata-not-an-object"),
        pytest.param(_without("search_metadata", "status"), id="status-missing"),
        pytest.param(_setting(7, "search_metadata", "status"), id="status-not-a-string"),
        pytest.param(_setting(None, "organic_results"), id="results-null"),
        pytest.param(_setting({}, "organic_results"), id="results-not-a-list"),
        pytest.param(_setting("none", "organic_results"), id="results-a-string"),
        pytest.param(_setting(["an item"], "organic_results"), id="item-not-an-object"),
        pytest.param(_without(*ITEM, "link"), id="link-missing"),
        pytest.param(_setting(7, *ITEM, "link"), id="link-not-a-string"),
        pytest.param(_setting(None, *ITEM, "link"), id="link-null"),
        pytest.param(_without(*ITEM, "title"), id="title-missing"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_setting(["a"], *ITEM, "snippet"), id="snippet-not-a-string"),
        pytest.param(_setting(None, *ITEM, "snippet"), id="snippet-null"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(change: Any) -> None:
    body = _answer([_item("a")])
    change(body)
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [{"message": SENTINEL}, None, 7], ids=["an-object", "null", "a-number"])
async def test_an_error_that_is_not_a_sentence_in_a_200_without_results_raises_without_the_body(error: Any) -> None:
    body = {"search_metadata": {"status": "Success"}, "error": error}
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
    monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="SEARCHAPI_API_KEY"):
        AnySearch.create("searchapi")


SENTINEL_KEY = "sentinel-key-5a1f62"
SENTINEL_QUERY = "sentinel-query-9e4c17"


@asynccontextmanager
async def _local_searchapi(*, answers: bool = True) -> AsyncIterator[str]:
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
    async with _local_searchapi() as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("searchapi", api_key=SENTINEL_KEY, api_base=api_base, client=client) as engine:
            result = await engine.search(SENTINEL_QUERY)
    assert result.hits
    messages = [record.getMessage() for record in caplog.records]
    assert {"httpx", "httpcore.http11"} <= {record.name for record in caplog.records}
    assert [message for message in messages if SENTINEL_KEY in message or SENTINEL_QUERY in message] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_a_transport_failure_carries_neither_the_key_nor_the_query(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    # The server hangs up without an answer, which httpx raises as a transport error.
    async with _local_searchapi(answers=False) as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("searchapi", api_key=SENTINEL_KEY, api_base=api_base, client=client) as engine:
            with pytest.raises(ProviderError) as raised:
                await engine.search(SENTINEL_QUERY)
    assert raised.value.tag == "network"
    texts = _texts(raised.value) + [record.getMessage() for record in caplog.records]
    assert [text for text in texts if SENTINEL_KEY in text or SENTINEL_QUERY in text] == []
