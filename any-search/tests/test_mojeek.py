"""The Mojeek adapter: the request it sends, how it reads Mojeek's recorded answers, and what it logs."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult, UnsupportedParameterError
from any_search.providers.mojeek import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mojeek"
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
        async with AnySearch.create("mojeek", api_key=KEY, api_base=api_base, client=client) as mojeek:
            result = await mojeek.search("latest stable python release", **kwargs)
    return result, sent[0]


def _params(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("mojeek")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "MOJEEK_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url) == (True, True)
    assert metadata.max_results == MAX_RESULTS == 10
    assert not [option for option in metadata.options if option.operator_only]
    # What the adapter sets itself, and OrgSearch's own parameters, are not options.
    names = {option.name for option in metadata.options}
    assert not names & {"q", "api_key", "t", "fmt", "categories", "num_ref_cats", "facet_date_gap"}


@pytest.mark.asyncio
async def test_sends_the_query_and_the_key_as_parameters_of_a_get() -> None:
    _, request = await _search()
    assert (request.method, str(request.url.copy_with(query=None))) == ("GET", "https://api.mojeek.com/search")
    assert _params(request) == {
        "q": "latest stable python release",
        "t": "10",
        "fmt": "json",
        "date": "1",
        "api_key": KEY,
    }
    assert request.content == b""


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/mojeek/")
    assert str(request.url.copy_with(query=None)) == "https://proxy.example/mojeek/search"


@pytest.mark.asyncio
async def test_max_results_sets_t_and_cuts_the_hits() -> None:
    result, request = await _search(max_results=2)
    assert _params(request)["t"] == "2"
    assert len(result.hits) == 2


@pytest.mark.asyncio
async def test_max_results_is_capped_at_the_entry_plans_limit() -> None:
    _, request = await _search(max_results=50)
    assert _params(request)["t"] == str(MAX_RESULTS)


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", ["q", "api_key", "t", "fmt"])
async def test_what_the_adapter_sets_is_not_an_option(reserved: str) -> None:
    options: dict[str, Any] = {reserved: "1"}
    with pytest.raises(UnsupportedParameterError):
        await _search(**options)


@pytest.mark.asyncio
async def test_native_options_pass_through_under_their_own_names() -> None:
    options: dict[str, Any] = {"site": "python.org", "lb": "EN", "lbb": 100, "dlen": 300, "fe": ".example.com"}
    _, request = await _search(**options)
    assert _params(request).items() >= {name: str(value) for name, value in options.items()}.items()


@pytest.mark.asyncio
async def test_booleans_are_sent_as_mojeeks_zero_and_one() -> None:
    _, request = await _search(safe=True, date=False, size=True)
    assert {name: _params(request)[name] for name in ("safe", "date", "size")} == {
        "safe": "1",
        "date": "0",
        "size": "1",
    }


@pytest.mark.asyncio
async def test_an_option_set_to_null_is_not_sent() -> None:
    _, request = await _search(date=None, site=None)
    assert "date" not in _params(request) and "site" not in _params(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("time_range", ["day", "month", "year"])
async def test_time_range_is_mojeeks_own_word_and_wins_over_since(time_range: str) -> None:
    _, request = await _search(time_range=time_range, since="20000101")
    assert _params(request)["since"] == time_range


@pytest.mark.asyncio
async def test_a_week_is_the_date_it_starts_on() -> None:
    """Mojeek has no word for a week, and takes a date as YYYYMMDD."""
    before = datetime.now(UTC) - timedelta(weeks=1)
    _, request = await _search(time_range="week")
    after = datetime.now(UTC) - timedelta(weeks=1)
    # Either day, should the search run across midnight.
    assert _params(request)["since"] in {before.strftime("%Y%m%d"), after.strftime("%Y%m%d")}


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["response"]["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    first, second, _ = result.hits
    assert (first.title, first.snippet, first.text) == (recorded[0]["title"], recorded[0]["desc"], None)
    assert first.published == datetime.fromtimestamp(recorded[0]["timestamp"], UTC)
    assert first.raw == recorded[0]
    # Mojeek answered this one with an empty description and a timestamp of 0, its "unknown".
    assert (second.snippet, second.published) == ("", None)
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_a_refused_key_is_an_error_inside_the_answer() -> None:
    result, _ = await _search(_recorded("in_body_error"))
    assert result.hits == []
    assert result.error is not None
    # Mojeek's status is inside a 200, which is no failure status.
    assert (result.error.tag, result.error.status) == ("access_denied", None)
    assert result.raw == _recorded("in_body_error").json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "tag"),
    [
        ("ERROR: Daily Limit Reached", "daily_limit_reached"),
        ("Access Denied: something else", "access_denied"),
        ("ERROR: sentinel-body-text", "error"),
        ("", "error"),
    ],
)
async def test_any_status_but_ok_is_tagged_without_its_text(status: str, tag: str) -> None:
    body = {"response": {"status": status, "results": [{"url": 7}]}}
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.error is not None
    assert (result.hits, result.error.tag) == ([], tag)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["OK ", "ok"])
async def test_an_ok_status_is_read_whatever_its_spacing_or_case(status: str) -> None:
    body = _recorded("normal").json()
    body["response"]["status"] = status
    result, _ = await _search(httpx.Response(200, json=body))
    assert (len(result.hits), result.error) == (3, None)


SENTINEL = "sentinel-body-text"


def _answer() -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {
        "response": {
            "status": "OK",
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": f"A {SENTINEL}",
                    "desc": f"about A {SENTINEL}",
                    "pdate": 1770000000,
                    "timestamp": 1780000000,
                }
            ],
        }
    }


@pytest.mark.asyncio
async def test_pdate_wins_over_timestamp() -> None:
    result, _ = await _search(httpx.Response(200, json=_answer()))
    assert result.hits[0].published == datetime.fromtimestamp(1770000000, UTC)


@pytest.mark.asyncio
async def test_a_timestamp_behind_a_usable_pdate_is_not_read() -> None:
    """Only the fields the adapter reads are checked (decision 64), as an Exa item past the count is not."""
    body = _answer()
    body["response"]["results"][0]["timestamp"] = "not a time"
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.hits[0].published == datetime.fromtimestamp(1770000000, UTC)


@pytest.mark.asyncio
async def test_a_zero_pdate_falls_back_to_timestamp() -> None:
    body = _answer()
    body["response"]["results"][0]["pdate"] = 0
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.hits[0].published == datetime.fromtimestamp(1780000000, UTC)


@pytest.mark.asyncio
async def test_the_dates_may_be_absent_and_unread_fields_are_ignored() -> None:
    body = _answer()
    item = body["response"]["results"][0]
    del item["pdate"], item["timestamp"]
    item.update(image={"url": 7}, score="unread", newField={"any": "thing"})
    body["response"]["head"] = "unread"
    body["newTopLevel"] = []
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.hits[0].published is None


@pytest.mark.asyncio
async def test_only_the_items_within_the_count_are_read() -> None:
    body = _answer()
    body["response"]["results"].append({"url": "https://example.com/past-the-count"})
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


def _then(*changes: Any) -> Any:
    """One change made of several, in order."""

    def change(body: dict[str, Any]) -> None:
        for each in changes:
            each(body)

    return change


ITEM = ("response", "results", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("response"), id="response-missing"),
        pytest.param(_setting([], "response"), id="response-not-an-object"),
        pytest.param(_without("response", "status"), id="status-missing"),
        pytest.param(_setting(None, "response", "status"), id="status-null"),
        pytest.param(_without("response", "results"), id="results-missing"),
        pytest.param(_setting({}, "response", "results"), id="results-not-a-list"),
        pytest.param(_setting(["an item"], "response", "results"), id="item-not-an-object"),
        pytest.param(_without(*ITEM, "url"), id="url-missing"),
        pytest.param(_setting("", *ITEM, "url"), id="url-empty"),
        pytest.param(_setting(7, *ITEM, "url"), id="url-not-a-string"),
        pytest.param(_without(*ITEM, "title"), id="title-missing"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_without(*ITEM, "desc"), id="desc-missing"),
        pytest.param(_setting(None, *ITEM, "desc"), id="desc-null"),
        pytest.param(_setting(f"2026 {SENTINEL}", *ITEM, "pdate"), id="pdate-a-string"),
        pytest.param(_setting(True, *ITEM, "pdate"), id="pdate-a-boolean"),
        pytest.param(_then(_without(*ITEM, "pdate"), _setting(None, *ITEM, "timestamp")), id="timestamp-null"),
        pytest.param(_setting(10**20, *ITEM, "pdate"), id="pdate-out-of-range"),
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
async def test_a_failure_raises_with_its_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (503, "http_error")
    assert "Send search failed" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="<response><status>OK</status></response>"), httpx.Response(200, json=[])],
)
async def test_a_body_that_is_not_a_search_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOJEEK_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="MOJEEK_API_KEY"):
        AnySearch.create("mojeek")


SENTINEL_KEY = "sentinel-key-7c3e91"
SENTINEL_QUERY = "sentinel-query-2b8d40"


@asynccontextmanager
async def _local_mojeek(*, answers: bool = True) -> AsyncIterator[str]:
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
    async with _local_mojeek() as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("mojeek", api_key=SENTINEL_KEY, api_base=api_base, client=client) as mojeek:
            result = await mojeek.search(SENTINEL_QUERY)
    assert result.hits
    messages = [record.getMessage() for record in caplog.records]
    assert {"httpx", "httpcore.http11"} <= {record.name for record in caplog.records}
    assert [message for message in messages if SENTINEL_KEY in message or SENTINEL_QUERY in message] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_a_transport_failure_carries_neither_the_key_nor_the_query(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    # The server hangs up without an answer, which httpx raises as a transport error.
    async with _local_mojeek(answers=False) as api_base, httpx.AsyncClient(trust_env=False) as client:
        async with AnySearch.create("mojeek", api_key=SENTINEL_KEY, api_base=api_base, client=client) as mojeek:
            with pytest.raises(ProviderError) as raised:
                await mojeek.search(SENTINEL_QUERY)
    assert raised.value.tag == "network"
    texts = _texts(raised.value) + [record.getMessage() for record in caplog.records]
    assert [text for text in texts if SENTINEL_KEY in text or SENTINEL_QUERY in text] == []
