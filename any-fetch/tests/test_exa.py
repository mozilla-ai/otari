"""The Exa contents adapter: the request it sends, and how it reads Exa's recorded answers."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_fetch import (
    AnyFetch,
    FetchedPage,
    FetchError,
    MissingCredentialError,
    ProviderError,
    UnsupportedParameterError,
)
from any_fetch.providers.exa import DEFAULT_MAX_CHARS, MAX_CHARACTERS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "exa"
KEY = "test-key"
URL = "https://docs.python.org/3/whatsnew/3.14.html"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    return httpx.Response(recorded["status"], json=recorded["body"])


async def _fetch(
    answer: httpx.Response | None = None, *, api_base: str | None = None, **kwargs: Any
) -> tuple[FetchedPage, httpx.Request]:
    """Fetch through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnyFetch.create("exa", api_key=KEY, api_base=api_base, client=client) as exa:
            page = await exa.fetch(URL, **kwargs)
    return page, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


SENTINEL = "sentinel-body-text"


def _answer(**fields: Any) -> dict[str, Any]:
    """A successful answer for one page, with every field the adapter reads, and a sentinel no error may carry."""
    return {
        "results": [{"url": URL, "title": f"Title {SENTINEL}", "text": f"page {SENTINEL}", **fields}],
        "statuses": [{"id": URL, "status": "success"}],
        "costDollars": {"total": 0.001},
    }


def _page(**fields: Any) -> httpx.Response:
    return httpx.Response(200, json=_answer(**fields))


def _failed_page(error: Any) -> dict[str, Any]:
    """A successful answer that reports the page as not fetched, with ``error`` as Exa's reason."""
    return {"results": [], "statuses": [{"id": URL, "status": "error", "error": error}], "costDollars": {"total": 0}}


def test_metadata() -> None:
    metadata = AnyFetch.get_provider_metadata("exa")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "EXA_API_KEY", True)
    assert metadata.formats == ["markdown"]
    assert not [option for option in metadata.options if option.operator_only]


@pytest.mark.asyncio
async def test_asks_for_the_text_one_character_past_the_sdks_default_cap() -> None:
    _, request = await _fetch()
    assert (request.method, str(request.url)) == ("POST", "https://api.exa.ai/contents")
    assert request.headers["x-api-key"] == KEY
    assert _payload(request) == {"urls": [URL], "text": {"maxCharacters": DEFAULT_MAX_CHARS + 1}}


@pytest.mark.asyncio
async def test_max_chars_sets_the_cap_and_text_options_join_it() -> None:
    _, request = await _fetch(max_chars=600, text={"verbosity": "full"}, maxAgeHours=0, highlights={"query": "news"})
    assert _payload(request) == {
        "urls": [URL],
        "text": {"verbosity": "full", "maxCharacters": 601},
        "maxAgeHours": 0,
        "highlights": {"query": "news"},
    }


@pytest.mark.asyncio
async def test_the_cap_asked_for_stays_within_exas_limit() -> None:
    _, request = await _fetch(max_chars=MAX_CHARACTERS)
    assert _payload(request)["text"] == {"maxCharacters": MAX_CHARACTERS}


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _fetch(api_base="https://proxy.example/exa/")
    assert str(request.url) == "https://proxy.example/exa/contents"


@pytest.mark.asyncio
async def test_the_text_length_is_max_chars_only() -> None:
    with pytest.raises(UnsupportedParameterError, match="text.maxCharacters"):
        await _fetch(text={"maxCharacters": 50})


@pytest.mark.asyncio
async def test_text_true_is_exas_defaults_and_another_value_is_refused() -> None:
    _, request = await _fetch(text=True, max_chars=5)
    assert _payload(request)["text"] == {"maxCharacters": 6}
    for value in (False, "full"):
        with pytest.raises(ProviderError) as raised:
            await _fetch(text=value)
        assert raised.value.tag == "invalid_option"


@pytest.mark.asyncio
async def test_reads_the_recorded_answer_and_cuts_the_text_at_max_chars() -> None:
    page, _ = await _fetch(max_chars=600)
    recorded = _recorded("normal").json()
    item = recorded["results"][0]
    assert (page.url, page.final_url, page.title) == (URL, item["url"], item["title"])
    # Exa returned the 601 characters asked for: one more than max_chars, so the page was longer.
    assert len(item["text"]) == 601
    assert (page.text, page.text_truncated) == (item["text"][:600], True)
    assert (page.content_type, page.published, page.error) == ("", None, None)
    assert (page.cost, page.cost_source) == (Decimal("0.001"), "reported")
    assert page.raw == recorded


@pytest.mark.asyncio
async def test_a_page_within_the_cap_is_not_marked_truncated() -> None:
    page, _ = await _fetch(_page(text="short", publishedDate="2026-02-03"), max_chars=5)
    assert (page.text, page.text_truncated) == ("short", False)
    assert page.published == datetime(2026, 2, 3, tzinfo=UTC)
    assert (page.title, page.cost, page.cost_source) == (f"Title {SENTINEL}", Decimal("0.001"), "reported")


@pytest.mark.asyncio
async def test_a_page_exa_could_not_fetch_is_an_error_inside_the_answer() -> None:
    page, _ = await _fetch(_recorded("in_body_error"))
    assert page.error == FetchError(tag="CRAWL_NOT_FOUND", status=404)
    assert (page.text, page.final_url) == ("", URL)
    assert (page.cost, page.cost_source) == (Decimal(0), "reported")


@pytest.mark.asyncio
async def test_text_that_fills_exas_own_limit_counts_as_cut() -> None:
    page, _ = await _fetch(_page(text="x" * MAX_CHARACTERS), max_chars=MAX_CHARACTERS)
    assert (len(page.text), page.text_truncated) == (MAX_CHARACTERS, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [None, []], ids=["absent", "empty"])
async def test_without_statuses_the_page_in_results_is_the_answer(statuses: Any) -> None:
    """Exa documents statuses as optional."""
    body = _answer()
    if statuses is None:
        del body["statuses"]
    else:
        body["statuses"] = statuses
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.title, page.error) == (f"Title {SENTINEL}", None)

    body["results"] = []
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(200, json=body))
    # No page and no reason.
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
async def test_a_success_without_text_is_an_empty_page() -> None:
    """Exa documents a result's text as optional."""
    body = _answer()
    del body["results"][0]["text"]
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.text, page.title, page.error, page.text_truncated) == ("", f"Title {SENTINEL}", None, False)


@pytest.mark.asyncio
async def test_a_page_with_no_text_is_empty_not_an_error() -> None:
    page, _ = await _fetch(_recorded("empty"))
    assert (page.text, page.error, page.text_truncated) == ("", None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("total", ["NaN", "Infinity"])
async def test_a_cost_that_is_not_a_finite_number_raises(total: str) -> None:
    """Python reads both from JSON, and neither is a cost."""
    body = json.dumps(_answer()).replace('"total": 0.001', f'"total": {total}')
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(200, content=body, headers={"content-type": "application/json"}))
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
async def test_a_page_error_with_a_null_status_code_has_none() -> None:
    page, _ = await _fetch(httpx.Response(200, json=_failed_page({"tag": "CRAWL_NOT_FOUND", "httpStatusCode": None})))
    assert page.error == FetchError(tag="CRAWL_NOT_FOUND", status=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [{}, None])
async def test_a_page_error_without_a_tag_still_counts(error: Any) -> None:
    page, _ = await _fetch(httpx.Response(200, json=_failed_page(error)))
    assert page.error == FetchError(tag="fetch_error", status=None)


@pytest.mark.asyncio
async def test_the_status_for_the_url_is_the_one_read() -> None:
    body = _failed_page({"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404})
    body["statuses"].insert(0, {"id": "https://elsewhere.example/", "status": "success"})
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert page.error == FetchError(tag="CRAWL_NOT_FOUND", status=404)


@pytest.mark.asyncio
@pytest.mark.parametrize("cost", [{"contents": {"text": 0.001}}, None])
async def test_an_answer_without_a_cost_reports_none(cost: Any) -> None:
    """Exa documents the cost as optional; the live tests are what notice it gone."""
    body = _answer()
    if cost is None:
        del body["costDollars"]
    else:
        body["costDollars"] = cost
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.cost, page.cost_source, page.error) == (None, "none", None)


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer(author=None, favicon=7, newField={"any": "thing"})
    body["statuses"][0].update(source="cached", newField=[])
    body["costDollars"]["contents"] = "unread"
    body["newTopLevel"] = []
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.title, page.published, page.error) == (f"Title {SENTINEL}", None, None)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("results"), id="results-missing"),
        pytest.param(_setting({}, "results"), id="results-not-a-list"),
        pytest.param(_setting(None, "statuses"), id="statuses-null"),
        pytest.param(_setting(["success"], "statuses"), id="status-not-an-object"),
        pytest.param(_without("statuses", 0, "id"), id="status-id-missing"),
        pytest.param(_setting(f"done {SENTINEL}", "statuses", 0, "status"), id="status-unknown"),
        pytest.param(_without("statuses", 0, "status"), id="status-missing"),
        pytest.param(
            _setting(
                [{"id": "https://a.example", "status": "success"}, {"id": "https://b.example", "status": "success"}],
                "statuses",
            ),
            id="several-statuses-none-for-the-url",
        ),
        pytest.param(_setting([], "results"), id="success-with-no-page"),
        pytest.param(_setting(["a page"], "results"), id="page-not-an-object"),
        pytest.param(_without("results", 0, "url"), id="url-missing"),
        pytest.param(_setting("", "results", 0, "url"), id="url-empty"),
        pytest.param(_setting(7, "results", 0, "url"), id="url-not-a-string"),
        pytest.param(_without("results", 0, "title"), id="title-missing"),
        pytest.param(_setting(None, "results", 0, "title"), id="title-null"),
        pytest.param(_setting(["Title"], "results", 0, "title"), id="title-not-a-string"),
        pytest.param(_setting(None, "results", 0, "text"), id="text-null"),
        pytest.param(_setting(["page"], "results", 0, "text"), id="text-not-a-string"),
        pytest.param(_setting(None, "results", 0, "publishedDate"), id="date-null"),
        pytest.param(_setting(f"not a date {SENTINEL}", "results", 0, "publishedDate"), id="date-unparsable"),
        pytest.param(_setting(0.001, "costDollars"), id="cost-not-an-object"),
        pytest.param(_setting("0.001", "costDollars", "total"), id="total-a-string"),
        pytest.param(_setting(True, "costDollars", "total"), id="total-a-boolean"),
        pytest.param(_setting(-0.001, "costDollars", "total"), id="total-negative"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(change: Any) -> None:
    body = _answer()
    change(body)
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(200, json=body))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(f"failed {SENTINEL}", id="error-not-an-object"),
        pytest.param({"tag": 404}, id="tag-not-a-string"),
        pytest.param({"tag": "CRAWL_NOT_FOUND", "httpStatusCode": "404"}, id="status-code-a-string"),
        pytest.param({"tag": "CRAWL_NOT_FOUND", "httpStatusCode": True}, id="status-code-a-boolean"),
    ],
)
async def test_a_page_error_of_the_wrong_shape_raises_without_the_body(error: Any) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(200, json=_failed_page(error)))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value)


@pytest.mark.asyncio
async def test_a_failure_raises_with_exas_status_and_tag() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (401, "INVALID_API_KEY")
    assert "Invalid API key" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(503, text="<html>overloaded</html>"),
        httpx.Response(500, json={"tag": "not a tag: it has spaces"}),
    ],
)
async def test_a_failure_without_a_usable_tag_is_an_http_error(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, "http_error")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="not json"), httpx.Response(200, json=[])],
)
async def test_a_body_that_is_not_a_contents_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="EXA_API_KEY"):
        AnyFetch.create("exa")
