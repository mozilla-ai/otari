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


def _page(**fields: Any) -> httpx.Response:
    """A successful answer for one page."""
    return httpx.Response(
        200, json={"results": [{"url": URL, **fields}], "statuses": [{"id": URL, "status": "success"}]}
    )


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
    assert (page.title, page.cost, page.cost_source) == ("", None, "none")


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
@pytest.mark.parametrize("statuses", [[], [{"id": URL, "status": "success"}], None])
async def test_no_page_and_no_reason_is_an_error_not_an_empty_page(statuses: Any) -> None:
    page, _ = await _fetch(httpx.Response(200, json={"results": [], "statuses": statuses}))
    assert page.error == FetchError(tag="no_result")


@pytest.mark.asyncio
async def test_a_page_with_no_text_is_empty_not_an_error() -> None:
    page, _ = await _fetch(_recorded("empty"))
    assert (page.text, page.error, page.text_truncated) == ("", None, False)


@pytest.mark.asyncio
async def test_a_page_error_without_a_tag_still_counts() -> None:
    body = {"results": [], "statuses": [{"id": URL, "status": "error", "error": {}}]}
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert page.error == FetchError(tag="fetch_error", status=None)


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
    [httpx.Response(200, text="not json"), httpx.Response(200, json=[]), httpx.Response(200, json={"statuses": []})],
)
async def test_a_body_that_is_not_a_contents_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="EXA_API_KEY"):
        AnyFetch.create("exa")
