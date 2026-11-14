"""The You.com Extract adapter: the request it sends, how it reads recorded answers, and the fields it checks."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_fetch import AnyFetch, FetchedPage, MissingCredentialError, ProviderError, UnsupportedParameterError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "youdotcom"
KEY = "test-key"
URL = "https://example.com/"
SENTINEL = "sentinel-body-text"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    body = recorded["body"]
    if isinstance(body, str):
        return httpx.Response(recorded["status"], text=body)
    return httpx.Response(recorded["status"], json=body)


async def _fetch(
    answer: httpx.Response | None = None, *, api_base: str | None = None, **kwargs: Any
) -> tuple[FetchedPage, httpx.Request]:
    """Fetch through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnyFetch.create("youdotcom", api_key=KEY, api_base=api_base, client=client) as youdotcom:
            page = await youdotcom.fetch(URL, **kwargs)
    return page, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


def _answer(**fields: Any) -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    body: dict[str, Any] = {
        "url": URL,
        "extraction_source": "blend",
        "contents": {"full_page": {"markdown": f"# Page\n\n{SENTINEL} text."}},
        "title": f"Title {SENTINEL}",
        "metadata": {"site_name": "", "favicon_url": "https://api.ydc-index.io/favicon?domain=example.com&size=128"},
    }
    body.update(fields)
    return body


def test_metadata() -> None:
    metadata = AnyFetch.get_provider_metadata("youdotcom")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "YDC_API_KEY", True)
    assert (metadata.default_api_base, metadata.env_api_base) == ("https://ydc-index.io", None)
    assert (metadata.max_urls_per_call, metadata.renders_javascript, metadata.formats) == (1, False, ["markdown"])
    assert (metadata.reports_final_url, metadata.reports_cost) == (False, False)
    assert [option.name for option in metadata.options] == ["full_page", "extraction_source", "timeout", "parse_effort"]
    assert not [option for option in metadata.options if option.operator_only]


def test_parse_effort_is_the_cost_tier_and_lite_is_its_default() -> None:
    [tier] = [option for option in AnyFetch.get_provider_metadata("youdotcom").options if option.cost_tier]
    assert (tier.name, tier.enum, tier.default) == ("parse_effort", ["lite", "standard"], "lite")


@pytest.mark.asyncio
async def test_sends_the_url_and_lite_parse_effort_with_the_key_in_the_x_api_key_header() -> None:
    _, request = await _fetch()
    assert (request.method, str(request.url)) == ("POST", "https://ydc-index.io/v1/extract")
    assert request.headers["X-API-Key"] == KEY
    assert "authorization" not in request.headers
    assert _payload(request) == {"url": URL, "parse_effort": "lite"}


@pytest.mark.asyncio
async def test_native_options_are_sent_as_they_are_and_max_chars_is_not_sent() -> None:
    _, request = await _fetch(
        max_chars=600,
        full_page={"extraction_formats": ["markdown", "html"]},
        extraction_source="fetch",
        timeout=60,
        parse_effort="standard",
    )
    assert _payload(request) == {
        "url": URL,
        "parse_effort": "standard",
        "extraction_source": "fetch",
        "timeout": 60,
        "full_page": {"extraction_formats": ["markdown", "html"]},
    }


@pytest.mark.asyncio
async def test_an_option_passed_as_none_counts_as_not_passed() -> None:
    _, request = await _fetch(full_page=None, extraction_source=None, timeout=None, parse_effort=None)
    assert _payload(request) == {"url": URL, "parse_effort": "lite"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("full_page", "sent"),
    [
        ({"extraction_formats": ["html"]}, {"extraction_formats": ["markdown", "html"]}),
        ({"extraction_formats": ["markdown"]}, {"extraction_formats": ["markdown"]}),
        # An empty list and no list at all are You.com's ["markdown"].
        ({"extraction_formats": []}, {"extraction_formats": []}),
        ({}, {}),
    ],
)
async def test_full_page_always_asks_for_markdown(full_page: dict[str, Any], sent: dict[str, Any]) -> None:
    given = json.loads(json.dumps(full_page))
    _, request = await _fetch(full_page=full_page)
    assert _payload(request)["full_page"] == sent
    assert full_page == given


@pytest.mark.asyncio
async def test_a_full_page_that_is_not_an_object_is_refused_before_the_request() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(full_page=["html"])
    assert (raised.value.status, raised.value.tag) == (None, "invalid_option")


@pytest.mark.asyncio
async def test_an_unknown_option_is_refused() -> None:
    with pytest.raises(UnsupportedParameterError, match="crawl_timeout"):
        await _fetch(crawl_timeout=5)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _fetch(api_base="https://proxy.example/ydc/")
    assert str(request.url) == "https://proxy.example/ydc/v1/extract"


@pytest.mark.asyncio
async def test_reads_the_recorded_answer_whole_when_no_max_chars_is_given() -> None:
    page, _ = await _fetch()
    recorded = _recorded("normal").json()
    assert (page.url, page.final_url, page.title) == (URL, recorded["url"], recorded["title"])
    assert (page.text, page.text_truncated) == (recorded["contents"]["full_page"]["markdown"], False)
    assert (page.content_type, page.published, page.error) == ("", None, None)
    assert (page.cost, page.cost_source) == (None, "none")
    assert page.raw == recorded


@pytest.mark.asyncio
async def test_max_chars_cuts_the_text_and_says_so() -> None:
    text = _recorded("normal").json()["contents"]["full_page"]["markdown"]
    page, _ = await _fetch(max_chars=10)
    assert (page.text, page.text_truncated) == (text[:10], True)
    page, _ = await _fetch(max_chars=len(text))
    assert (page.text, page.text_truncated) == (text, False)


@pytest.mark.asyncio
async def test_the_final_url_is_the_one_you_com_answered_for_else_the_url_asked_for() -> None:
    page, _ = await _fetch(httpx.Response(200, json=_answer(url="https://www.example.com/")))
    assert (page.url, page.final_url) == (URL, "https://www.example.com/")
    body = _answer()
    del body["url"]
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert page.final_url == URL
    page, _ = await _fetch(httpx.Response(200, json=_answer(url="")))
    assert page.final_url == URL


@pytest.mark.asyncio
async def test_an_answer_without_a_title_has_none() -> None:
    body = _answer()
    del body["title"]
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.title, page.text) == ("", f"# Page\n\n{SENTINEL} text.")


@pytest.mark.asyncio
async def test_the_recorded_page_with_no_text_is_empty_not_an_error() -> None:
    page, _ = await _fetch(_recorded("empty"))
    assert (page.text, page.error, page.text_truncated) == ("", None, False)
    assert page.title == _recorded("empty").json()["title"]


@pytest.mark.asyncio
async def test_the_recorded_answer_without_contents_under_cache_is_the_error_not_cached() -> None:
    page, _ = await _fetch(_recorded("in_body_error"), extraction_source="cache")
    assert page.error is not None
    assert (page.error.tag, page.error.status) == ("not_cached", None)
    assert (page.text, page.title, page.final_url, page.raw) == ("", "", URL, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("extraction_source", [None, "blend", "fetch"])
async def test_an_answer_without_contents_otherwise_is_the_error_nothing_extracted(
    extraction_source: str | None,
) -> None:
    body = _answer()
    del body["contents"]
    page, _ = await _fetch(httpx.Response(200, json=body), extraction_source=extraction_source)
    assert page.error is not None
    assert page.error.tag == "nothing_extracted"
    assert (page.text, page.title) == ("", f"Title {SENTINEL}")


@pytest.mark.asyncio
async def test_an_html_copy_stays_in_raw() -> None:
    contents = {"full_page": {"markdown": "text", "html": "<p>text</p>"}}
    page, _ = await _fetch(httpx.Response(200, json=_answer(contents=contents)))
    assert page.text == "text"
    assert page.raw["contents"]["full_page"]["html"] == "<p>text</p>"


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    metadata: dict[str, Any] = {"site_name": None, "favicon_url": 7, "new": []}
    page, _ = await _fetch(httpx.Response(200, json=_answer(metadata=metadata, extraction_source=3, newField={})))
    assert (page.title, page.error) == (f"Title {SENTINEL}", None)


def _without(*path: str) -> Any:
    """A change to an answer that deletes the field at ``path``."""

    def change(body: dict[str, Any]) -> None:
        parent: Any = body
        for step in path[:-1]:
            parent = parent[step]
        del parent[path[-1]]

    return change


def _setting(value: Any, *path: str) -> Any:
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
        pytest.param(_setting(None, "contents"), id="contents-null"),
        pytest.param(_setting([SENTINEL], "contents"), id="contents-not-an-object"),
        pytest.param(_without("contents", "full_page"), id="full-page-missing"),
        pytest.param(_setting(None, "contents", "full_page"), id="full-page-null"),
        pytest.param(_setting(SENTINEL, "contents", "full_page"), id="full-page-not-an-object"),
        pytest.param(_without("contents", "full_page", "markdown"), id="markdown-missing"),
        pytest.param(_setting(None, "contents", "full_page", "markdown"), id="markdown-null"),
        pytest.param(_setting([SENTINEL], "contents", "full_page", "markdown"), id="markdown-not-a-string"),
        pytest.param(_setting(None, "url"), id="url-null"),
        pytest.param(_setting(7, "url"), id="url-not-a-string"),
        pytest.param(_setting(None, "title"), id="title-null"),
        pytest.param(_setting([SENTINEL], "title"), id="title-not-a-string"),
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
async def test_the_wrong_type_is_refused_on_an_answer_without_contents_too() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(200, json={"title": 3}))
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
async def test_the_recorded_refused_key_is_tagged_forbidden() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(_recorded("error"))
    assert (raised.value.provider, raised.value.status, raised.value.tag) == ("youdotcom", 403, "forbidden")
    assert "Forbidden" not in str(raised.value)


def _error(code: Any, **details: Any) -> dict[str, Any]:
    """You.com's error body, with a message and details that quote the request."""
    return {
        "error": {
            "code": code,
            "message": f"could not extract {URL} {SENTINEL}",
            "retryable": False,
            "request_id": "767e53fd-36fa-46bd-aee0-27d908cbb726",
            "details": details,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "tag"),
    [
        (401, {"detail": f"bad key {SENTINEL}"}, "unauthorized"),
        (403, {"detail": f"no scope {SENTINEL}"}, "forbidden"),
        (422, _error("invalid_url", reason="unresolvable"), "invalid_url"),
        (422, _error("invalid_request", fields=[{"field": "timeout", "message": SENTINEL}]), "invalid_request"),
        (422, _error("pdf_too_large", timeout_s=30, page_count=900), "pdf_too_large"),
        (500, _error("internal_error"), "internal_error"),
        (502, _error("site_unreachable", reason="load_failed"), "site_unreachable"),
        (502, _error("site_blocked", reason="anti_bot"), "site_blocked"),
        (503, _error("service_unavailable"), "service_unavailable"),
        (504, _error("timeout", timeout_s=30), "timeout"),
        # A 402 from the API, and the gateway's own, which a request without a key gets.
        (402, _error("payment_required", upgrade_url="https://you.com/platform"), "payment_required"),
        (
            402,
            {"error": "payment_required", "message": SENTINEL, "upgrade_url": "https://you.com/platform"},
            "payment_required",
        ),
        (402, {"message": SENTINEL}, "payment_required"),
        (429, _error("rate_limited"), "rate_limited"),
        (429, {"message": SENTINEL}, "http_error"),
        (502, _error(f"not a code {SENTINEL}"), "http_error"),
        (502, _error(None), "http_error"),
    ],
)
async def test_a_failure_is_tagged_with_its_code_or_status_and_carries_no_body_text(
    status: int, body: dict[str, Any], tag: str
) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(status, json=body))
    assert (raised.value.status, raised.value.tag) == (status, tag)
    assert SENTINEL not in str(raised.value) and "example.com" not in str(raised.value)


@pytest.mark.asyncio
async def test_a_failure_with_a_plain_body_is_tagged_http_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(502, text=f"Bad Gateway for {URL}"))
    assert (raised.value.status, raised.value.tag) == (502, "http_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [httpx.Response(200, text="not json"), httpx.Response(200, json=[])])
async def test_a_body_that_is_not_an_object_is_an_invalid_response(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="YDC_API_KEY"):
        AnyFetch.create("youdotcom")
