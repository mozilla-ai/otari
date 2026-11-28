"""The Firecrawl scrape adapter: the request it sends, how it reads recorded answers, and the fields it checks."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_fetch import AnyFetch, FetchedPage, MissingCredentialError, ProviderError, UnsupportedParameterError
from any_fetch.providers.firecrawl import DEFAULT_PARSERS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "firecrawl"
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
        async with AnyFetch.create("firecrawl", api_key=KEY, api_base=api_base, client=client) as firecrawl:
            page = await firecrawl.fetch(URL, **kwargs)
    return page, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


def _answer(**metadata: Any) -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    fields: dict[str, Any] = {
        "title": f"Title {SENTINEL}",
        "sourceURL": URL,
        "url": "https://www.example.com/",
        "statusCode": 200,
        "contentType": "text/html; charset=utf-8",
        "creditsUsed": 1,
    }
    fields.update(metadata)
    return {"success": True, "data": {"markdown": f"# Page\n\n{SENTINEL} text.", "metadata": fields}}


def test_metadata() -> None:
    metadata = AnyFetch.get_provider_metadata("firecrawl")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "FIRECRAWL_API_KEY", True)
    assert (metadata.default_api_base, metadata.env_api_base) == ("https://api.firecrawl.dev", None)
    assert (metadata.max_urls_per_call, metadata.renders_javascript, metadata.formats) == (1, True, ["markdown"])
    assert (metadata.reports_final_url, metadata.reports_cost) == (True, False)
    assert not any(option.cost_tier for option in metadata.options)
    assert not any(option.operator_only for option in metadata.options)
    assert next(option.default for option in metadata.options if option.name == "parsers") == DEFAULT_PARSERS
    names = {option.name for option in metadata.options}
    # The format is the envelope's, and the deprecated proxy and the options that add a charge per page are left out.
    assert not names & {"url", "formats", "origin", "integration", "proxy", "lockdown", "redactPII"}


@pytest.mark.asyncio
async def test_posts_the_url_markdown_and_capped_pdf_parsers_with_a_bearer_key() -> None:
    _, request = await _fetch()
    assert (request.method, str(request.url)) == ("POST", "https://api.firecrawl.dev/v2/scrape")
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert _payload(request) == {"url": URL, "formats": ["markdown"], "parsers": DEFAULT_PARSERS}
    assert DEFAULT_PARSERS == [{"type": "pdf", "maxPages": 100}, "image"]


@pytest.mark.asyncio
async def test_native_options_are_sent_as_they_are_and_max_chars_is_not_sent() -> None:
    options: dict[str, Any] = {
        "onlyMainContent": False,
        "includeTags": ["article"],
        "excludeTags": ["nav"],
        "maxAge": 0,
        "waitFor": 1000,
        "mobile": True,
        "timeout": 30000,
        "location": {"country": "DE", "languages": ["de"]},
        "removeBase64Images": False,
        "blockAds": False,
        "storeInCache": False,
        "parsers": [{"type": "pdf", "maxPages": 20}],
    }
    _, request = await _fetch(max_chars=600, **options)
    assert _payload(request) == {"url": URL, "formats": ["markdown"], **options}


@pytest.mark.asyncio
async def test_an_option_passed_as_none_counts_as_not_passed() -> None:
    _, request = await _fetch(onlyMainContent=None, maxAge=None, parsers=None)
    assert _payload(request) == {"url": URL, "formats": ["markdown"], "parsers": DEFAULT_PARSERS}


@pytest.mark.asyncio
async def test_parsers_passed_replace_the_default_even_when_empty() -> None:
    _, request = await _fetch(parsers=[])
    assert _payload(request)["parsers"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["formats", "origin", "integration", "proxy", "lockdown", "actions"])
async def test_a_reserved_or_unoffered_option_is_refused(name: str) -> None:
    with pytest.raises(UnsupportedParameterError, match=name):
        await _fetch(None, **{name: "x"})


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _fetch(api_base="https://firecrawl.internal.example/")
    assert str(request.url) == "https://firecrawl.internal.example/v2/scrape"


@pytest.mark.asyncio
async def test_reads_the_recorded_answer_whole_when_no_max_chars_is_given() -> None:
    page, _ = await _fetch()
    recorded = _recorded("normal").json()
    data = recorded["data"]
    assert (page.url, page.final_url, page.title) == (URL, data["metadata"]["url"], data["metadata"]["title"])
    assert (page.text, page.text_truncated) == (data["markdown"], False)
    assert (page.content_type, page.published, page.error) == (data["metadata"]["contentType"], None, None)
    assert (page.cost, page.cost_source) == (None, "none")
    assert page.raw == recorded


@pytest.mark.asyncio
async def test_max_chars_cuts_the_text_and_says_so() -> None:
    text = _recorded("normal").json()["data"]["markdown"]
    page, _ = await _fetch(max_chars=10)
    assert (page.text, page.text_truncated) == (text[:10], True)
    page, _ = await _fetch(max_chars=len(text))
    assert (page.text, page.text_truncated) == (text, False)


@pytest.mark.asyncio
async def test_the_final_url_is_where_firecrawl_found_the_page() -> None:
    page, _ = await _fetch(httpx.Response(200, json=_answer(url="https://www.example.com/after")))
    assert (page.url, page.final_url) == (URL, "https://www.example.com/after")


@pytest.mark.asyncio
async def test_the_recorded_error_page_with_no_text_is_an_empty_page() -> None:
    page, _ = await _fetch(_recorded("empty"))
    metadata = _recorded("empty").json()["data"]["metadata"]
    assert (page.text, page.error, page.text_truncated) == ("", None, False)
    assert (page.title, page.final_url, page.content_type) == ("", metadata["url"], metadata["contentType"])


@pytest.mark.asyncio
async def test_the_recorded_unresolvable_host_is_the_pages_error() -> None:
    page, _ = await _fetch(_recorded("in_body_error"))
    assert page.error is not None
    assert (page.error.tag, page.error.status) == ("SCRAPE_DNS_RESOLUTION_ERROR", None)
    assert (page.text, page.title, page.final_url, page.content_type) == ("", "", URL, "")
    assert page.raw == _recorded("in_body_error").json()


@pytest.mark.asyncio
async def test_success_false_without_a_code_is_unsuccessful() -> None:
    page, _ = await _fetch(httpx.Response(200, json={"success": False, "error": f"{URL} {SENTINEL}"}))
    assert page.error is not None
    assert page.error.tag == "unsuccessful"


@pytest.mark.asyncio
@pytest.mark.parametrize(("title", "read"), [(["First", "Second"], "First"), ([], ""), ("Plain", "Plain")])
async def test_a_title_may_be_a_list_and_the_first_is_read(title: Any, read: str) -> None:
    page, _ = await _fetch(httpx.Response(200, json=_answer(title=title)))
    assert page.title == read


@pytest.mark.asyncio
async def test_an_answer_without_a_title_or_content_type_has_empty_ones() -> None:
    body = _answer()
    del body["data"]["metadata"]["title"], body["data"]["metadata"]["contentType"]
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.title, page.content_type, page.text) == ("", "", f"# Page\n\n{SENTINEL} text.")


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer(statusCode="ok", ogTitle=[7], language=None, newField={})
    body["data"]["links"] = 3
    page, _ = await _fetch(httpx.Response(200, json=body))
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
        pytest.param(_setting("yes", "success"), id="success-not-a-boolean"),
        pytest.param(_without("success"), id="success-missing"),
        pytest.param(_without("data"), id="data-missing"),
        pytest.param(_setting([SENTINEL], "data"), id="data-not-an-object"),
        pytest.param(_without("data", "markdown"), id="markdown-missing"),
        pytest.param(_setting(None, "data", "markdown"), id="markdown-null"),
        pytest.param(_setting([SENTINEL], "data", "markdown"), id="markdown-not-a-string"),
        pytest.param(_without("data", "metadata"), id="metadata-missing"),
        pytest.param(_setting(SENTINEL, "data", "metadata"), id="metadata-not-an-object"),
        pytest.param(_without("data", "metadata", "url"), id="url-missing"),
        pytest.param(_setting("", "data", "metadata", "url"), id="url-empty"),
        pytest.param(_setting(7, "data", "metadata", "url"), id="url-not-a-string"),
        pytest.param(_setting(None, "data", "metadata", "title"), id="title-null"),
        pytest.param(_setting([SENTINEL, 3], "data", "metadata", "title"), id="title-list-not-of-strings"),
        pytest.param(_setting({"t": SENTINEL}, "data", "metadata", "title"), id="title-not-a-string"),
        pytest.param(_setting(None, "data", "metadata", "contentType"), id="content-type-null"),
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
async def test_the_recorded_refused_key_is_tagged_unauthorized() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(_recorded("error"))
    assert (raised.value.provider, raised.value.status, raised.value.tag) == ("firecrawl", 401, "unauthorized")
    assert "Invalid token" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "tag"),
    [
        (401, {"success": False, "error": f"Unauthorized: {SENTINEL}"}, "unauthorized"),
        (402, {"success": False, "error": SENTINEL, "code": "insufficient_credits"}, "payment_required"),
        (429, {"error": f"Request rate limit exceeded {SENTINEL}"}, "rate_limited"),
        (400, {"success": False, "code": "BAD_REQUEST", "error": "Bad Request", "details": [SENTINEL]}, "BAD_REQUEST"),
        (
            403,
            {"success": False, "code": "THIRD_PARTY_DATA_TERMS_REQUIRED", "error": SENTINEL},
            "THIRD_PARTY_DATA_TERMS_REQUIRED",
        ),
        (404, {"success": False, "code": "SCRAPE_NO_CACHED_DATA", "error": SENTINEL}, "SCRAPE_NO_CACHED_DATA"),
        (408, {"success": False, "code": "SCRAPE_TIMEOUT", "error": SENTINEL}, "SCRAPE_TIMEOUT"),
        (
            500,
            {"success": False, "code": "SCRAPE_ALL_ENGINES_FAILED", "error": f"{URL} {SENTINEL}", "jobId": "01a2"},
            "SCRAPE_ALL_ENGINES_FAILED",
        ),
        (500, {"success": False, "code": f"not a code {SENTINEL}", "error": SENTINEL}, "http_error"),
        (503, {"success": False, "error": SENTINEL}, "http_error"),
    ],
)
async def test_a_failure_is_tagged_by_status_or_code_and_carries_no_body_text(
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
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="FIRECRAWL_API_KEY"):
        AnyFetch.create("firecrawl")
