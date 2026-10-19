"""The DeepSeek fetch adapter: the request it sends, how it reads recorded answers, and the fields it checks."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_fetch import AnyFetch, FetchedPage, ProviderError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "deepseek"
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
        async with AnyFetch.create("deepseek", api_key=KEY, api_base=api_base, client=client) as deepseek:
            page = await deepseek.fetch(URL, **kwargs)
    return page, sent[0]


def _answer(**fields: Any) -> httpx.Response:
    """A successful answer with both fields DeepSeek documents, and a sentinel no error may carry."""
    body: dict[str, Any] = {"title": f"Title {SENTINEL}", "content": f"# Page\n\n{SENTINEL} text."}
    body.update(fields)
    return httpx.Response(200, json={name: value for name, value in body.items() if value is not ...})


def test_metadata() -> None:
    metadata = AnyFetch.get_provider_metadata("deepseek")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "DEEPSEEK_API_KEY", True)
    assert (metadata.max_urls_per_call, metadata.renders_javascript, metadata.formats) == (1, False, ["markdown"])
    assert metadata.reports_final_url is False
    assert metadata.options == []


@pytest.mark.asyncio
async def test_sends_the_url_in_a_post_with_the_key_in_a_bearer_header() -> None:
    _, request = await _fetch(_answer())
    assert (request.method, str(request.url)) == ("POST", "https://api.deepseek.com/v1/tools/fetch")
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {"url": URL}


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _fetch(_answer(), api_base="https://proxy.example/deepseek/")
    assert str(request.url) == "https://proxy.example/deepseek/v1/tools/fetch"


@pytest.mark.asyncio
async def test_reads_the_title_and_the_text() -> None:
    page, _ = await _fetch(_answer(title="Example", content="# Example\n\nText."))
    assert (page.url, page.final_url, page.title, page.text) == (URL, URL, "Example", "# Example\n\nText.")
    assert (page.content_type, page.cost, page.cost_source, page.error) == ("", None, "none", None)
    assert (page.text_truncated, page.source_truncated) == (False, False)
    assert page.raw == {"title": "Example", "content": "# Example\n\nText."}


@pytest.mark.asyncio
async def test_max_chars_cuts_the_text_and_says_so() -> None:
    page, _ = await _fetch(_answer(content="0123456789"), max_chars=4)
    assert (page.text, page.text_truncated) == ("0123", True)


@pytest.mark.asyncio
async def test_text_within_max_chars_is_not_truncated() -> None:
    page, _ = await _fetch(_answer(content="0123"), max_chars=4)
    assert (page.text, page.text_truncated) == ("0123", False)


@pytest.mark.asyncio
async def test_an_empty_title_is_no_title() -> None:
    page, _ = await _fetch(_answer(title=""))
    assert page.title == ""


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    page, _ = await _fetch()
    recorded = _recorded("normal").json()
    assert (page.title, page.text) == (recorded["title"], recorded["content"])
    assert page.raw == recorded


@pytest.mark.asyncio
async def test_the_recorded_missing_page_fails_the_call_tagged_not_found_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (404, "not_found_error")


@pytest.mark.asyncio
async def test_an_error_without_a_type_is_tagged_with_its_code() -> None:
    answer = httpx.Response(400, json={"error": {"message": "bad", "code": "invalid_request_error"}})
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (400, "invalid_request_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [None, 42, "a sentence, not a code"])
async def test_an_error_without_a_usable_type_or_code_is_tagged_http_error(code: Any) -> None:
    answer = httpx.Response(404, json={"error": {"message": f"could not fetch {URL}", "type": code, "code": code}})
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (404, "http_error")
    assert URL not in str(raised.value)


@pytest.mark.asyncio
async def test_an_error_with_a_plain_body_is_tagged_http_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(httpx.Response(502, text=f"Bad Gateway for {URL}"))
    assert (raised.value.status, raised.value.tag) == (502, "http_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [httpx.Response(200, text="not json"), httpx.Response(200, json=["content"])])
async def test_a_body_that_is_not_an_object_is_an_invalid_response(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert raised.value.tag == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [{"title": ...}, {"content": ...}, {"title": None}, {"content": None}, {"content": ["text"]}, {"title": 3}],
)
async def test_a_documented_field_missing_or_of_the_wrong_type_is_an_unexpected_response(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(_answer(**fields))
    assert raised.value.tag == "unexpected_response"
    assert SENTINEL not in str(raised.value)


@pytest.mark.asyncio
async def test_fields_the_adapter_does_not_read_are_ignored() -> None:
    page, _ = await _fetch(_answer(brand_new_field={"a": 1}))
    assert page.error is None
