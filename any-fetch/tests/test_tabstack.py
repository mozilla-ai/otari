"""The Tabstack extract adapter: the request it sends, and how it reads Tabstack's recorded answers."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_fetch import AnyFetch, FetchedPage, MissingCredentialError, ProviderError, UnsupportedParameterError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tabstack"
KEY = "test-key"
URL = "https://example.com/"


@pytest.fixture(autouse=True)
def _default_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a base URL set in the developer's environment out of the requests checked here."""
    monkeypatch.delenv("TABSTACK_BASE_URL", raising=False)


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
        async with AnyFetch.create("tabstack", api_key=KEY, api_base=api_base, client=client) as tabstack:
            page = await tabstack.fetch(URL, **kwargs)
    return page, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


SENTINEL = "sentinel-body-text"


def _answer(**fields: Any) -> dict[str, Any]:
    """A successful answer with every field the adapter reads, and a sentinel no error may carry."""
    return {
        "url": URL,
        "content": f"page {SENTINEL}",
        "metadata": {"title": f"Title {SENTINEL}", "url": URL},
        **fields,
    }


def test_metadata() -> None:
    metadata = AnyFetch.get_provider_metadata("tabstack")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "TABSTACK_API_KEY", True)
    assert (metadata.max_urls_per_call, metadata.formats) == (1, ["markdown"])
    assert not [option for option in metadata.options if option.operator_only]


@pytest.mark.asyncio
async def test_sends_the_url_with_the_key_in_a_bearer_header_and_asks_for_the_metadata() -> None:
    _, request = await _fetch()
    assert (request.method, str(request.url)) == ("POST", "https://api.tabstack.ai/v1/extract/markdown")
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert _payload(request) == {"url": URL, "metadata": True}


@pytest.mark.asyncio
async def test_native_options_are_sent_as_they_are_and_max_chars_is_not_sent() -> None:
    _, request = await _fetch(max_chars=600, content="full", effort="max", geo_target={"country": "GB"}, nocache=True)
    assert _payload(request) == {
        "url": URL,
        "metadata": True,
        "content": "full",
        "effort": "max",
        "geo_target": {"country": "GB"},
        "nocache": True,
    }


@pytest.mark.asyncio
async def test_metadata_is_not_an_option() -> None:
    with pytest.raises(UnsupportedParameterError, match="metadata"):
        await _fetch(metadata=False)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _fetch(api_base="https://proxy.example/tabstack/")
    assert str(request.url) == "https://proxy.example/tabstack/extract/markdown"


@pytest.mark.asyncio
async def test_the_base_comes_from_the_sdks_variable_unless_one_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABSTACK_BASE_URL", "https://env.example/v1")
    _, request = await _fetch()
    assert str(request.url) == "https://env.example/v1/extract/markdown"
    # An explicit empty base is the provider's default, never the environment's.
    _, request = await _fetch(api_base="")
    assert str(request.url) == "https://api.tabstack.ai/v1/extract/markdown"


@pytest.mark.asyncio
async def test_reads_the_recorded_answer_whole_when_no_max_chars_is_given() -> None:
    page, _ = await _fetch()
    recorded = _recorded("normal").json()
    assert (page.url, page.final_url, page.title) == (URL, recorded["url"], recorded["metadata"]["title"])
    assert (page.text, page.text_truncated) == (recorded["content"], False)
    assert (page.content_type, page.published, page.error) == ("", None, None)
    assert (page.cost, page.cost_source) == (None, "none")
    assert page.raw == recorded


@pytest.mark.asyncio
async def test_max_chars_cuts_the_text() -> None:
    text = _recorded("normal").json()["content"]
    page, _ = await _fetch(max_chars=10)
    assert (page.text, page.text_truncated) == (text[:10], True)
    page, _ = await _fetch(max_chars=len(text))
    assert (page.text, page.text_truncated) == (text, False)


@pytest.mark.asyncio
async def test_the_final_url_is_the_one_tabstack_answered_for() -> None:
    page, _ = await _fetch(httpx.Response(200, json=_answer(url="https://www.example.com/")))
    assert (page.url, page.final_url) == (URL, "https://www.example.com/")


@pytest.mark.asyncio
async def test_a_page_with_no_text_is_empty_not_an_error() -> None:
    page, _ = await _fetch(_recorded("empty"))
    assert (page.text, page.title, page.error, page.text_truncated) == ("", "", None, False)


@pytest.mark.asyncio
async def test_an_answer_without_metadata_or_a_title_has_no_title() -> None:
    body = _answer()
    del body["metadata"]
    page, _ = await _fetch(httpx.Response(200, json=body))
    assert (page.title, page.text) == ("", f"page {SENTINEL}")
    page, _ = await _fetch(httpx.Response(200, json=_answer(metadata={"url": URL})))
    assert page.title == ""


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    # created_at is not read: Tabstack sends a PDF's own date format there, not the ISO 8601 it documents.
    metadata: dict[str, Any] = {
        "title": "Title",
        "author": None,
        "favicon": 7,
        "created_at": "D:20240410211143Z",
        "new": [],
    }
    page, _ = await _fetch(httpx.Response(200, json=_answer(metadata=metadata, newTopLevel={"any": "thing"})))
    assert (page.title, page.published, page.error) == ("Title", None, None)


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
        pytest.param(_without("content"), id="content-missing"),
        pytest.param(_setting(None, "content"), id="content-null"),
        pytest.param(_setting([f"page {SENTINEL}"], "content"), id="content-not-a-string"),
        pytest.param(_without("url"), id="url-missing"),
        pytest.param(_setting("", "url"), id="url-empty"),
        pytest.param(_setting(7, "url"), id="url-not-a-string"),
        pytest.param(_setting(None, "metadata"), id="metadata-null"),
        pytest.param(_setting([f"Title {SENTINEL}"], "metadata"), id="metadata-not-an-object"),
        pytest.param(_setting(None, "metadata", "title"), id="title-null"),
        pytest.param(_setting([f"Title {SENTINEL}"], "metadata", "title"), id="title-not-a-string"),
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
async def test_a_refused_key_raises_with_the_status_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(_recorded("error"))
    assert (raised.value.provider, raised.value.status, raised.value.tag) == ("tabstack", 401, "http_error")
    assert "Unauthorized" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        # Tabstack's error bodies are sentences, and some quote the URL.
        httpx.Response(422, json={"error": f"domain resolution failed for {URL}"}),
        httpx.Response(500, json={"error": "failed to fetch URL"}),
        httpx.Response(503, text="<html>overloaded</html>"),
    ],
)
async def test_a_failure_is_an_http_error_that_carries_no_body_text(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, "http_error")
    assert "example.com" not in str(raised.value) and "fetch URL" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="not json"), httpx.Response(200, json=[])],
)
async def test_a_body_that_is_not_an_extract_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABSTACK_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="TABSTACK_API_KEY"):
        AnyFetch.create("tabstack")
