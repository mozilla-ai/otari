"""The Exa adapter: the request it sends, and how it reads Exa's recorded answers."""

import json
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, MissingCredentialError, ProviderError, SearchResult
from any_search.providers.exa import DEFAULT_CONTENTS, MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "exa"
KEY = "test-key"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    return httpx.Response(recorded["status"], json=recorded["body"])


async def _search(
    answer: httpx.Response | None = None, *, api_base: str | None = None, **kwargs: Any
) -> tuple[SearchResult, httpx.Request]:
    """Search through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnySearch.create("exa", api_key=KEY, api_base=api_base, client=client) as exa:
            result = await exa.search("latest stable python release", **kwargs)
    return result, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("exa")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "EXA_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url) == (False, False)
    assert metadata.max_results == MAX_RESULTS
    assert not [option for option in metadata.options if option.operator_only]
    types = next(option for option in metadata.options if option.name == "type")
    assert types.default == "auto"
    assert "neural" not in (types.enum or [])


@pytest.mark.asyncio
async def test_sends_the_query_with_the_sdk_defaults() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://api.exa.ai/search")
    assert request.headers["x-api-key"] == KEY
    assert _payload(request) == {"query": "latest stable python release", "type": "auto", "contents": DEFAULT_CONTENTS}


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(api_base="https://proxy.example/exa/")
    assert str(request.url) == "https://proxy.example/exa/search"


@pytest.mark.asyncio
async def test_max_results_sets_num_results_and_wins_over_the_option() -> None:
    _, request = await _search(max_results=3, numResults=7)
    assert _payload(request)["numResults"] == 3


@pytest.mark.asyncio
async def test_max_results_is_capped_at_exas_limit() -> None:
    _, request = await _search(max_results=500)
    assert _payload(request)["numResults"] == MAX_RESULTS


@pytest.mark.asyncio
async def test_the_num_results_option_applies_without_max_results() -> None:
    result, request = await _search(numResults=2)
    assert _payload(request)["numResults"] == 2
    assert len(result.hits) == 2


@pytest.mark.asyncio
async def test_the_num_results_option_is_capped_too() -> None:
    _, request = await _search(numResults=500)
    assert _payload(request)["numResults"] == MAX_RESULTS


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, -1, "3"])
async def test_a_num_results_exa_cannot_take_is_left_for_exa_to_refuse(count: Any) -> None:
    result, request = await _search(numResults=count)
    assert _payload(request)["numResults"] == count
    assert len(result.hits) == len(_recorded("normal").json()["results"])


def test_the_default_contents_are_never_shared() -> None:
    option = next(option for option in AnySearch.get_provider_metadata("exa").options if option.name == "contents")
    assert option.default == DEFAULT_CONTENTS
    assert option.default is not DEFAULT_CONTENTS


@pytest.mark.asyncio
async def test_contents_are_sent_as_given_and_false_sends_none() -> None:
    contents = {"highlights": {"maxCharacters": 300}}
    _, request = await _search(contents=contents)
    assert _payload(request)["contents"] == contents
    _, request = await _search(contents=False)
    assert "contents" not in _payload(request)


@pytest.mark.asyncio
async def test_native_options_pass_through_under_their_own_names() -> None:
    options: dict[str, Any] = {
        "type": "fast",
        "category": "news",
        "includeDomains": ["python.org"],
        "excludeDomains": ["example.com"],
        "endPublishedDate": "2026-01-01T00:00:00Z",
        "userLocation": "US",
        "moderation": True,
        "additionalQueries": ["python release"],
    }
    _, request = await _search(**options)
    assert _payload(request).items() >= options.items()


@pytest.mark.asyncio
async def test_time_range_sets_the_start_date_and_wins_over_the_option() -> None:
    _, request = await _search(time_range="week", startPublishedDate="2000-01-01T00:00:00Z")
    start = datetime.fromisoformat(_payload(request)["startPublishedDate"])
    # The start of that day, so pages Exa dates by their day alone, as midnight, stay in.
    assert start == datetime.combine((datetime.now(UTC) - timedelta(weeks=1)).date(), time(), UTC)


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    first, second, _ = result.hits
    assert first.title == recorded[0]["title"]
    assert first.snippet == " ".join(part.strip() for part in recorded[0]["highlights"])
    assert first.text == recorded[0]["text"]
    assert first.published == datetime(2025, 10, 1, tzinfo=UTC)
    assert first.raw == recorded[0]
    # Exa answered this one with an empty title and no date.
    assert (second.title, second.published) == ("", None)
    assert (result.cost, result.cost_source, result.error) == (Decimal("0.007"), "reported", None)
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_an_empty_answer_reports_its_cost() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.cost, result.cost_source) == ([], Decimal(0), "reported")


@pytest.mark.asyncio
async def test_an_item_without_a_url_and_a_missing_field_are_tolerated() -> None:
    body = {
        "results": [
            {"title": "no url"},
            "not an item",
            {"url": "https://example.com/a", "highlights": [" one ", "", 3, "two"], "publishedDate": "2026-02-03"},
            {"url": "https://example.com/b", "text": "", "publishedDate": "not a date"},
        ]
    }
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a", "https://example.com/b"]
    first, second = result.hits
    assert (first.snippet, first.published) == ("one two", datetime(2026, 2, 3, tzinfo=UTC))
    assert (second.snippet, second.text, second.published) == ("", None, None)
    assert (result.cost, result.cost_source) == (None, "none")


@pytest.mark.asyncio
async def test_a_failure_raises_with_exas_status_and_tag() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
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
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, "http_error")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="not json"), httpx.Response(200, json=[]), httpx.Response(200, json={"results": {}})],
)
async def test_a_body_that_is_not_a_search_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_a_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="EXA_API_KEY"):
        AnySearch.create("exa")
