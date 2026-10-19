"""The DeepSeek adapter: the request it sends, how it reads DeepSeek's recorded answers, and the fields it checks."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, ProviderError, SearchResult, UnsupportedParameterError
from any_search.providers.deepseek import MAX_RESULTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "deepseek"
KEY = "test-key"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    body = recorded["body"]
    if isinstance(body, str):
        return httpx.Response(recorded["status"], text=body)
    return httpx.Response(recorded["status"], json=body)


def _item(**overrides: Any) -> dict[str, Any]:
    """One result with every field DeepSeek documents, as its reference shows it."""
    item: dict[str, Any] = {
        "rerank_score": 2.4,
        "title": "DeepSeek V4 model released",
        "url": "https://example.com/deepseek-v4",
        "snippet": "DeepSeek V4 is now available.",
        "summary": "DeepSeek has released the V4 model.",
        "content": "",
        "site_name": "Example News",
        "site_icon": None,
        "date": "2026-06-01T00:00:00+00:00",
        "authority": 3,
    }
    item.update(overrides)
    return item


async def _search(
    answer: httpx.Response | None = None, *, api_base: str | None = None, **kwargs: Any
) -> tuple[SearchResult, httpx.Request]:
    """Search through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnySearch.create("deepseek", api_key=KEY, api_base=api_base, client=client) as deepseek:
            result = await deepseek.search("latest stable python release", **kwargs)
    return result, sent[0]


def _payload(request: httpx.Request) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.content)
    return payload


def _answer(*items: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"results": list(items)})


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("deepseek")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "DEEPSEEK_API_KEY", True)
    assert (metadata.query_in_url, metadata.key_in_url) == (False, False)
    assert metadata.max_results == MAX_RESULTS == 50
    assert [option.name for option in metadata.options] == ["ori_question", "localization", "need_content"]


@pytest.mark.asyncio
async def test_sends_the_query_in_a_post_with_the_key_in_a_bearer_header() -> None:
    _, request = await _search(_answer())
    assert (request.method, str(request.url)) == ("POST", "https://api.deepseek.com/v1/tools/search")
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert _payload(request) == {"query": "latest stable python release"}
    assert KEY not in str(request.url)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint() -> None:
    _, request = await _search(_answer(), api_base="https://proxy.example/deepseek/")
    assert str(request.url) == "https://proxy.example/deepseek/v1/tools/search"


@pytest.mark.asyncio
async def test_max_results_sets_count_and_cuts_the_hits() -> None:
    result, request = await _search(_answer(_item(), _item(), _item()), max_results=2)
    assert _payload(request)["count"] == 2
    assert len(result.hits) == 2


@pytest.mark.asyncio
async def test_max_results_is_capped_at_deepseeks_limit() -> None:
    _, request = await _search(_answer(), max_results=80)
    assert _payload(request)["count"] == MAX_RESULTS


@pytest.mark.asyncio
async def test_deepseeks_own_count_is_not_an_option() -> None:
    """``max_results`` is the one name for the count."""
    with pytest.raises(UnsupportedParameterError):
        await _search(_answer(), count=3)


@pytest.mark.asyncio
async def test_native_options_pass_through_under_their_own_names() -> None:
    options: dict[str, Any] = {
        "ori_question": "Which Python is current?",
        "localization": "8.8.8.8",
        "need_content": True,
    }
    _, request = await _search(_answer(), **options)
    assert _payload(request).items() >= options.items()


@pytest.mark.asyncio
async def test_an_option_set_to_null_is_not_sent() -> None:
    _, request = await _search(_answer(), ori_question=None, need_content=None)
    assert _payload(request) == {"query": "latest stable python release"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_range", "span"),
    [
        ("day", timedelta(days=1)),
        ("week", timedelta(weeks=1)),
        ("month", timedelta(days=30)),
        ("year", timedelta(days=365)),
    ],
)
async def test_time_range_is_the_after_operator_on_the_day_the_range_starts(time_range: str, span: timedelta) -> None:
    before = (datetime.now(UTC) - span).date().isoformat()
    _, request = await _search(_answer(), time_range=time_range)
    after = (datetime.now(UTC) - span).date().isoformat()
    # Either day, should the search run across midnight.
    assert _payload(request)["query"] in {
        f"latest stable python release after:{before}",
        f"latest stable python release after:{after}",
    }


@pytest.mark.asyncio
async def test_reads_a_hit_from_the_fields_deepseek_documents() -> None:
    item = _item(content="Page text.")
    result, _ = await _search(_answer(item))
    (hit,) = result.hits
    assert (hit.url, hit.title, hit.snippet, hit.text) == (item["url"], item["title"], item["snippet"], "Page text.")
    assert hit.published == datetime(2026, 6, 1, tzinfo=UTC)
    assert hit.raw == item
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)


@pytest.mark.asyncio
async def test_empty_content_is_no_text_and_a_null_date_is_no_date() -> None:
    result, _ = await _search(_answer(_item(content="", date=None)))
    (hit,) = result.hits
    assert (hit.text, hit.published) == (None, None)


@pytest.mark.asyncio
async def test_a_date_without_a_zone_is_utc() -> None:
    result, _ = await _search(_answer(_item(date="2026-06-01T08:30:00")))
    assert result.hits[0].published == datetime(2026, 6, 1, 8, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    recorded = _recorded("normal").json()["results"]
    assert [hit.url for hit in result.hits] == [item["url"] for item in recorded]
    assert [hit.snippet for hit in result.hits] == [item["snippet"] for item in recorded]
    assert result.raw == _recorded("normal").json()


@pytest.mark.asyncio
async def test_reads_the_recorded_empty_answer() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
async def test_the_recorded_refused_key_is_a_provider_error_tagged_with_deepseeks_type() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (401, "authentication_error")
    # The message quotes the key's last four characters, so none of it reaches the error.
    assert "0000" not in str(raised.value)


@pytest.mark.asyncio
async def test_an_error_without_a_type_is_tagged_with_its_code() -> None:
    answer = httpx.Response(400, json={"error": {"message": "bad", "code": "invalid_request_error"}})
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (400, "invalid_request_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [None, 42, "a sentence, not a code"])
async def test_an_error_without_a_usable_type_or_code_is_tagged_http_error(code: Any) -> None:
    answer = httpx.Response(429, json={"error": {"message": "slow down", "type": code, "code": code}})
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (429, "http_error")


@pytest.mark.asyncio
async def test_an_error_with_a_plain_body_is_tagged_http_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(502, text="Bad Gateway"))
    assert (raised.value.status, raised.value.tag) == (502, "http_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [httpx.Response(200, text="not json"), httpx.Response(200, json=["results"])])
async def test_a_body_that_is_not_an_object_is_an_invalid_response(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert raised.value.tag == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [None, "results", {"url": "https://example.com"}])
async def test_results_must_be_a_list(results: Any) -> None:
    body = {} if results is None else {"results": results}
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["url", "title", "snippet", "content", "date"])
async def test_a_documented_field_missing_is_an_unexpected_response(field: str) -> None:
    item = _item()
    del item[field]
    with pytest.raises(ProviderError) as raised:
        await _search(_answer(item))
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"url": ""},
        {"url": None},
        {"title": None},
        {"snippet": 7},
        {"content": None},
        {"date": 1717200000},
        {"date": "last tuesday"},
    ],
)
async def test_a_field_of_the_wrong_type_is_an_unexpected_response(overrides: dict[str, Any]) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_answer(_item(**overrides)))
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
async def test_an_item_that_is_not_an_object_is_an_unexpected_response() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_answer("https://example.com"))  # type: ignore[arg-type]
    assert raised.value.tag == "unexpected_response"


@pytest.mark.asyncio
async def test_only_the_hits_the_caller_gets_are_checked() -> None:
    result, _ = await _search(_answer(_item(), {"broken": True}), max_results=1)
    assert len(result.hits) == 1


@pytest.mark.asyncio
async def test_fields_the_adapter_does_not_read_are_ignored() -> None:
    result, _ = await _search(_answer(_item(rerank_score="high", brand_new_field=[1, 2])))
    assert len(result.hits) == 1
