"""The fake provider: canned hits, and each option's behavior."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from any_search import AnySearch, ProviderError, SearchResult
from any_search.providers.fake import CANNED_HITS, FakeOptions


async def _search(query: str = "q", **kwargs: Any) -> SearchResult:
    async with AnySearch.create("fake") as fake:
        return await fake.search(query, **kwargs)


def test_metadata_lists_every_option_the_provider_reads() -> None:
    metadata = AnySearch.get_provider_metadata("fake")
    assert metadata.tier == "test"
    assert {option.name for option in metadata.options} == set(FakeOptions.model_fields)


def test_account_is_the_one_operator_only_option() -> None:
    metadata = AnySearch.get_provider_metadata("fake")
    assert [option.name for option in metadata.options if option.operator_only] == ["account"]


@pytest.mark.asyncio
async def test_answers_with_the_canned_hits() -> None:
    result = await _search()
    assert [hit.url for hit in result.hits] == [item["url"] for item in CANNED_HITS]
    assert result.hits[0].published == datetime(2026, 1, 15, 9, 30, tzinfo=UTC)
    assert result.hits[1].text is not None
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)


@pytest.mark.asyncio
async def test_hits_replace_the_canned_ones_and_an_item_without_a_url_is_dropped() -> None:
    result = await _search(hits=[{"url": "https://a.example/", "title": None}, {"title": "no url"}])
    assert [(hit.url, hit.title, hit.snippet) for hit in result.hits] == [("https://a.example/", "", "")]


@pytest.mark.asyncio
async def test_max_results_cuts_the_hits() -> None:
    assert len((await _search(max_results=2)).hits) == 2


@pytest.mark.asyncio
async def test_cost_is_reported() -> None:
    result = await _search(cost=0.007)
    assert (result.cost, result.cost_source) == (Decimal("0.007"), "reported")


@pytest.mark.asyncio
async def test_error_raises_with_its_tag_and_status() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(error="rate_limit", error_status=429)
    assert (raised.value.provider, raised.value.tag, raised.value.status) == ("fake", "rate_limit", 429)


@pytest.mark.asyncio
async def test_an_option_of_the_wrong_type_raises_a_library_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(delay="abc")
    assert (raised.value.provider, raised.value.tag, raised.value.status) == ("fake", "invalid_option", None)
    assert "abc" not in str(raised.value)


@pytest.mark.asyncio
async def test_in_body_error_is_returned_not_raised() -> None:
    result = await _search(in_body_error="engine_unavailable")
    assert result.hits == []
    assert result.error is not None and result.error.tag == "engine_unavailable"


@pytest.mark.asyncio
async def test_delay_holds_the_call_open() -> None:
    call = asyncio.create_task(_search(delay=30))
    await asyncio.sleep(0.05)
    assert not call.done()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_leak_query_raises_with_the_query_in_the_message() -> None:
    with pytest.raises(RuntimeError, match="sentinel-query"):
        await _search("sentinel-query", leak_query=True)


@pytest.mark.asyncio
async def test_raw_echoes_the_request_but_not_the_query() -> None:
    result = await _search("sentinel-query", max_results=1, time_range="week", account="acme")
    assert result.raw["request"] == {"max_results": 1, "time_range": "week", "options": {"account": "acme"}}
    assert "sentinel-query" not in repr(result.raw)
