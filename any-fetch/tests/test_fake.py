"""The fake provider: a canned page, and each option's behavior."""

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from any_fetch import AnyFetch, FetchedPage, ProviderError
from any_fetch.providers.fake import CANNED_TEXT, CANNED_TITLE, FakeOptions

URL = "https://example.com/page"


async def _fetch(url: str = URL, **kwargs: Any) -> FetchedPage:
    async with AnyFetch.create("fake") as fake:
        return await fake.fetch(url, **kwargs)


def test_metadata_lists_every_option_the_provider_reads() -> None:
    metadata = AnyFetch.get_provider_metadata("fake")
    assert metadata.tier == "test"
    assert {option.name for option in metadata.options} == set(FakeOptions.model_fields)


def test_account_is_the_one_operator_only_option() -> None:
    metadata = AnyFetch.get_provider_metadata("fake")
    assert [option.name for option in metadata.options if option.operator_only] == ["account"]


@pytest.mark.asyncio
async def test_answers_with_the_canned_page() -> None:
    page = await _fetch()
    assert (page.url, page.final_url, page.title, page.text) == (URL, URL, CANNED_TITLE, CANNED_TEXT)
    assert (page.content_type, page.cost, page.cost_source, page.error) == ("text/html", None, "none", None)
    assert not page.source_truncated and not page.text_truncated


@pytest.mark.asyncio
async def test_options_shape_the_page() -> None:
    page = await _fetch(
        text="body", title="", content_type="text/plain", final_url="https://example.com/moved", source_truncated=True
    )
    assert (page.text, page.title, page.content_type) == ("body", "", "text/plain")
    assert (page.final_url, page.source_truncated) == ("https://example.com/moved", True)


@pytest.mark.asyncio
async def test_max_chars_cuts_the_text_and_says_so() -> None:
    page = await _fetch(max_chars=10)
    assert (page.text, page.text_truncated) == (CANNED_TEXT[:10], True)
    assert not (await _fetch(max_chars=len(CANNED_TEXT))).text_truncated


@pytest.mark.asyncio
async def test_cost_is_reported() -> None:
    page = await _fetch(cost=0.001)
    assert (page.cost, page.cost_source) == (Decimal("0.001"), "reported")


@pytest.mark.asyncio
async def test_error_raises_with_its_tag_and_status() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(error="http_status", error_status=404)
    assert (raised.value.provider, raised.value.tag, raised.value.status) == ("fake", "http_status", 404)


@pytest.mark.asyncio
async def test_an_option_of_the_wrong_type_raises_a_library_error() -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(delay="abc")
    assert (raised.value.provider, raised.value.tag, raised.value.status) == ("fake", "invalid_option", None)
    assert "abc" not in str(raised.value)


@pytest.mark.asyncio
async def test_in_body_error_is_returned_not_raised() -> None:
    page = await _fetch(in_body_error="crawl_not_found")
    assert page.text == ""
    assert page.error is not None and page.error.tag == "crawl_not_found"


@pytest.mark.asyncio
async def test_delay_holds_the_call_open() -> None:
    call = asyncio.create_task(_fetch(delay=30))
    await asyncio.sleep(0.05)
    assert not call.done()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_leak_url_raises_with_the_url_in_the_message() -> None:
    with pytest.raises(RuntimeError, match="sentinel-url"):
        await _fetch("https://example.com/sentinel-url", leak_url=True)


@pytest.mark.asyncio
async def test_raw_echoes_the_request_but_not_the_url() -> None:
    page = await _fetch("https://example.com/sentinel-url", max_chars=5, account="acme")
    assert page.raw == {"request": {"max_chars": 5, "options": {"account": "acme"}}}
