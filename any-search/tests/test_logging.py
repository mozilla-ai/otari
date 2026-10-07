"""The filter that keeps provider URLs out of httpx's log."""

import logging
from typing import Any

import httpx
import pytest

from any_search import AnySearch, SearchResult, TimeRange
from any_search._logging import REDACTED, RedactProviderUrls, install, provider_call
from any_search.providers.fake import FakeProvider

URL = "https://api.example/search?q=sentinel-query"


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))


def _httpx_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "httpx"]


class _Calling(AnySearch):
    """A provider that sends the query in its URL, the case the filter exists for."""

    METADATA = FakeProvider.METADATA.model_copy(update={"name": "calling", "options": []})

    async def _search(
        self, query: str, *, max_results: int | None, time_range: TimeRange | None, options: dict[str, Any]
    ) -> SearchResult:
        await self._http.request("GET", "https://api.example/search", params={"q": query})
        return SearchResult(provider="calling", hits=[], cost_source="none", raw={})


@pytest.mark.asyncio
async def test_a_request_made_during_a_provider_call_is_logged_without_its_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client:
        with provider_call("p"):
            await client.get(URL)
    [message] = _httpx_messages(caplog)
    assert "sentinel-query" not in message
    assert f"GET {REDACTED}" in message


@pytest.mark.asyncio
async def test_a_request_made_outside_one_keeps_its_url(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client:
        await client.get(URL)
    assert URL in _httpx_messages(caplog)[0]


@pytest.mark.asyncio
async def test_search_runs_inside_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client, _Calling(client=client) as provider:
        await provider.search("sentinel-query")
    [message] = _httpx_messages(caplog)
    assert "sentinel-query" not in message


def test_the_filter_is_installed_once() -> None:
    install()
    filters = logging.getLogger("httpx").filters
    assert len([existing for existing in filters if isinstance(existing, RedactProviderUrls)]) == 1
