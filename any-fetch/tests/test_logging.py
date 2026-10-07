"""The filter that keeps provider URLs, and the URLs fetched, out of httpx's log."""

import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from any_fetch import AnyFetch, FetchedPage
from any_fetch._logging import REDACTED, RedactProviderUrls, install, provider_call
from any_fetch.providers import builtin

URL = "https://site.example/sentinel-url"


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="page")))


def _httpx_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "httpx"]


class _HostFetcher:
    """A host's fetcher that requests the page itself, as otari's does."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, url: str, *, max_chars: int | None) -> FetchedPage:
        response = await self.client.get(url)
        return FetchedPage(
            url=url, final_url=url, text=response.text, content_type="text/plain", cost_source="none", raw={}
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture
def _no_builtin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(builtin, "_factory", None)
    yield


@pytest.mark.asyncio
async def test_a_request_made_during_a_provider_call_is_logged_without_its_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client:
        with provider_call("p"):
            await client.get(URL)
    [message] = _httpx_messages(caplog)
    assert "sentinel-url" not in message
    assert f"GET {REDACTED}" in message


@pytest.mark.asyncio
async def test_a_request_made_outside_one_keeps_its_url(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client:
        await client.get(URL)
    assert URL in _httpx_messages(caplog)[0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_builtin")
async def test_the_hosts_builtin_runs_inside_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")

    def factory(**settings: Any) -> _HostFetcher:
        return _HostFetcher(settings["client"])

    AnyFetch.register_builtin(factory)
    async with _client() as client, AnyFetch.create("builtin", client=client) as fetcher:
        assert (await fetcher.fetch(URL)).text == "page"
    [message] = _httpx_messages(caplog)
    assert "sentinel-url" not in message


def test_the_filter_is_installed_once() -> None:
    install()
    filters = logging.getLogger("httpx").filters
    assert len([existing for existing in filters if isinstance(existing, RedactProviderUrls)]) == 1
