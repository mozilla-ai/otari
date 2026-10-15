"""The filter that keeps provider URLs out of httpx's log."""

import logging
import subprocess
import sys
from typing import Any

import httpx
import pytest

from any_search import AnySearch, SearchResult, TimeRange
from any_search._logging import LOGGERS, REDACTED, RedactProviderUrls, install, provider_call
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
@pytest.mark.usefixtures("log_filter")
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
@pytest.mark.usefixtures("log_filter")
async def test_a_request_made_outside_one_keeps_its_url(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client:
        await client.get(URL)
    assert URL in _httpx_messages(caplog)[0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
async def test_search_runs_inside_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client, _Calling(client=client) as provider:
        await provider.search("sentinel-query")
    [message] = _httpx_messages(caplog)
    assert "sentinel-query" not in message


def test_importing_the_package_installs_no_filter() -> None:
    code = (
        "import logging, any_search, any_search.cli; "
        "from any_search._logging import LOGGERS, RedactProviderUrls; "
        "print(any(isinstance(f, RedactProviderUrls) for n in LOGGERS for f in logging.getLogger(n).filters))"
    )
    answer = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert answer.stdout.strip() == "False"


@pytest.mark.asyncio
async def test_without_the_filter_a_provider_call_keeps_its_url(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="httpx")
    async with _client() as client:
        with provider_call("p"):
            await client.get(URL)
    assert URL in _httpx_messages(caplog)[0]


@pytest.mark.usefixtures("log_filter")
def test_the_filter_is_installed_once() -> None:
    install()
    for name in LOGGERS:
        filters = logging.getLogger(name).filters
        assert len([existing for existing in filters if isinstance(existing, RedactProviderUrls)]) == 1


def _encode_a_path_over_http2() -> None:
    """Encode a request's headers as an HTTP/2 client does, which hpack logs at DEBUG."""
    hpack = pytest.importorskip("hpack")
    hpack.Encoder().encode([(b":method", b"GET"), (b":path", b"/search?q=sentinel-query&api_key=sentinel-key")])


@pytest.mark.usefixtures("log_filter")
def test_hpack_records_during_a_provider_call_carry_no_path(caplog: pytest.LogCaptureFixture) -> None:
    """Over HTTP/2, hpack logs each header it encodes, the path with its query included."""
    caplog.set_level(logging.DEBUG)
    with provider_call("p"):
        _encode_a_path_over_http2()
    messages = [record.getMessage() for record in caplog.records if record.name.startswith("hpack")]
    assert messages
    assert [message for message in messages if "sentinel" in message] == []


@pytest.mark.usefixtures("log_filter")
def test_hpack_records_outside_one_are_left_alone(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    _encode_a_path_over_http2()
    assert any("sentinel-query" in record.getMessage() for record in caplog.records if record.name.startswith("hpack"))
