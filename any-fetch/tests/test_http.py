"""The HTTP client a provider sends through."""

import re
from typing import Any

import httpx
import pytest

from any_fetch import ProviderError, __version__
from any_fetch._http import USER_AGENT, ProviderHttp

SENTINEL_URL = "https://api.example/contents?url=sentinel-url&key=sentinel-key"
TIMEOUT = {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}
# RFC 9110's product token, which the version must fit, a pre-release suffix included.
PRODUCT_TOKEN = re.compile(r"^any-fetch/[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


@pytest.mark.asyncio
async def test_a_given_client_is_used_with_the_timeout_and_left_open() -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(answer), timeout=1.0)
    http = ProviderHttp("p", client, 7.5)
    await http.request("GET", "https://api.example/one")
    await http.request("POST", "https://api.example/two", json={})
    await http.aclose()
    assert [request.extensions["timeout"] for request in sent] == [TIMEOUT, TIMEOUT]
    assert not client.is_closed
    assert client.timeout == httpx.Timeout(1.0)
    await client.aclose()


@pytest.mark.asyncio
async def test_without_a_client_one_is_opened_on_first_use_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[httpx.AsyncClient] = []
    original = httpx.AsyncClient

    def open_client(**kwargs: object) -> httpx.AsyncClient:
        assert kwargs == {"timeout": 7.5}
        client = original(transport=httpx.MockTransport(lambda request: httpx.Response(204)))
        opened.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", open_client)
    http = ProviderHttp("p", None, 7.5)
    assert opened == []
    assert (await http.request("GET", "https://api.example/")).status_code == 204
    await http.request("GET", "https://api.example/")
    assert len(opened) == 1
    await http.aclose()
    assert opened[0].is_closed


def test_the_user_agent_names_the_library_and_its_version_as_one_product_token() -> None:
    assert USER_AGENT == f"any-fetch/{__version__}"
    assert PRODUCT_TOKEN.match(USER_AGENT), USER_AGENT


@pytest.mark.asyncio
async def test_every_request_through_a_hosts_client_names_the_library_beside_the_providers_headers() -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        http = ProviderHttp("p", client, 7.5)
        await http.request("GET", "https://api.example/one")
        await http.request("POST", "https://api.example/two", headers={"Authorization": "k"}, json={})
        # The client itself is left as it was.
        assert client.headers["User-Agent"] == f"python-httpx/{httpx.__version__}"
    assert [request.headers["User-Agent"] for request in sent] == [USER_AGENT, USER_AGENT]
    assert sent[1].headers["Authorization"] == "k"


@pytest.mark.asyncio
async def test_every_request_through_its_own_client_names_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    original = httpx.AsyncClient

    def open_client(**kwargs: Any) -> httpx.AsyncClient:
        return original(transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", open_client)
    http = ProviderHttp("p", None, 7.5)
    await http.request("GET", "https://api.example/", headers={"Authorization": "k"})
    await http.aclose()
    assert (sent[0].headers["User-Agent"], sent[0].headers["Authorization"]) == (USER_AGENT, "k")


@pytest.mark.asyncio
async def test_a_hosts_client_with_a_user_agent_of_its_own_keeps_it() -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(answer), headers={"User-Agent": "host/1.0"})
    async with client:
        await ProviderHttp("p", client, 7.5).request("GET", "https://api.example/")
    assert sent[0].headers["User-Agent"] == "host/1.0"


@pytest.mark.asyncio
async def test_a_hosts_client_that_carries_no_user_agent_is_filled_in() -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    del client.headers["User-Agent"]
    async with client:
        await ProviderHttp("p", client, 7.5).request("GET", "https://api.example/")
    assert sent[0].headers["User-Agent"] == USER_AGENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "tag"),
    [(httpx.ConnectTimeout(SENTINEL_URL), "timeout"), (httpx.ConnectError(SENTINEL_URL), "network")],
)
async def test_a_transport_failure_becomes_a_provider_error_without_its_text(failure: Exception, tag: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(ProviderError) as raised:
            await ProviderHttp("p", client, 7.5).request("GET", SENTINEL_URL)
    assert raised.value.tag == tag
    # Raised after the handler, so the failure, whose text carries the URL, is not even its context.
    assert (raised.value.__cause__, raised.value.__context__) == (None, None)
    assert "sentinel" not in str(raised.value)


@pytest.mark.asyncio
async def test_a_header_that_is_not_ascii_becomes_a_provider_error_without_the_header() -> None:
    # httpx's own error for it quotes the whole header, so a key pasted with a stray character would leak.
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        with pytest.raises(ProviderError) as raised:
            await ProviderHttp("p", client, 7.5).request(
                "GET", "https://api.example/", headers={"Authorization": "Bearer sentinel-key\u00a0"}
            )
    assert (raised.value.tag, raised.value.__context__) == ("invalid_header", None)
    assert "sentinel" not in str(raised.value) and "sentinel" not in repr(raised.value)
