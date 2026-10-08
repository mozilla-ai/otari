"""The HTTP client a provider sends through."""

import httpx
import pytest

from any_search import ProviderError
from any_search._http import ProviderHttp

SENTINEL_URL = "https://api.example/search?q=sentinel-query&key=sentinel-key"
TIMEOUT = {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}


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
    assert raised.value.__suppress_context__
    assert "sentinel" not in str(raised.value)
