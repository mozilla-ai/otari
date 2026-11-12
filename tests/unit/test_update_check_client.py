"""The one request an update check sends, and how a reply that cannot be used is refused."""

import asyncio
import json
import uuid
from datetime import timedelta

import httpx
import pytest

from gateway.schemas.update_check import UpdateCheckReply
from gateway.services.update_check._client import MAX_REPLY_BYTES, UpdateCheckFailedError, send_check

_URL = "https://updates.example/api/v1/update-checks"
_INSTALL = uuid.UUID("2f1c7e0e-8a43-4f53-9a6e-2c1d4f1e9b70")


async def _send(handler: httpx.MockTransport) -> UpdateCheckReply:
    return await send_check(_URL, version="0.30.1", install_id=_INSTALL, transport=handler)


@pytest.mark.asyncio
async def test_the_check_sends_the_version_and_install_id_and_nothing_else() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"latest_version": "0.31.0", "release_url": None, "advisories": []})

    reply = await send_check(_URL, version="0.30.1", install_id=_INSTALL, transport=httpx.MockTransport(handler))

    assert reply.latest_version == "0.31.0"
    [request] = seen
    assert request.method == "POST"
    assert str(request.url) == _URL
    assert json.loads(request.content) == {"version": "0.30.1", "install_id": str(_INSTALL)}
    assert "authorization" not in request.headers
    assert "cookie" not in request.headers


@pytest.mark.asyncio
async def test_a_refusal_carries_the_servers_retry_after() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(429, headers={"Retry-After": "120"}))
    with pytest.raises(UpdateCheckFailedError) as raised:
        await _send(transport)
    assert raised.value.retry_after == timedelta(seconds=120)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"advisories": [{"id": "GHSA-1"}]}),
        httpx.Response(200, content=b'{"latest_version": "' + b"9" * MAX_REPLY_BYTES + b'"}'),
        httpx.Response(302, headers={"Location": "https://elsewhere.example/"}),
        httpx.Response(200, json={"latest_version": "0.31.0", "release_url": "javascript:alert(1)"}),
        httpx.Response(
            200,
            json={"advisories": [{"id": "GHSA-1", "summary": "s", "severity": "high", "url": "data:text/html,x"}]},
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_reply_that_cannot_be_used_fails_the_check(response: httpx.Response) -> None:
    with pytest.raises(UpdateCheckFailedError):
        await _send(httpx.MockTransport(lambda _request: response))


@pytest.mark.asyncio
async def test_an_unreachable_server_fails_the_check() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(UpdateCheckFailedError):
        await _send(httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_a_long_advisory_is_shortened_rather_than_failing_the_check() -> None:
    advisory = {"id": "GHSA-1", "summary": "x" * 1024, "severity": "high", "url": "https://example/1"}
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"advisories": [advisory] * 60}))

    reply = await _send(transport)

    assert len(reply.advisories) == 50
    assert len(reply.advisories[0].summary) == 500


@pytest.mark.asyncio
async def test_a_reply_that_never_finishes_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.services.update_check._client.TIMEOUT_SECONDS", 0.05)

    async def stalls(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    with pytest.raises(UpdateCheckFailedError):
        await _send(httpx.MockTransport(stalls))
