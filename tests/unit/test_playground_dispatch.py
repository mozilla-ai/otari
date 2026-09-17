"""Forwarding a Playground completion to the data-plane gateway.

The transport half of ``services/playground_dispatch``, which is the half that
needs no database: where the request is addressed, what travels on it, and what
comes back. The key it presents and the route that calls it are covered in
``tests/integration/test_playground_hosted_dispatch.py``.
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from gateway.core.config import API_ROOT
from gateway.services import playground_dispatch

# Bound before any test patches the name, so the factory below builds a real
# client rather than calling itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _Stream(httpx.AsyncByteStream):
    """A response body the mock transport hands over unread.

    ``httpx.Response(content=...)`` marks its stream consumed, and the code under
    test streams what it is given, so a mocked answer has to arrive as a stream
    or it cannot be read at all.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._payload


def _answer(status_code: int, payload: bytes, **headers: str) -> httpx.Response:
    return httpx.Response(status_code, stream=_Stream(payload), headers=headers)


def _client_factory(handler: object) -> object:
    """An ``httpx.AsyncClient`` that answers from ``handler`` instead of the network."""

    def build(**kwargs: object) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    return build


async def _drain(body: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in body])


def test_the_url_is_the_data_plane_plus_the_api_root() -> None:
    """``data_plane_url`` is a base address, so the path is appended here."""
    assert (
        playground_dispatch.completions_url("https://gateway.example.com")
        == f"https://gateway.example.com{API_ROOT}/chat/completions"
    )


@pytest.mark.asyncio
async def test_the_request_carries_the_key_and_the_body_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the data plane is handed: the caller's body, under the minted key.

    No cookie in particular. The data plane authenticates the key, and a
    dashboard session forwarded beside it would be a second credential meaning
    something else on a deployment that is not this one.
    """
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["cookie"] = request.headers.get("cookie")
        seen["body"] = request.content
        return _answer(200, b'{"id": "chat-1"}', **{"content-type": "application/json"})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    status_code, _headers, body = await playground_dispatch.forward_completion(
        url="https://gateway.example.com/api/v1/chat/completions",
        api_key="gw-secret",
        payload={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    await _drain(body)

    assert status_code == 200
    assert seen["url"] == "https://gateway.example.com/api/v1/chat/completions"
    assert seen["authorization"] == "Bearer gw-secret"
    assert seen["cookie"] is None
    assert b'"model":"openai:gpt-4o"' in seen["body"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_the_answer_is_returned_as_the_data_plane_wrote_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Status, content type and bytes pass through unchanged.

    A streamed completion is the case this exists for: the page reads the same
    event stream it reads standalone, so nothing here parses or re-frames it.
    """
    chunks = b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\ndata: [DONE]\n\n'

    def handler(_request: httpx.Request) -> httpx.Response:
        return _answer(200, chunks, **{"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    status_code, headers, body = await playground_dispatch.forward_completion(
        url="https://gateway.example.com/api/v1/chat/completions",
        api_key="gw-secret",
        payload={"stream": True},
    )

    assert status_code == 200
    assert headers["content-type"] == "text/event-stream"
    assert await _drain(body) == chunks


@pytest.mark.asyncio
async def test_a_refusal_from_the_data_plane_is_forwarded_rather_than_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget refusal or an unknown model is the gateway's answer to give.

    The control plane adds nothing to it: re-deriving the reason here is how the
    page comes to show something the gateway did not say.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return _answer(402, b'{"detail": "Budget exceeded"}', **{"content-type": "application/json"})

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    status_code, _headers, body = await playground_dispatch.forward_completion(
        url="https://gateway.example.com/api/v1/chat/completions",
        api_key="gw-secret",
        payload={},
    )

    assert status_code == 402
    assert b"Budget exceeded" in await _drain(body)


@pytest.mark.asyncio
async def test_framing_headers_are_not_copied_onto_the_new_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The body is re-framed on the way through, so its old framing cannot ride along.

    A ``Content-Length`` taken from the upstream describes a message this
    response is not, and the browser is the one that has to reconcile it.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return _answer(
            200,
            b'{"id": "chat-1"}',
            **{"content-type": "application/json", "x-request-id": "abc", "content-length": "16"},
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    _status, headers, body = await playground_dispatch.forward_completion(
        url="https://gateway.example.com/api/v1/chat/completions",
        api_key="gw-secret",
        payload={},
    )
    await _drain(body)

    assert "content-length" not in {name.lower() for name in headers}
    assert headers["x-request-id"] == "abc"


@pytest.mark.asyncio
async def test_an_unreachable_data_plane_raises_rather_than_returning_a_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route turns this into a 502; it must not look like an answer."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    with pytest.raises(httpx.HTTPError):
        await playground_dispatch.forward_completion(
            url="https://gateway.example.com/api/v1/chat/completions",
            api_key="gw-secret",
            payload={},
        )
