"""Unit tests for the security headers middleware."""

import pytest
from starlette.types import Message, Receive, Scope, Send

from gateway.main import SecurityHeadersMiddleware


@pytest.mark.asyncio
async def test_a_response_start_without_headers_still_gets_them() -> None:
    """``headers`` is optional on ``http.response.start``, so the middleware must not assume it."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 204})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> Message:
        return {"type": "http.request", "body": b""}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {"type": "http", "method": "GET", "path": "/v1/models", "headers": [], "query_string": b""}
    await SecurityHeadersMiddleware(app)(scope, receive, send)

    names = {name.lower() for name, _ in sent[0]["headers"]}
    assert b"x-content-type-options" in names
