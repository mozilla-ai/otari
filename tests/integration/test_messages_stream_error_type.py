"""A failure after a streaming ``/api/v1/messages`` response committed keeps its type.

The status line is already a 200 by then, so the SSE ``error`` event is the only
thing that tells a client whether the failure was transient. An upstream
``overloaded_error`` used to reach the client as a generic ``api_error``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import anthropic
import httpx
from any_llm.types.messages import MessageStreamEvent
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

from .conftest import MODEL_NAME
from .test_messages_streaming_usage import _message_start, _seed_budgeted_user, _text_delta


def _overloaded_mid_stream() -> anthropic.APIStatusError:
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}, "request_id": "req_x"}
    response = httpx.Response(200, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return anthropic.APIStatusError(str(body), response=response, body=body)


async def _stream_then_overload(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
    async def _gen() -> AsyncIterator[MessageStreamEvent]:
        yield _message_start()
        yield _text_delta("hello")
        raise _overloaded_mid_stream()

    return _gen()


def _error_events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(frame.split("\ndata: ", 1)[1]) for frame in body.split("\n\n") if frame.startswith("event: error\n")
    ]


def test_an_upstream_overload_mid_stream_reaches_the_client_as_overloaded(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _seed_budgeted_user(client, master_key_header, "stream-overloaded")

    with patch("gateway.api.routes.messages.amessages", new=_stream_then_overload):
        response = client.post(
            f"{API_ROOT}/messages",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "stream": True,
                "metadata": {"user_id": "stream-overloaded"},
            },
            headers=master_key_header,
        )
        body = response.text

    assert response.status_code == 200
    assert "hello" in body
    [error] = _error_events(body)
    assert error["error"]["type"] == "overloaded_error"
    assert "req_x" not in body
