"""Tests for error handling on streaming requests."""

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.exceptions import ContextLengthExceededError
from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.usage import UsageLog

from .conftest import MODEL_NAME

_STREAM_USER = "stream-error-user"
_UPSTREAM_SECRET = "upstream detail sk-live-abc123 at https://internal.example/v1"


class _UpstreamStatusError(Exception):
    """A provider failure carrying the HTTP status the provider answered with."""

    def __init__(self, status_code: int) -> None:
        super().__init__(_UPSTREAM_SECRET)
        self.status_code = status_code


def _first_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=0,
        model=MODEL_NAME,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason=None)],
    )


def _stream_chat_failing_with(client: TestClient, headers: dict[str, str], exc: Exception) -> Any:
    """Stream a chat completion whose provider fails with ``exc`` after its first chunk."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": _STREAM_USER}, headers=headers)
    assert created.status_code == 200, created.text

    async def _acompletion(**_kwargs: Any) -> Any:
        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            yield _first_chunk()
            raise exc

        return _stream()

    with patch("gateway.api.routes.chat.acompletion", side_effect=_acompletion):
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": _STREAM_USER,
                "stream": True,
            },
            headers=headers,
        )


def _error_event(text: str) -> dict[str, Any]:
    """The one ``data:`` payload in an SSE body that carries an ``error`` object."""
    events: list[dict[str, Any]] = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    errors = [event for event in events if "error" in event]
    assert len(errors) == 1, text
    return errors[0]


def _usage_row(make_session: Callable[[], Session], *, timeout: float = 5.0) -> UsageLog:
    """The request's usage row, polled because the background log writer can lag the response."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            row = db.execute(select(UsageLog).where(UsageLog.user_id == _STREAM_USER)).scalar_one_or_none()
            if row is not None:
                return row
        assert time.monotonic() < deadline, "the usage row was never written"
        time.sleep(0.1)


def test_streaming_creation_error_returns_http_error(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a streaming request to an invalid model returns an HTTP error.

    When the stream cannot be created (e.g., invalid model, missing API key),
    the gateway returns a proper HTTP error response rather than starting a
    stream and emitting an SSE error event.
    """
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:totally-invalid-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        headers=api_key_header,
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider error"}


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (_UpstreamStatusError(503), "provider_error"),
        (_UpstreamStatusError(429), "upstream_rate_limited"),
        (ContextLengthExceededError("prompt is too long", status_code=400), "context_length_exceeded"),
    ],
    ids=["provider", "rate-limited", "context-length"],
)
def test_a_chat_stream_the_provider_fails_ends_in_a_coded_error_event(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    exc: Exception,
    code: str,
) -> None:
    """The status line has already gone out as 200, so the event is where the caller learns the code."""
    response = _stream_chat_failing_with(client, master_key_header, exc)

    assert response.status_code == 200, response.text
    error = _error_event(response.text)["error"]
    assert error == {"message": "An error occurred during streaming", "type": "server_error", "code": code}
    assert "sk-live-abc123" not in response.text
    assert "internal.example" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")

    row = _usage_row(db_session_factory)
    assert row.status == "error"
    assert row.status_code == getattr(exc, "status_code", None)


def test_a_chat_stream_the_gateway_fails_is_not_blamed_on_the_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    response = _stream_chat_failing_with(client, master_key_header, RuntimeError("gateway bug"))

    assert response.status_code == 200, response.text
    assert "code" not in _error_event(response.text)["error"]
    assert _usage_row(db_session_factory).status == "error"


class _FakeStreamEvent:
    def __init__(self, event_type: str) -> None:
        self.type = event_type
        self.response = None

    def model_dump_json(self, *, exclude_none: bool = False) -> str:
        return json.dumps({"type": self.type})


def test_a_responses_stream_the_provider_fails_ends_in_a_coded_error_event(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    async def _stream() -> AsyncIterator[_FakeStreamEvent]:
        yield _FakeStreamEvent("response.created")
        raise _UpstreamStatusError(500)

    with patch("gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=_stream()):
        response = client.post(
            f"{API_ROOT}/responses", json={**responses_request_body, "stream": True}, headers=master_key_header
        )

    assert response.status_code == 200, response.text
    assert "event: error" in response.text
    assert _error_event(response.text)["error"]["code"] == "provider_error"
    assert "sk-live-abc123" not in response.text
