"""Tests for error handling on streaming requests."""

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.exceptions import ContextLengthExceededError
from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.usage import UsageLog

from .conftest import MODEL_NAME, build_test_client

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
    """Test that a streaming request whose stream cannot be opened returns an HTTP error.

    When the stream cannot be created, the gateway returns a proper HTTP error
    response rather than starting a stream and emitting an SSE error event.
    """
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
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
    ("exc", "code", "message"),
    [
        (_UpstreamStatusError(503), "provider_error", "The provider failed the request with HTTP 503"),
        (_UpstreamStatusError(429), "upstream_rate_limited", "upstream detail [redacted] at [redacted]"),
        (
            ContextLengthExceededError("prompt is too long", status_code=400),
            "context_length_exceeded",
            "prompt is too long",
        ),
        (
            TypeError("AsyncClientV2.chat_stream() got an unexpected keyword argument 'n'"),
            "provider_error",
            "The provider serving this model does not accept the 'n' parameter",
        ),
    ],
    ids=["provider", "rate-limited", "context-length", "rejected-param"],
)
def test_a_chat_stream_the_provider_fails_ends_in_a_coded_error_event(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    exc: Exception,
    code: str,
    message: str,
) -> None:
    """The status line has already gone out as 200, so the event is where the caller learns the code.

    The message is the detail the same failure gets unstreamed, so it names a
    cause where that detail does and stays fixed where it does not.
    """
    response = _stream_chat_failing_with(client, master_key_header, exc)

    assert response.status_code == 200, response.text
    error = _error_event(response.text)["error"]
    assert error == {"message": message, "type": "server_error", "code": code}
    assert "sk-live-abc123" not in response.text
    assert "internal.example" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")

    row = _usage_row(db_session_factory)
    assert row.status == "error"
    assert row.status_code == getattr(exc, "status_code", 400)


def _stream_chat_failing_before_its_first_chunk(client: TestClient, headers: dict[str, str], exc: Exception) -> Any:
    """Stream a chat completion whose provider fails with ``exc`` when the stream is first read."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": _STREAM_USER}, headers=headers)
    assert created.status_code == 200, created.text

    async def _acompletion(**_kwargs: Any) -> Any:
        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            raise exc
            yield  # pragma: no cover - makes this an async generator

        return _stream()

    with patch("gateway.api.routes.chat.acompletion", side_effect=_acompletion):
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": _STREAM_USER,
                "n": 1,
                "stream": True,
            },
            headers=headers,
        )


def test_a_param_the_provider_refuses_before_the_first_chunk_is_answered_with_a_400(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """An SDK that refuses a keyword only once the stream is read gets the unstreamed status, not a 200."""
    exc = TypeError("AsyncClientV2.chat_stream() got an unexpected keyword argument 'n'")
    response = _stream_chat_failing_before_its_first_chunk(client, master_key_header, exc)

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "The provider serving this model does not accept the 'n' parameter"}
    assert "chat_stream" not in response.text

    row = _usage_row(db_session_factory)
    assert row.status == "error"
    assert row.status_code == 400


def test_an_upstream_outage_before_the_first_chunk_is_answered_with_a_502(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    response = _stream_chat_failing_before_its_first_chunk(client, master_key_header, _UpstreamStatusError(503))

    assert response.status_code == 502, response.text
    assert response.json() == {"detail": "The provider failed the request with HTTP 503"}
    assert "sk-live-abc123" not in response.text
    assert "x-should-retry" not in response.headers

    row = _usage_row(db_session_factory)
    assert row.status == "error"
    assert row.status_code == 503


def test_a_local_adapter_fault_before_the_first_chunk_tells_a_client_not_to_retry(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """An adapter that breaks converting the request fails the same way on every resend.

    An OpenAI or Anthropic SDK client would otherwise retry the 502 with backoff.
    """
    response = _stream_chat_failing_before_its_first_chunk(client, master_key_header, KeyError("content"))

    assert response.status_code == 502, response.text
    assert response.json() == {"detail": "LLM provider error"}
    assert response.headers["x-should-retry"] == "false"

    row = _usage_row(db_session_factory)
    assert row.status == "error"
    assert row.status_code == 502


def test_an_unstreamed_local_adapter_fault_tells_a_client_not_to_retry(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=KeyError("content")):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == 502, response.text
    assert response.json() == {"detail": "LLM provider error"}
    assert response.headers["x-should-retry"] == "false"


def test_a_messages_local_adapter_fault_tells_a_client_not_to_retry(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, side_effect=KeyError("content")):
        response = client.post(f"{API_ROOT}/messages", json=messages_request_body, headers=api_key_header)

    assert response.status_code == 500, response.text
    assert response.json()["detail"]["error"]["type"] == "api_error"
    assert response.headers["x-should-retry"] == "false"


def test_with_the_hold_off_a_failure_before_the_first_chunk_ends_a_200_in_a_coded_event(
    test_config: GatewayConfig,
    clean_database: None,
    master_key_header: dict[str, str],
) -> None:
    """``streaming_first_chunk_hold_ms: 0`` commits the response as soon as the stream opens."""
    exc = TypeError("AsyncClientV2.chat_stream() got an unexpected keyword argument 'n'")
    for client in build_test_client(test_config.model_copy(update={"streaming_first_chunk_hold_ms": 0})):
        response = _stream_chat_failing_before_its_first_chunk(client, master_key_header, exc)

    assert response.status_code == 200, response.text
    assert _error_event(response.text)["error"] == {
        "message": "The provider serving this model does not accept the 'n' parameter",
        "type": "server_error",
        "code": "provider_error",
    }


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


def test_a_responses_stream_that_fails_before_its_first_event_is_answered_with_its_status(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    async def _stream() -> AsyncIterator[_FakeStreamEvent]:
        raise _UpstreamStatusError(400)
        yield  # pragma: no cover - makes this an async generator

    with patch("gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=_stream()):
        response = client.post(
            f"{API_ROOT}/responses", json={**responses_request_body, "stream": True}, headers=master_key_header
        )

    assert response.status_code == 400, response.text
    assert "event: error" not in response.text
    assert "sk-live-abc123" not in response.text


def test_a_messages_stream_that_fails_before_its_first_event_is_answered_with_its_status(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    async def _amessages(**_kwargs: Any) -> Any:
        async def _stream() -> AsyncIterator[Any]:
            raise _UpstreamStatusError(503)
            yield  # pragma: no cover - makes this an async generator

        return _stream()

    with patch("gateway.api.routes.messages.amessages", new=_amessages):
        response = client.post(
            f"{API_ROOT}/messages", json={**messages_request_body, "stream": True}, headers=api_key_header
        )

    assert response.status_code == 500, response.text
    assert response.json()["detail"]["error"]["type"] == "api_error"
    assert "event: error" not in response.text
    assert "sk-live-abc123" not in response.text


# ---------------------------------------------------------------------------
# A provider stream that ends without finishing is an error, not a success
# ---------------------------------------------------------------------------


def _chunk(content: str | None = None, *, finish_reason: Any = None, usage: bool = False) -> ChatCompletionChunk:
    choices = (
        []
        if content is None and finish_reason is None
        else [ChunkChoice(index=0, delta=ChoiceDelta(content=content), finish_reason=finish_reason)]
    )
    return ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=0,
        model=MODEL_NAME,
        choices=choices,
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=3, total_tokens=6) if usage else None,
    )


def _stream_chat(
    client: TestClient, headers: dict[str, str], chunks: list[ChatCompletionChunk], exc: Exception | None = None
) -> Any:
    """Stream a chat completion whose provider sends ``chunks`` and then ends, or raises ``exc``."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": _STREAM_USER}, headers=headers)
    assert created.status_code == 200, created.text

    async def _acompletion(**_kwargs: Any) -> Any:
        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in chunks:
                yield chunk
            if exc is not None:
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


_CONTENT = [_chunk("one"), _chunk("two"), _chunk("three")]


@pytest.mark.parametrize(
    "exc",
    [None, httpx.RemoteProtocolError("peer closed connection without sending complete message body")],
    ids=["closed-without-finishing", "chunked-body-cut"],
)
def test_a_chat_stream_the_provider_ends_early_is_an_interrupted_stream_error(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    exc: Exception | None,
) -> None:
    """Three content chunks and no ``finish_reason``: the client is told the text is truncated."""
    response = _stream_chat(client, master_key_header, _CONTENT, exc)

    assert response.status_code == 200, response.text
    assert _error_event(response.text)["error"] == {
        "message": "The provider's stream ended before it finished",
        "type": "server_error",
        "code": "provider_stream_interrupted",
    }
    assert _usage_row(db_session_factory).status == "error"


@pytest.mark.parametrize(
    "tail",
    [
        [_chunk(finish_reason="stop")],
        [_chunk(finish_reason="stop"), _chunk(usage=True)],
        [_chunk(finish_reason="tool_calls", usage=True)],
    ],
    ids=["finish", "finish-then-usage-only", "tool-calls-with-usage"],
)
def test_a_chat_stream_that_finishes_ends_cleanly(
    client: TestClient,
    master_key_header: dict[str, str],
    tail: list[ChatCompletionChunk],
) -> None:
    response = _stream_chat(client, master_key_header, [*_CONTENT, *tail])

    assert response.status_code == 200, response.text
    assert '"error"' not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def _choice_chunk(index: int, content: str | None = None, finish_reason: Any = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=0,
        model=MODEL_NAME,
        choices=[ChunkChoice(index=index, delta=ChoiceDelta(content=content), finish_reason=finish_reason)],
    )


@pytest.mark.parametrize(
    ("chunks", "interrupted"),
    [
        ([_choice_chunk(0, "a"), _choice_chunk(1, "b"), _choice_chunk(0, finish_reason="stop")], True),
        ([_choice_chunk(0, "a"), _choice_chunk(0, finish_reason="stop"), _choice_chunk(1, "b")], True),
        (
            [
                _choice_chunk(0, "a"),
                _choice_chunk(1, "b"),
                _choice_chunk(0, finish_reason="stop"),
                _choice_chunk(1, finish_reason="stop"),
            ],
            False,
        ),
    ],
    ids=["second-cut", "second-starts-after-first-finished", "both-finish"],
)
def test_a_chat_stream_with_several_choices_is_complete_only_when_each_finishes(
    client: TestClient,
    master_key_header: dict[str, str],
    chunks: list[ChatCompletionChunk],
    interrupted: bool,
) -> None:
    """With ``n`` above 1, one finished choice does not stand in for another that was cut."""
    response = _stream_chat(client, master_key_header, chunks)

    assert response.status_code == 200, response.text
    if interrupted:
        assert _error_event(response.text)["error"]["code"] == "provider_stream_interrupted"
    else:
        assert '"error"' not in response.text
