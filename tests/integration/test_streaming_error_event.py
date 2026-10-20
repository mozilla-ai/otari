"""Tests for error handling on streaming requests."""

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.exceptions import ContextLengthExceededError
from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice, CompletionUsage
from any_llm.types.responses import Response, ResponseStreamEvent
from fastapi.testclient import TestClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
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


_GEMINI_503 = {"provider": "gemini", "status": 503, "tag": "http_error", "retryable": True}


@pytest.mark.parametrize(
    ("exc", "code", "message", "provider_failure"),
    [
        (_UpstreamStatusError(503), "provider_error", "The provider failed the request with HTTP 503", _GEMINI_503),
        (_UpstreamStatusError(429), "upstream_rate_limited", "upstream detail [redacted] at [redacted]", None),
        (
            ContextLengthExceededError("prompt is too long", status_code=400),
            "context_length_exceeded",
            "prompt is too long",
            None,
        ),
        (
            TypeError("AsyncClientV2.chat_stream() got an unexpected keyword argument 'n'"),
            "provider_error",
            "The provider serving this model does not accept the 'n' parameter",
            None,
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
    provider_failure: dict[str, Any] | None,
) -> None:
    """The status line has already gone out as 200, so the event is where the caller learns the code.

    The message is the detail the same failure gets unstreamed, so it names a
    cause where that detail does and stays fixed where it does not. Only a
    failure on the provider's own side names it.
    """
    response = _stream_chat_failing_with(client, master_key_header, exc)

    assert response.status_code == 200, response.text
    error = _error_event(response.text)["error"]
    expected: dict[str, Any] = {"message": message, "type": "server_error", "code": code}
    if provider_failure is not None:
        expected["provider_failure"] = provider_failure
    assert error == expected
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


def _response_snapshot(status: str, usage: ResponseUsage | None = None) -> Response:
    return Response(
        id="resp_cut",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        status=cast(Any, status),
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


_RESPONSES_OPENING_EVENTS: list[ResponseStreamEvent] = [
    ResponseCreatedEvent(type="response.created", response=_response_snapshot("in_progress"), sequence_number=0),
    ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id="msg_cut",
        output_index=0,
        content_index=0,
        delta="Hel",
        logprobs=[],
        sequence_number=1,
    ),
]


def _scripted_aresponses(*rounds: list[ResponseStreamEvent]) -> Callable[..., Any]:
    """A provider that answers each call with the next round's events and then closes the stream."""
    remaining = iter(rounds)

    async def _aresponses(**_kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        events = next(remaining)

        async def _stream() -> AsyncIterator[ResponseStreamEvent]:
            for event in events:
                yield event

        return _stream()

    return _aresponses


def _post_responses_stream(client: TestClient, headers: dict[str, str], **fields: Any) -> Any:
    created = client.post(f"{API_ROOT}/users", json={"user_id": _STREAM_USER}, headers=headers)
    assert created.status_code == 200, created.text
    return client.post(
        f"{API_ROOT}/responses",
        json={"model": "openai:gpt-4o-mini", "input": "hi", "user": _STREAM_USER, "stream": True, **fields},
        headers=headers,
    )


def _stream_responses(client: TestClient, headers: dict[str, str], events: list[ResponseStreamEvent]) -> Any:
    """Stream a Responses request whose provider sends ``events`` and then closes the stream."""
    with patch("gateway.api.routes.responses.aresponses", new=_scripted_aresponses(events)):
        return _post_responses_stream(client, headers)


def test_a_responses_stream_the_provider_ends_before_its_terminal_event_is_an_interrupted_stream_error(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """``response.created`` and one text delta, then the stream closes: the client is told the text is truncated."""
    response = _stream_responses(client, master_key_header, _RESPONSES_OPENING_EVENTS)

    assert response.status_code == 200, response.text
    assert _error_event(response.text)["error"] == {
        "message": "The provider's stream ended before it finished",
        "type": "server_error",
        "code": "provider_stream_interrupted",
    }
    assert _usage_row(db_session_factory).status == "error"


@pytest.mark.parametrize(
    ("event_class", "status"),
    [
        (ResponseCompletedEvent, "completed"),
        (ResponseIncompleteEvent, "incomplete"),
        (ResponseFailedEvent, "failed"),
    ],
    ids=["completed", "incomplete", "failed"],
)
def test_a_responses_stream_that_reaches_its_terminal_event_ends_cleanly(
    client: TestClient,
    master_key_header: dict[str, str],
    event_class: type[ResponseCompletedEvent | ResponseIncompleteEvent | ResponseFailedEvent],
    status: str,
) -> None:
    """Each event the Responses API ends a response with completes the stream, whatever the response's status."""
    usage = ResponseUsage(
        input_tokens=45,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=3,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=48,
    )
    terminal = event_class(
        type=cast(Any, f"response.{status}"), response=_response_snapshot(status, usage), sequence_number=2
    )

    response = _stream_responses(client, master_key_header, [*_RESPONSES_OPENING_EVENTS, terminal])

    assert response.status_code == 200, response.text
    assert "event: error" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


class _LookupPool:
    """An MCP pool whose server owns ``lookup``, so the gateway runs that tool itself."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "_LookupPool":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "lookup", "description": "", "parameters": {}}}]

    def owns_tool(self, name: str) -> bool:
        return name == "lookup"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "found"


def _lookup_call_started(arguments: str) -> list[ResponseStreamEvent]:
    """A round's opening of a ``lookup`` call whose arguments have streamed as far as ``arguments``."""
    lookup = ResponseFunctionToolCall(type="function_call", call_id="call_lookup", name="lookup", arguments="")
    return [
        ResponseOutputItemAddedEvent(type="response.output_item.added", item=lookup, output_index=0, sequence_number=0),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id="fc_lookup",
            output_index=0,
            delta=arguments,
            sequence_number=1,
        ),
    ]


_ANSWER_ROUND: list[ResponseStreamEvent] = [
    _RESPONSES_OPENING_EVENTS[1],
    ResponseCompletedEvent(type="response.completed", response=_response_snapshot("completed"), sequence_number=2),
]


def _stream_responses_with_lookup(
    client: TestClient, headers: dict[str, str], pool: _LookupPool, *rounds: list[ResponseStreamEvent]
) -> Any:
    """Stream a Responses request through the gateway's tool loop, with an MCP server that owns ``lookup``."""
    with (
        patch("gateway.api.routes._pipeline.MCPClientPool", new=lambda *_args, **_kwargs: pool),
        patch("gateway.services.mcp_loop_responses.aresponses", new=_scripted_aresponses(*rounds)),
    ):
        return _post_responses_stream(
            client, headers, mcp_servers=[{"name": "docs", "url": "http://127.0.0.1:9999/mcp"}]
        )


def _event_names(text: str) -> list[str]:
    return [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]


def test_a_responses_tool_loop_round_cut_after_the_gateway_ran_a_tool_is_an_interrupted_stream_error(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Round one asks for ``lookup`` and completes; the round after the gateway ran it is cut."""
    pool = _LookupPool()
    first_round = [
        *_lookup_call_started('{"q": "docs"}'),
        ResponseCompletedEvent(type="response.completed", response=_response_snapshot("completed"), sequence_number=2),
    ]

    response = _stream_responses_with_lookup(
        client, master_key_header, pool, first_round, _RESPONSES_OPENING_EVENTS[1:]
    )

    assert response.status_code == 200, response.text
    assert _error_event(response.text)["error"]["code"] == "provider_stream_interrupted"
    assert _usage_row(db_session_factory).status == "error"
    assert pool.calls == [("lookup", {"q": "docs"})]


def test_a_responses_tool_loop_round_that_runs_out_of_output_mid_call_runs_no_tool_and_ends_incomplete(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The provider stops at its output limit partway through a call's arguments.

    The arguments are not a call anyone asked for, so the gateway runs nothing and
    the client gets the unfinished response as the stream's last event.
    """
    pool = _LookupPool()
    unfinished_round = [
        *_lookup_call_started('{"q": "do'),
        ResponseIncompleteEvent(
            type="response.incomplete", response=_response_snapshot("incomplete"), sequence_number=2
        ),
    ]

    response = _stream_responses_with_lookup(client, master_key_header, pool, unfinished_round, _ANSWER_ROUND)

    assert response.status_code == 200, response.text
    assert "event: error" not in response.text
    assert _event_names(response.text)[-1] == "response.incomplete", response.text
    assert pool.calls == []


def test_a_responses_tool_loop_round_cut_mid_call_runs_no_tool_and_is_an_interrupted_stream_error(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    pool = _LookupPool()

    response = _stream_responses_with_lookup(
        client, master_key_header, pool, _lookup_call_started('{"q": "do'), _ANSWER_ROUND
    )

    assert response.status_code == 200, response.text
    assert _error_event(response.text)["error"]["code"] == "provider_stream_interrupted"
    assert pool.calls == []


class _WebSearchBackend:
    """The gateway's web retrieval backend, answering every search from memory."""

    def __init__(self, **_kwargs: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "_WebSearchBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "web_search", "description": "", "parameters": {}}}]

    def owns_tool(self, name: str) -> bool:
        return name == "web_search"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "[1] Result\nhttps://example.com"

    def take_last_results(self) -> list[dict[str, Any]]:
        return [{"url": "https://example.com", "title": "Example"}]


def test_a_responses_tool_loop_round_that_ends_incomplete_runs_none_of_the_gateway_tools_it_asked_for(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One round asks for the gateway's search and the caller's own tool, then ends in ``response.incomplete``.

    The search does not run, and the client is handed its own call and then the
    unfinished response, last.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    search = ResponseFunctionToolCall(
        type="function_call", call_id="call_search", name="web_search", arguments='{"query": "otari"}'
    )
    caller_tool = ResponseFunctionToolCall(type="function_call", call_id="call_caller", name="lookup", arguments="{}")
    mixed_round: list[ResponseStreamEvent] = [
        ResponseOutputItemAddedEvent(type="response.output_item.added", item=search, output_index=0, sequence_number=0),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added", item=caller_tool, output_index=1, sequence_number=1
        ),
        ResponseIncompleteEvent(
            type="response.incomplete", response=_response_snapshot("incomplete"), sequence_number=2
        ),
    ]
    backend = _WebSearchBackend()

    with (
        patch("gateway.api.routes._pipeline._build_web_retrieval_backend", new=lambda **_kwargs: backend),
        patch("gateway.services.mcp_loop_responses.aresponses", new=_scripted_aresponses(mixed_round)),
    ):
        response = _post_responses_stream(
            client,
            {**master_key_header, "Otari-Web-Search": "otari"},
            tools=[
                {"type": "web_search"},
                {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}},
            ],
        )

    assert response.status_code == 200, response.text
    assert "event: error" not in response.text
    assert _event_names(response.text) == ["response.output_item.added", "response.incomplete"], response.text
    assert backend.calls == []
