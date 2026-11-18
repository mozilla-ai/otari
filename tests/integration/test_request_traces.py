"""A completion served by the gateway leaves a trace of what happened, and nothing it should not.

Each request is a trace of its own until it names a session: a step span for the
request as the client saw it, and an LLM span per provider attempt carrying the
tokens and cost its usage row recorded. The writer stores traces off the request
path, so these tests wait for the rows to land rather than assuming they have.
"""

import time
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, CONVERSATION_HEADER, REQUEST_ID_HEADER
from gateway.models.traces import Trace, TraceSpan
from gateway.models.usage import UsageLog

from .conftest import MODEL_NAME

_MESSAGES = [{"role": "user", "content": "Summarize the quarterly report for Acme"}]


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-trace",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _message() -> MessageResponse:
    return MessageResponse.model_validate(
        {
            "id": "msg_trace",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": MODEL_NAME,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


def _chunks() -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-trace",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi"), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    ]


def _wait_for_spans(db_session_factory: Callable[[], Session], trace_id: str, count: int) -> list[TraceSpan]:
    deadline = time.monotonic() + 10
    while True:
        with db_session_factory() as session:
            spans = list(session.scalars(select(TraceSpan).where(TraceSpan.trace_id == trace_id)))
            if len(spans) >= count or time.monotonic() > deadline:
                return spans
        time.sleep(0.1)


def _only_trace_id(db_session_factory: Callable[[], Session]) -> str:
    deadline = time.monotonic() + 10
    while True:
        with db_session_factory() as session:
            ids = list(session.scalars(select(Trace.trace_id)))
        if ids or time.monotonic() > deadline:
            [trace_id] = ids
            return trace_id
        time.sleep(0.1)


def _by_kind(spans: list[TraceSpan]) -> dict[str, TraceSpan]:
    return {span.kind: span for span in spans}


def test_a_completion_records_its_step_and_llm_call(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        response = client.post(
            f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": _MESSAGES}, headers=api_key_header
        )
    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]

    spans = _by_kind(_wait_for_spans(db_session_factory, request_id, 2))

    step, llm = spans["step"], spans["llm"]
    assert (step.span_id, step.name, step.outcome, step.parent_span_id) == (
        request_id,
        "/v1/chat/completions",
        "ok",
        None,
    )
    assert (llm.parent_span_id, llm.outcome, llm.request_id) == (request_id, "ok", request_id)
    assert (llm.input_tokens, llm.output_tokens) == (10, 5)
    with db_session_factory() as session:
        trace = session.scalars(select(Trace).where(Trace.trace_id == request_id)).one()
    assert (trace.session_source, trace.step_count, trace.input_tokens) == ("none", 1, 10)


def test_an_anthropic_call_counts_its_cached_prompt_and_costs_what_its_usage_row_does(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Anthropic reports cache reads and writes beside ``input_tokens``; the span counts the whole prompt."""
    model = "anthropic:claude-3-5-sonnet-20241022"
    pricing = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": model,
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.3,
            "cache_write_price_per_million": 3.75,
            "effective_at": "2020-01-01T00:00:00Z",
        },
        headers=master_key_header,
    )
    assert pricing.status_code == 200
    reply = MessageResponse(
        id="msg_trace",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text="hi", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=5,
            output_tokens=2,
            cache_read_input_tokens=300,
            cache_creation_input_tokens=40,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )

    async def fake_amessages(**_: Any) -> MessageResponse:
        return reply

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        response = client.post(
            f"{API_ROOT}/messages",
            json={"model": model, "messages": _MESSAGES, "max_tokens": 16},
            headers=api_key_header,
        )
    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]

    llm = _by_kind(_wait_for_spans(db_session_factory, request_id, 2))["llm"]

    assert (llm.input_tokens, llm.cache_read_tokens, llm.cache_write_tokens) == (345, 300, 40)
    with db_session_factory() as session:
        row = session.scalars(select(UsageLog).where(UsageLog.request_group_id == request_id)).one()
    # 5 fresh at $3, 300 read at $0.30, 40 written at $3.75 and 2 out at $15, per million.
    assert llm.cost_snapshot == row.cost == Decimal("0.000285")


def test_no_prompt_text_reaches_the_trace(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        response = client.post(
            f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": _MESSAGES}, headers=api_key_header
        )
    request_id = response.headers[REQUEST_ID_HEADER]

    spans = _wait_for_spans(db_session_factory, request_id, 2)

    stored = repr([(span.name, span.attributes, span.error_class, span.model) for span in spans])
    assert "quarterly" not in stored
    assert "Acme" not in stored


_CONTEXT_ATTRIBUTES = (
    "otari.context.system_chars",
    "otari.context.tools_chars",
    "otari.context.user_chars",
    "otari.context.assistant_chars",
    "otari.context.tool_result_chars",
    "otari.context.file_chars",
    "otari.context.tool_count",
    "otari.context.message_count",
)
_TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}


def test_a_chat_request_keeps_the_size_of_each_part_and_never_its_text(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    messages = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Summarize the quarterly report for Acme"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "revenue is up"},
    ]
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": messages, "tools": [_TOOL]},
            headers=api_key_header,
        )
    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]

    llm = _by_kind(_wait_for_spans(db_session_factory, request_id, 2))["llm"]
    attributes = llm.attributes or {}

    assert set(_CONTEXT_ATTRIBUTES) <= set(attributes)
    assert attributes["otari.context.system_chars"] == len("Answer in French.")
    assert attributes["otari.context.user_chars"] == len("Summarize the quarterly report for Acme")
    assert attributes["otari.context.tool_result_chars"] == len("revenue is up")
    assert (attributes["otari.context.tool_count"], attributes["otari.context.message_count"]) == (1, 4)
    assert "quarterly" not in repr(attributes)
    detail = client.get(f"{API_ROOT}/traces/{request_id}", headers=master_key_header).json()
    [read] = [span for span in detail["spans"] if span["kind"] == "llm"]
    assert read["attributes"]["otari.context.user_chars"] == len("Summarize the quarterly report for Acme")
    assert "context_window" in read


def test_a_messages_request_counts_its_system_parameter(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with patch("gateway.api.routes.messages.amessages", return_value=_message()):
        response = client.post(
            f"{API_ROOT}/messages",
            json={"model": MODEL_NAME, "max_tokens": 64, "messages": _MESSAGES, "system": "You are terse."},
            headers=api_key_header,
        )
    assert response.status_code == 200

    llm = _by_kind(_wait_for_spans(db_session_factory, response.headers[REQUEST_ID_HEADER], 2))["llm"]
    attributes = llm.attributes or {}

    assert attributes["otari.context.system_chars"] == len("You are terse.")
    assert attributes["otari.context.message_count"] == 1


_FAILING_ENDPOINTS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "chat": ("chat/completions", "gateway.api.routes.chat.acompletion", {"model": MODEL_NAME, "messages": _MESSAGES}),
    "messages": (
        "messages",
        "gateway.api.routes.messages.amessages",
        {"model": "anthropic:claude-3-5-sonnet", "messages": _MESSAGES, "max_tokens": 64},
    ),
    "responses": (
        "responses",
        "gateway.api.routes.responses.aresponses",
        {"model": "openai:gpt-4o-mini", "input": _MESSAGES},
    ),
}


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "streamed"])
@pytest.mark.parametrize("endpoint", list(_FAILING_ENDPOINTS))
def test_a_failed_provider_call_is_an_error_step_and_an_error_llm_call(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    endpoint: str,
    stream: bool,
) -> None:
    path, provider_call, body = _FAILING_ENDPOINTS[endpoint]
    with patch(provider_call, side_effect=RuntimeError("provider down")):
        response = client.post(f"{API_ROOT}/{path}", json={**body, "stream": stream}, headers=api_key_header)
    assert response.status_code >= 500
    request_id = response.headers[REQUEST_ID_HEADER]

    spans = _by_kind(_wait_for_spans(db_session_factory, request_id, 2))

    step = spans["step"]
    assert (step.span_id, step.outcome, step.error_class) == (request_id, "error", f"status_{response.status_code}")
    assert spans["llm"].outcome == "error"


def test_a_streamed_completion_is_traced_once_its_body_is_sent(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    async def open_stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in _chunks():
                yield chunk

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": _MESSAGES, "stream": True},
            headers=api_key_header,
        )
        body = response.text
    assert "[DONE]" in body
    request_id = response.headers[REQUEST_ID_HEADER]

    spans = _by_kind(_wait_for_spans(db_session_factory, request_id, 2))

    assert spans["step"].outcome == "ok"
    assert spans["llm"].output_tokens == 5


def test_requests_of_one_harness_session_share_one_trace(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = api_key_header | {"x-claude-code-session-id": "b57732d4", "user-agent": "claude-cli/2.1.291"}
    first = [{"role": "user", "content": "x"}]
    second = [
        *first,
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "x"},
    ]
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        for messages in (first, second):
            response = client.post(
                f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": messages}, headers=headers
            )
            assert response.status_code == 200

    trace_id = _only_trace_id(db_session_factory)
    spans = _wait_for_spans(db_session_factory, trace_id, 5)

    steps = sorted((span for span in spans if span.kind == "step"), key=lambda span: span.start_time or 0)
    assert [step.opens_turn for step in steps] == [True, False]
    [tool] = [span for span in spans if span.kind == "tool"]
    assert (tool.tool_name, tool.tool_type, tool.outcome) == ("Bash", "client", "ok")
    with db_session_factory() as session:
        trace = session.scalars(select(Trace).where(Trace.trace_id == trace_id)).one()
    assert (trace.session_source, trace.harness, trace.step_count) == ("harness", "claude-code", 2)


def test_requests_of_one_harness_session_on_messages_share_one_trace(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """Claude Code ends each request with a mid-conversation system message."""
    headers = api_key_header | {"x-claude-code-session-id": "b57732d4", "user-agent": "claude-cli/2.1.291"}
    system = {"role": "system", "content": [{"type": "text", "text": "x"}]}
    first = [{"role": "user", "content": [{"type": "text", "text": "x"}]}, system]
    second = [
        *first,
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x"}]},
        system,
    ]
    with patch("gateway.api.routes.messages.amessages", return_value=_message()):
        for messages in (first, second):
            response = client.post(
                f"{API_ROOT}/messages",
                json={"model": MODEL_NAME, "max_tokens": 64, "messages": messages},
                headers=headers,
            )
            assert response.status_code == 200

    trace_id = _only_trace_id(db_session_factory)
    spans = _wait_for_spans(db_session_factory, trace_id, 5)

    steps = sorted((span for span in spans if span.kind == "step"), key=lambda span: span.start_time or 0)
    assert [step.opens_turn for step in steps] == [True, False]
    [tool] = [span for span in spans if span.kind == "tool"]
    assert (tool.tool_name, tool.tool_type, tool.outcome) == ("Bash", "client", "ok")
    with db_session_factory() as session:
        trace = session.scalars(select(Trace).where(Trace.trace_id == trace_id)).one()
    assert (trace.session_source, trace.harness, trace.step_count) == ("harness", "claude-code", 2)


def test_a_claude_code_subagent_continues_its_parents_turn(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A subagent shares its parent's session id; only its agent header tells its task from a new prompt."""
    parent = api_key_header | {"x-claude-code-session-id": "b57732d4", "user-agent": "claude-cli/2.1.291"}
    subagent = parent | {"x-claude-code-agent-id": "a1b2c3"}
    system = {"role": "system", "content": [{"type": "text", "text": "x"}]}
    prompt = [{"role": "user", "content": [{"type": "text", "text": "x"}] * 3}, system]
    tool_result = [
        *prompt,
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x"}]},
        system,
    ]
    with patch("gateway.api.routes.messages.amessages", return_value=_message()):
        for headers, messages in ((parent, prompt), (subagent, prompt), (subagent, tool_result)):
            response = client.post(
                f"{API_ROOT}/messages",
                json={"model": MODEL_NAME, "max_tokens": 64, "messages": messages},
                headers=headers,
            )
            assert response.status_code == 200

    trace_id = _only_trace_id(db_session_factory)
    spans = _wait_for_spans(db_session_factory, trace_id, 7)

    steps = sorted((span for span in spans if span.kind == "step"), key=lambda span: span.start_time or 0)
    assert [step.opens_turn for step in steps] == [True, False, False]
    [tool] = [span for span in spans if span.kind == "tool"]
    assert (tool.tool_name, tool.tool_type, tool.outcome) == ("Bash", "client", "ok")


def test_usage_breaks_spend_down_by_the_session_a_request_was_traced_into(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    session_headers = api_key_header | {CONVERSATION_HEADER: "run-1"}
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        for headers in (session_headers, session_headers, api_key_header):
            response = client.post(
                f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": _MESSAGES}, headers=headers
            )
            assert response.status_code == 200
    unnamed_request_id = response.headers[REQUEST_ID_HEADER]

    _wait_for_spans(db_session_factory, unnamed_request_id, 2)
    deadline = time.monotonic() + 10
    while True:
        with db_session_factory() as session:
            trace_ids = list(session.scalars(select(Trace.trace_id).where(Trace.session_source == "client")))
        if trace_ids or time.monotonic() > deadline:
            [trace_id] = trace_ids
            break
        time.sleep(0.1)

    with db_session_factory() as session:
        labels = sorted(session.scalars(select(UsageLog.source_label)), key=str)
    assert labels == sorted([None, trace_id, trace_id], key=str)

    summary = client.get(
        f"{API_ROOT}/usage/summary", params={"dimensions": "source_label"}, headers=master_key_header
    ).json()
    assert {row["key"]: row["requests"] for row in summary["by_source_label"]} == {trace_id: 2, None: 1}
