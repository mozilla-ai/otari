"""The model's output for a request, read back from the response the client got, and tool-loop rounds."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gateway.models.usage import UsageLog
from gateway.services.traces import RequestTrace, TurnFacts
from gateway.services.traces._output_reading import dialect_of, read_output


def _sse(*events: object) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"


def test_chat_json_output_is_the_message_and_its_tool_calls() -> None:
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "Let me look.",
                        "tool_calls": [{"function": {"name": "search", "arguments": '{"q": "x"}'}}],
                    }
                }
            ]
        }
    ).encode()

    assert read_output("chat", body, is_stream=False) == 'Let me look.\n[tool call search] {"q": "x"}'


def test_chat_stream_output_joins_the_deltas() -> None:
    body = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [], "usage": {"prompt_tokens": 3}},
    )

    assert read_output("chat", body, is_stream=True) == "Hello"


def test_messages_stream_output_keeps_text_and_tool_input() -> None:
    body = _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Running it."}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "Bash"}},
        {"type": "content_block_delta", "index": 1, "delta": {"partial_json": '{"command": "ls"}'}},
    )

    assert read_output("messages", body, is_stream=True) == 'Running it.\n[tool call Bash] {"command": "ls"}'


def test_responses_stream_output_comes_from_the_completed_event() -> None:
    body = _sse(
        {"type": "response.output_text.delta", "delta": "Hi"},
        {
            "type": "response.completed",
            "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi there"}]}]},
        },
    )

    assert read_output("responses", body, is_stream=True) == "Hi there"


@pytest.mark.parametrize("body", [b"", b"not json", b"[1, 2]", b'{"choices": 5}', b"data: {broken"])
def test_a_malformed_body_reads_as_no_output(body: bytes) -> None:
    for stream in (True, False):
        assert read_output("chat", body, is_stream=stream) == ""


def test_the_endpoint_label_names_the_dialect() -> None:
    assert [dialect_of(e) for e in ("/v1/chat/completions", "/v1/messages", "/v1/responses")] == [
        "chat",
        "messages",
        "responses",
    ]


def _trace() -> RequestTrace:
    return RequestTrace(
        request_id="req-1",
        workspace_id=uuid.uuid4(),
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/chat/completions",
        started_at=datetime(2026, 10, 8, 12, tzinfo=UTC),
        max_spans=256,
        turn=TurnFacts(opens_turn=True),
    )


def _row() -> UsageLog:
    return UsageLog(
        id="row-1",
        timestamp=datetime(2026, 10, 8, 12, 0, 5, tzinfo=UTC),
        model="gpt-5",
        provider="openai",
        endpoint="/v1/chat/completions",
        status="success",
        latency_ms=5000,
        prompt_tokens=100,
        completion_tokens=20,
        cost=Decimal("0.01"),
    )


def test_each_round_of_a_tool_loop_is_a_span_under_the_row_that_billed_them() -> None:
    trace = _trace()
    start = datetime(2026, 10, 8, 12, tzinfo=UTC)
    trace.record_model_round(started=start, ended=start + timedelta(seconds=1))
    trace.record_model_round(started=start + timedelta(seconds=2), ended=start + timedelta(seconds=4))

    trace.record_llm_call(_row())

    billed, *rounds = [span for span in trace.spans if span.kind == "llm"]
    assert billed.span_id == "row-1" and billed.input_tokens == 100
    assert [span.parent_span_id for span in rounds] == ["row-1", "row-1"]
    assert all(span.input_tokens is None and span.cost_snapshot is None for span in rounds)


def test_a_loop_of_one_round_is_just_its_billed_span() -> None:
    trace = _trace()
    start = datetime(2026, 10, 8, 12, tzinfo=UTC)
    trace.record_model_round(started=start, ended=start + timedelta(seconds=1))

    trace.record_llm_call(_row())

    assert [span.kind for span in trace.spans] == ["llm"]


def test_full_capture_keeps_this_requests_output() -> None:
    trace = _trace()
    trace.content_level = "full"
    from gateway.services.traces import RequestContent

    trace.request_content = RequestContent(input="What is 2+2?")
    trace.response_output = "4"

    trace.finish(status_code=200, completed=True)

    assert trace.content["req-1"]["output"] == "4"
