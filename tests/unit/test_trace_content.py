"""What a request's content is, per dialect, and how much of it each capture level keeps."""

import uuid
from datetime import UTC, datetime

from gateway.services.traces import RequestTrace, TurnFacts, read_content
from gateway.services.traces._collector import client_tool_span_id
from gateway.services.traces._content_reading import MAX_FIELD_CHARS
from gateway.services.traces._turns import AnsweredCall

_HISTORY = [
    {"role": "user", "content": "list the repo"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running ls."},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"}]},
]


def test_messages_content_is_the_new_input_the_prior_output_and_the_tool_io() -> None:
    content = read_content("messages", _HISTORY)

    assert content.input == "README.md"
    assert "Running ls." in content.prior_output
    assert content.tool_calls["toolu_1"].arguments == '{"command": "ls"}'
    assert content.tool_calls["toolu_1"].result == "README.md"


def test_chat_and_responses_read_their_own_shapes() -> None:
    chat = read_content(
        "chat",
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"a":1}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
    )
    responses = read_content(
        "responses",
        [
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ],
    )

    assert (chat.tool_calls["c1"].arguments, chat.tool_calls["c1"].result) == ('{"a":1}', "ok")
    assert (responses.tool_calls["c1"].arguments, responses.tool_calls["c1"].result) == ('{"cmd":"ls"}', "ok")


def test_a_field_is_capped() -> None:
    content = read_content("chat", [{"role": "user", "content": "x" * (MAX_FIELD_CHARS + 50)}])

    assert len(content.input) < MAX_FIELD_CHARS + 20
    assert content.input.endswith("[truncated]")


def _trace(level: str) -> RequestTrace:
    trace = RequestTrace(
        request_id="req-1",
        workspace_id=uuid.uuid4(),
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/messages",
        started_at=datetime(2026, 10, 7, tzinfo=UTC),
        max_spans=256,
        turn=TurnFacts(opens_turn=False, answered=(AnsweredCall(call_id="toolu_1", name="Bash", is_error=False),)),
        content_level=level,
        request_content=read_content("messages", _HISTORY),
    )
    trace.record_tool_call(
        tool_name="web_search",
        tool_type="otari_web_search",
        started=trace.started_at,
        ok=True,
        arguments='{"query": "x"}',
        result="hits",
    )
    trace.finish(status_code=200, completed=True)
    return trace


def test_off_keeps_nothing() -> None:
    assert _trace("off").content == {}


def test_tool_io_keeps_tool_calls_but_not_the_prompt() -> None:
    content = _trace("tool_io").content

    assert content[client_tool_span_id("toolu_1")] == {"arguments": '{"command": "ls"}', "result": "README.md"}
    assert any(value.get("arguments") == '{"query": "x"}' for value in content.values())
    assert "req-1" not in content


def test_full_also_keeps_the_input_and_the_prior_output() -> None:
    content = _trace("full").content

    assert content["req-1"]["input"] == "README.md"
    assert "Running ls." in content["req-1"]["prior_output"]


def test_content_never_appears_in_the_traces_repr() -> None:
    assert "README.md" not in repr(_trace("full"))
