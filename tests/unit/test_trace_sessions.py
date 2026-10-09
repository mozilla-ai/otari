"""Which session a request joins, whether it opens a turn, and which client tools it answers.

The request shapes here are structure only, shaped like what each harness sends:
roles, block types and ids, never real content.
"""

import uuid
from datetime import UTC, datetime

import pytest

from gateway.services.traces import AnsweredCall, RequestTrace, SessionRef, TurnFacts, harness_of, read_turn
from gateway.services.traces import resolve_session as resolve
from gateway.services.traces._collector import client_tool_span_id

_WORKSPACE = uuid.uuid4()
_CLAUDE_CODE = {"user-agent": "claude-cli/2.1.291 (external, sdk-cli)", "x-claude-code-session-id": "b57732d4"}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_claude_codes_own_session_header_names_the_session() -> None:
    assert resolve(_CLAUDE_CODE, session_label=None, tags=None) == SessionRef(source="harness", key="b57732d4")


@pytest.mark.parametrize(
    ("headers", "label", "tags"),
    [
        ({"Otari-Conversation-Id": "mine"}, None, None),
        ({}, "mine", None),
        ({}, None, {"session_id": "mine"}),
    ],
)
def test_a_session_the_client_names_on_purpose(
    headers: dict[str, str], label: str | None, tags: dict[str, str] | None
) -> None:
    assert resolve(headers, session_label=label, tags=tags) == SessionRef(source="client", key="mine")


def test_the_clients_own_session_wins_over_its_harness() -> None:
    ref = resolve(_CLAUDE_CODE | {"Otari-Conversation-Id": "mine"}, session_label=None, tags=None)

    assert ref == SessionRef(source="client", key="mine")


def test_a_request_that_names_no_session_has_none() -> None:
    assert resolve({"user-agent": "python-httpx/0.28"}, session_label="  ", tags={"purpose": "x"}) is None


def test_the_same_label_from_two_users_is_two_sessions() -> None:
    ref = SessionRef(source="client", key="chat-1")

    assert ref.trace_id(_WORKSPACE, "alice") != ref.trace_id(_WORKSPACE, "bob")
    assert ref.trace_id(_WORKSPACE, "alice") != ref.trace_id(uuid.uuid4(), "alice")
    assert ref.trace_id(_WORKSPACE, "alice") == ref.trace_id(_WORKSPACE, "alice")


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (_CLAUDE_CODE, "claude-code"),
        ({"user-agent": "claude-cli/2.1.291"}, "claude-code"),
        ({"user-agent": "opencode/0.15 ai-sdk"}, "opencode"),
        ({"user-agent": "OpenAI/Python 1.99"}, "openai"),
        ({}, None),
    ],
)
def test_the_harness_comes_from_the_user_agents_product_name(headers: dict[str, str], expected: str | None) -> None:
    assert harness_of(headers) == expected


# ---------------------------------------------------------------------------
# Turns, per dialect
# ---------------------------------------------------------------------------


def test_messages_a_new_prompt_opens_a_turn() -> None:
    assert read_turn("messages", [{"role": "user", "content": [{"type": "text", "text": "x"}]}]).opens_turn


def test_messages_tool_results_answer_the_calls_the_history_named() -> None:
    history = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
                {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "x"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "x", "is_error": True},
            ],
        },
    ]

    assert read_turn("messages", history) == TurnFacts(
        opens_turn=False,
        answered=(
            AnsweredCall(call_id="toolu_1", name="Bash", is_error=False),
            AnsweredCall(call_id="toolu_2", name="Read", is_error=True),
        ),
    )


def test_chat_trailing_tool_messages_answer_the_assistants_calls() -> None:
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "x"},
    ]

    assert read_turn("chat", history) == TurnFacts(
        opens_turn=False, answered=(AnsweredCall(call_id="call_1", name="bash", is_error=False),)
    )
    assert read_turn("chat", [{"role": "user", "content": "x"}]).opens_turn


def test_responses_function_call_outputs_answer_their_calls() -> None:
    items = [
        {"type": "message", "role": "user", "content": "x"},
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "x"},
    ]

    assert read_turn("responses", items).answered == (AnsweredCall(call_id="call_1", name="shell", is_error=False),)
    assert read_turn("responses", "a prompt").opens_turn


@pytest.mark.parametrize("dialect", ["chat", "messages", "responses"])
def test_an_empty_or_unexpected_input_says_nothing(dialect: str) -> None:
    assert read_turn(dialect, None) == TurnFacts(opens_turn=False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


def _trace(*, session: SessionRef | None, turn: TurnFacts | None) -> RequestTrace:
    return RequestTrace(
        request_id="req-2",
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/messages",
        started_at=datetime(2026, 10, 7, 12, tzinfo=UTC),
        max_spans=256,
        session=session,
        harness="claude-code",
        turn=turn,
    )


def test_a_request_in_a_session_writes_into_the_sessions_trace() -> None:
    session = SessionRef(source="harness", key="b57732d4")

    write = _trace(session=session, turn=TurnFacts(opens_turn=True)).finish(status_code=200, completed=True)

    assert write.trace_id == session.trace_id(_WORKSPACE, "alice")
    assert (write.session_source, write.harness) == ("harness", "claude-code")
    assert write.spans[0].opens_turn is True


def test_answered_calls_become_client_tool_spans_ending_when_this_request_arrived() -> None:
    turn = TurnFacts(
        opens_turn=False,
        answered=(
            AnsweredCall(call_id="toolu_1", name="Bash", is_error=False),
            AnsweredCall(call_id="toolu_2", name="rm -rf /", is_error=True),
        ),
    )

    write = _trace(session=None, turn=turn).finish(status_code=200, completed=True)

    tools = [span for span in write.spans if span.kind == "tool"]
    assert [(span.span_id, span.tool_name, span.outcome, span.tool_type) for span in tools] == [
        (client_tool_span_id("toolu_1"), "Bash", "ok", "client"),
        (client_tool_span_id("toolu_2"), None, "error", "client"),
    ]
    assert all(span.start_time is None and span.end_time is not None for span in tools)


def test_a_client_tool_span_id_never_carries_the_call_id() -> None:
    turn = TurnFacts(
        opens_turn=False, answered=(AnsweredCall(call_id="my private\u0000note", name=None, is_error=False),)
    )

    write = _trace(session=None, turn=turn).finish(status_code=200, completed=True)

    (tool,) = [span for span in write.spans if span.kind == "tool"]
    assert tool.span_id == client_tool_span_id("my private\u0000note")
    assert "private" not in tool.span_id and tool.tool_call_id is None


def test_client_tool_spans_count_against_the_span_budget() -> None:
    answered = tuple(AnsweredCall(call_id=f"c{i}", name="Bash", is_error=False) for i in range(20))
    trace = _trace(session=None, turn=TurnFacts(opens_turn=False, answered=answered))
    trace.max_spans = 2

    write = trace.finish(status_code=200, completed=True)

    assert len([span for span in write.spans if span.kind == "tool"]) == 2
    assert trace.dropped == 18
