"""What a routed request keeps of its own content, where it, its deployment and its workspace keep content.

The bodies here are shaped like what each dialect sends, with placeholder text.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from gateway.services.traces import RequestTrace, SessionRef, TurnContent, read_turn, read_turn_content
from gateway.services.traces._turn_content import ToolContent
from gateway.types.traces import SPAN_CONTENT_MAX_LENGTH

_WORKSPACE = uuid.uuid4()


# ---------------------------------------------------------------------------
# Reading the body
# ---------------------------------------------------------------------------


def test_chat_keeps_the_new_prompt() -> None:
    messages = [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "image_url", "image_url": {}}]},
    ]

    content = read_turn_content("chat", messages)

    assert content == TurnContent(prompt="first")


def test_chat_keeps_each_answered_calls_arguments_and_result() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": '{"p":"."}'}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
    ]

    content = read_turn_content("chat", messages)

    assert content is not None
    assert content.prompt is None
    assert content.calls == {("call_1", 2): ToolContent(arguments='{"p":"."}', result="a.txt")}


def test_messages_keeps_the_prompt_under_a_trailing_system_message() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "fix it"}]},
        {"role": "system", "content": "reminder"},
    ]

    content = read_turn_content("messages", messages)

    assert content is not None
    assert content.prompt == "fix it"


def test_messages_keeps_tool_use_input_as_json_and_the_result_text() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"cmd": "ls"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt"}]}
            ],
        },
    ]

    content = read_turn_content("messages", messages)
    answered = read_turn("messages", messages).answered

    assert content is not None
    assert [(call.call_id, call.position) for call in answered] == list(content.calls)
    assert content.calls[("toolu_1", 2)] == ToolContent(arguments='{"cmd": "ls"}', result="a.txt")


def test_responses_keeps_a_string_input_as_the_prompt() -> None:
    assert read_turn_content("responses", "hello") == TurnContent(prompt="hello")


def test_responses_keeps_function_call_arguments_and_output() -> None:
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        {"type": "function_call", "call_id": "fc_1", "name": "ls", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fc_1", "output": "a.txt"},
    ]

    content = read_turn_content("responses", items)

    assert content is not None
    assert content.calls == {("fc_1", 2): ToolContent(arguments="{}", result="a.txt")}


def test_a_malformed_body_reads_as_nothing() -> None:
    assert read_turn_content("chat", "not a list") == TurnContent()
    assert read_turn_content("messages", [None, 3]) == TurnContent()


# ---------------------------------------------------------------------------
# What the trace keeps
# ---------------------------------------------------------------------------


def _answering_trace(*, generation: int | None, keep_tool_output: bool = False) -> RequestTrace:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"c": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "out"}]},
    ]
    return RequestTrace(
        request_id="req-1",
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/messages",
        started_at=datetime(2026, 10, 29, 12, tzinfo=UTC),
        max_spans=256,
        session=SessionRef(source="harness", key="b57732d4"),
        turn=read_turn("messages", messages),
        content=read_turn_content("messages", messages) if generation is not None else None,
        content_generation=generation,
        keep_tool_output=keep_tool_output,
    )


def _prompt_trace(prompt: str, *, generation: int | None, subagent: bool = False) -> RequestTrace:
    messages = [{"role": "user", "content": prompt}]
    return RequestTrace(
        request_id="req-2",
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/messages",
        started_at=datetime(2026, 10, 29, 12, tzinfo=UTC),
        max_spans=256,
        turn=read_turn("messages", messages),
        content=read_turn_content("messages", messages),
        content_generation=generation,
        subagent=subagent,
    )


def test_a_request_that_keeps_content_puts_the_prompt_on_its_step() -> None:
    write = _prompt_trace("fix the bug", generation=3).finish(status_code=200, completed=True)

    assert write.content_generation == 3
    assert write.spans[0].content == {"prompt": "fix the bug"}


def test_a_request_without_a_generation_keeps_no_content() -> None:
    write = _prompt_trace("fix the bug", generation=None).finish(status_code=200, completed=True)

    assert write.content_generation is None
    assert all(not span.content for span in write.spans)


def test_a_subagents_task_is_no_prompt() -> None:
    write = _prompt_trace("the task", generation=1, subagent=True).finish(status_code=200, completed=True)

    assert write.spans[0].content == {}


def test_a_long_prompt_is_cut_to_what_the_store_keeps() -> None:
    write = _prompt_trace("x" * (SPAN_CONTENT_MAX_LENGTH + 10), generation=1).finish(status_code=200, completed=True)

    assert len(write.spans[0].content["prompt"]) == SPAN_CONTENT_MAX_LENGTH


def test_an_answered_call_keeps_its_arguments_but_not_its_result_by_default() -> None:
    write = _answering_trace(generation=1).finish(status_code=200, completed=True)

    tool = next(span for span in write.spans if span.tool_type == "client")
    assert tool.content == {"tool.input": '{"c": 1}'}


def test_an_answered_call_keeps_its_result_where_the_deployment_keeps_tool_output() -> None:
    write = _answering_trace(generation=1, keep_tool_output=True).finish(status_code=200, completed=True)

    tool = next(span for span in write.spans if span.tool_type == "client")
    assert tool.content == {"tool.input": '{"c": 1}', "tool.output": "out"}


def test_a_tool_the_gateway_ran_keeps_its_arguments_and_result() -> None:
    trace = _answering_trace(generation=1, keep_tool_output=True)

    trace.record_tool_call(
        tool_name="web_search", tool_type="otari_web_search", started=trace.started_at, ok=True,
        arguments={"query": "otari"}, result="hits",
    )  # fmt: skip

    assert trace.spans[-1].content == {"tool.input": '{"query": "otari"}', "tool.output": "hits"}


def test_a_tool_the_gateway_ran_keeps_nothing_without_a_generation() -> None:
    trace = _answering_trace(generation=None, keep_tool_output=True)

    trace.record_tool_call(
        tool_name="web_search", tool_type="otari_web_search", started=trace.started_at, ok=True,
        arguments={"query": "otari"}, result="hits",
    )  # fmt: skip

    assert trace.spans[-1].content == {}


# ---------------------------------------------------------------------------
# One session, routed and exported
# ---------------------------------------------------------------------------


def test_an_exported_session_id_joins_the_routed_session_with_the_same_id() -> None:
    routed = SessionRef(source="harness", key="b57732d4")
    exported = SessionRef(source="otlp", key="b57732d4")

    assert routed.trace_id(_WORKSPACE, "alice") == exported.trace_id(_WORKSPACE, "alice")


def test_a_clients_chosen_key_stays_apart_from_a_harness_id() -> None:
    chosen = SessionRef(source="client", key="b57732d4")

    assert chosen.trace_id(_WORKSPACE, "alice") != SessionRef(source="otlp", key="b57732d4").trace_id(
        _WORKSPACE, "alice"
    )
