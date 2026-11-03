"""What a request's trace keeps of its composition: character weights per kind of input, never text."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from gateway.services.traces import ContextWeights, LlmCall, RequestTrace, context_of, read_context
from gateway.services.traces._context import FILE_CHARS
from gateway.types.traces import SPAN_ATTRIBUTES, InvalidSpanError, SpanRecord

_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)
_SECRET = "the Q3 launch plan is confidential"


def _weights(dialect: Any, messages: Any, **kwargs: Any) -> ContextWeights:
    weights = read_context(dialect, messages, **kwargs)
    assert weights is not None
    return weights


def test_a_chat_request_is_split_by_role_with_tool_results_and_definitions_counted_apart() -> None:
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    weights = _weights(
        "chat",
        [
            {"role": "system", "content": "be brief"},
            {"role": "developer", "content": "xx"},
            {"role": "user", "content": "hello there"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"c":"Oslo"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "12 degrees"},
            {"role": "assistant", "content": "It is 12."},
        ],
        tools=tools,
    )

    assert weights.system_chars == len("be brief") + 2
    assert weights.user_chars == len("hello there")
    assert weights.assistant_chars == len("get_weather") + len('{"c":"Oslo"}') + len("It is 12.")
    assert weights.tool_result_chars == len("12 degrees")
    assert weights.file_chars == 0
    # Names and text, not JSON punctuation: type, function, function, name, get_weather, parameters, type, object.
    assert weights.tools_chars == len("typefunctionfunctionnameget_weatherparameterstypeobject")
    assert (weights.tool_count, weights.message_count) == (1, 6)


def test_a_chat_image_counts_a_flat_allowance_whatever_its_payload() -> None:
    huge = "A" * 1_000_000
    weights = _weights(
        "chat",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge}"}},
                    {"type": "file", "file": {"file_data": huge}},
                ],
            }
        ],
    )

    assert weights.user_chars == len("what is this")
    assert weights.file_chars == 2 * FILE_CHARS


def test_a_messages_request_counts_the_system_parameter_blocks_and_tool_results() -> None:
    weights = _weights(
        "messages",
        [
            {"role": "user", "content": [{"type": "text", "text": "list files"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "src"}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "a.py\nb.py"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "ok"}]},
                    {"type": "text", "text": "thanks"},
                    {"type": "document", "source": {"type": "base64", "data": "Zm9v"}},
                ],
            },
            {"role": "system", "content": "mid"},
        ],
        tools=[{"name": "ls", "input_schema": {}}, {"name": "cat", "input_schema": {}}],
        system=[{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
    )

    assert weights.system_chars == len("You are helpful.") + len("mid")
    assert weights.user_chars == len("list files") + len("thanks")
    assert weights.assistant_chars == len("ls") + len("src")
    assert weights.tool_result_chars == len("a.py\nb.py") + len("ok")
    assert weights.file_chars == FILE_CHARS
    assert (weights.tool_count, weights.message_count) == (2, 4)


def test_a_messages_request_may_carry_its_system_prompt_as_a_string() -> None:
    weights = _weights("messages", [{"role": "user", "content": "hi"}], system="Be terse.")

    assert (weights.system_chars, weights.user_chars) == (len("Be terse."), 2)


def test_a_responses_request_counts_instructions_items_and_function_outputs() -> None:
    weights = _weights(
        "responses",
        [
            {"type": "message", "role": "developer", "content": "dev"},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}, {"type": "input_image", "image_url": "x"}],
            },
            {"type": "function_call", "call_id": "c1", "name": "sh", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "file.txt"},
            {"type": "reasoning", "id": "r1", "summary": []},
            {"type": "item_reference", "id": "ref"},
        ],
        tools=[{"type": "function", "name": "sh"}],
        system="You are an agent.",
    )

    assert weights.system_chars == len("You are an agent.") + 3
    assert weights.user_chars == 2
    assert weights.assistant_chars == len("sh") + len('{"cmd":"ls"}')
    assert weights.tool_result_chars == len("file.txt")
    assert weights.file_chars == FILE_CHARS
    assert (weights.tool_count, weights.message_count) == (1, 6)


def test_a_responses_input_that_is_one_string_is_one_user_message() -> None:
    weights = _weights("responses", "just a question")

    assert (weights.user_chars, weights.message_count) == (len("just a question"), 1)


@pytest.mark.parametrize("dialect", ["chat", "messages", "responses"])
def test_an_empty_request_weighs_nothing(dialect: Any) -> None:
    assert _weights(dialect, [], tools=[]) == ContextWeights()
    assert _weights(dialect, None) == ContextWeights()


@pytest.mark.parametrize("dialect", ["chat", "messages", "responses"])
def test_a_malformed_body_reads_as_nothing_rather_than_raising(dialect: Any) -> None:
    weights = read_context(dialect, [42, None, "x", {"role": 7, "content": {"a": [1, 2.5, True]}}], tools="nope")

    assert weights is not None
    assert weights.tools_chars == 0


def test_a_deeply_nested_body_is_cut_off_instead_of_recursing_forever() -> None:
    nested: Any = "deep text"
    for _ in range(5_000):
        nested = [nested]

    assert _weights("chat", [{"role": "user", "content": nested}]).user_chars == 0


def test_a_reader_that_fails_gives_no_weights() -> None:
    def broken() -> ContextWeights | None:
        raise RuntimeError("boom")

    assert context_of(broken) is None
    assert context_of(None) is None
    assert context_of(lambda: ContextWeights(user_chars=3)) == ContextWeights(user_chars=3)


def test_the_attributes_hold_only_integers_and_pass_span_validation() -> None:
    weights = _weights(
        "chat", [{"role": "user", "content": _SECRET}], tools=[{"type": "function", "function": {"name": "f"}}]
    )
    attributes = weights.attributes()

    assert set(attributes) <= set(SPAN_ATTRIBUTES)
    assert len(attributes) == 8
    assert all(type(value) is int for value in attributes.values())
    assert _SECRET not in json.dumps(attributes)
    SpanRecord(span_id="s", kind="llm", origin="gateway", name="chat", outcome="ok", attributes=attributes)


def test_a_context_attribute_must_be_an_integer() -> None:
    with pytest.raises(InvalidSpanError):
        SpanRecord(
            span_id="s",
            kind="llm",
            origin="gateway",
            name="chat",
            outcome="ok",
            attributes={"otari.context.user_chars": "12"},
        )


def _call(attempt: str, status: str = "success") -> LlmCall:
    return LlmCall(
        attempt_id=attempt,
        ended=_T0 + timedelta(seconds=2),
        latency_ms=1000,
        status=status,
        status_code=None if status == "success" else 503,
        model="gpt-5",
        provider="openai",
        input_tokens=100,
        output_tokens=5,
        cost=None,
    )


def _trace(context: ContextWeights | None) -> RequestTrace:
    return RequestTrace(
        request_id="req-1",
        workspace_id=uuid.uuid4(),
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/chat/completions",
        started_at=_T0,
        max_spans=16,
        context=context,
    )


def test_every_attempt_of_a_request_carries_its_weights_and_no_text() -> None:
    weights = _weights("chat", [{"role": "user", "content": _SECRET}])
    trace = _trace(weights)

    trace.record_llm_call(_call("attempt-1", status="error"))
    trace.record_llm_call(_call("attempt-2"))

    assert len(trace.spans) == 2
    for span in trace.spans:
        assert span.attributes["otari.context.user_chars"] == len(_SECRET)
        assert span.attributes["otari.context.message_count"] == 1
    assert _SECRET not in repr([span.attributes for span in trace.spans])


def test_a_span_of_a_request_with_no_weights_carries_none() -> None:
    trace = _trace(None)

    trace.record_llm_call(_call("attempt-1"))

    assert not any(key.startswith("otari.context.") for key in trace.spans[0].attributes)


def test_opaque_provider_blobs_weigh_nothing_but_readable_thinking_does() -> None:
    blob = "Zm9v" * 5_000
    weights = _weights(
        "messages",
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me see", "signature": blob},
                    {"type": "redacted_thinking", "data": blob},
                    {"type": "text", "text": "done"},
                    {"type": "tool_use", "id": "t1", "name": "sign", "input": {"signature": "abc"}},
                ],
            }
        ],
    )

    assert weights.assistant_chars == len("let me see") + len("done") + len("sign") + len("abc")


def test_an_encrypted_reasoning_item_weighs_only_its_summary() -> None:
    blob = "gAAAA" * 5_000
    weights = _weights(
        "responses",
        [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": blob,
                "summary": [{"type": "summary_text", "text": "thought"}],
            }
        ],
    )

    assert weights.assistant_chars == len("thought")


def test_a_type_or_role_that_is_not_a_string_weighs_what_it_holds_instead_of_dropping_the_request() -> None:
    schema = {"type": ["string", "null"], "description": "a name"}
    messages = _weights(
        "messages",
        [
            {"role": ["user"], "content": "hello"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": schema}]},
        ],
    )
    chat = _weights("chat", [{"role": {"x": 1}, "content": [{"type": ["text"], "text": "hi"}]}])
    responses = _weights("responses", [{"type": ["message"], "role": "user", "content": "hi"}, {"type": {}}])

    assert messages.user_chars == len("hello")
    assert messages.assistant_chars == len("f") + len("a name")
    assert chat.user_chars == len("hi")
    assert responses.message_count == 2


def test_a_tool_result_nested_deep_still_weighs_its_text() -> None:
    payload: object = "deep text"
    for _ in range(12):
        payload = {"data": [payload]}
    weights = _weights("chat", [{"role": "tool", "tool_call_id": "c1", "content": [payload]}])

    assert weights.tool_result_chars == len("deep text")
