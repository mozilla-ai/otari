"""Unit tests for the declarative composite-plan interpreter."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.services.composite_interpreter import (
    EmitTerminal,
    EmitToolUse,
    ExpressionError,
    Punt,
    evaluate_expression,
    executed_tool_names,
    last_tool_result,
    matches_envelope,
    next_action,
    verify_action,
)

_MODAL = ["slack_read_channel_messages", "resolve_time", "google_sheets_append_row"]


def _plan(args_for_first: dict[str, Any] | None = None) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for i, tool in enumerate(_MODAL):
        node: dict[str, Any] = {"type": "emit_tool_use", "tool": tool, "args": {}}
        if i == 0 and args_for_first is not None:
            node["args"] = args_for_first
        nodes.append(node)
    return {"version": 1, "matcher": {"modal_tool_sequence": list(_MODAL)}, "nodes": nodes}


def _assistant_tool_use(name: str, tool_id: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input or {}}
    return {"role": "assistant", "content": [block]}


def _tool_result(tool_use_id: str, content: Any) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}]}


# ---------------------------------------------------------------------------
# Visible-state helpers
# ---------------------------------------------------------------------------


def test_executed_tool_names_in_order() -> None:
    messages = [
        {"role": "user", "content": "go"},
        _assistant_tool_use("slack_read_channel_messages", "t1"),
        _tool_result("t1", {"messages": []}),
        _assistant_tool_use("resolve_time", "t2"),
    ]
    assert executed_tool_names(messages) == ["slack_read_channel_messages", "resolve_time"]


def test_last_tool_result_matches_by_tool_name() -> None:
    messages = [
        _assistant_tool_use("slack_read_channel_messages", "t1"),
        _tool_result("t1", {"channel": "C123"}),
    ]
    assert last_tool_result(messages, "slack_read_channel_messages") == {"channel": "C123"}
    assert last_tool_result(messages, "nonexistent") is None


# ---------------------------------------------------------------------------
# Expression language
# ---------------------------------------------------------------------------


def test_expr_const() -> None:
    assert evaluate_expression({"const": "Sheet1"}, []) == "Sheet1"


def test_expr_last_tool_result_path() -> None:
    messages = [
        _assistant_tool_use("slack_read_channel_messages", "t1"),
        _tool_result("t1", {"latest": {"ts": "1700.5"}}),
    ]
    expr = {"last_tool_result": {"tool": "slack_read_channel_messages", "path": "latest.ts"}}
    assert evaluate_expression(expr, messages) == "1700.5"


def test_expr_trigger_path() -> None:
    messages = [{"role": "user", "content": {"channel": "C123", "team": "T1"}}]
    assert evaluate_expression({"trigger": "channel"}, messages) == "C123"


def test_expr_lower_and_join() -> None:
    expr = {"join": {"sep": "/", "values": [{"const": "A"}, {"lower": {"const": "B"}}]}}
    assert evaluate_expression(expr, []) == "A/b"


def test_expr_regex_extract_whitelisted() -> None:
    expr = {"regex_extract": {"value": {"const": "date is 2026-07-09 ok"}, "pattern": "iso_date"}}
    assert evaluate_expression(expr, []) == "2026-07-09"


def test_expr_unknown_op_raises() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression({"eval": "os.system('x')"}, [])


def test_expr_non_whitelisted_pattern_raises() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression({"regex_extract": {"value": {"const": "x"}, "pattern": "(.*)+"}}, [])


def test_expr_depth_bounded() -> None:
    expr: dict[str, Any] = {"const": "x"}
    for _ in range(12):
        expr = {"lower": expr}
    with pytest.raises(ExpressionError):
        evaluate_expression(expr, [])


def test_expr_missing_path_raises() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression({"trigger": "nope"}, [{"role": "user", "content": {"a": 1}}])


# ---------------------------------------------------------------------------
# Envelope + next_action
# ---------------------------------------------------------------------------


def test_envelope_prefix_matches() -> None:
    plan = _plan()
    messages = [_assistant_tool_use("slack_read_channel_messages", "t1")]
    assert matches_envelope(plan, messages) is True


def test_envelope_deviation_fails() -> None:
    plan = _plan()
    messages = [_assistant_tool_use("gmail_send_email", "t1")]
    assert matches_envelope(plan, messages) is False


def test_next_action_emits_first_then_second() -> None:
    plan = _plan()
    first = next_action(plan, [{"role": "user", "content": "go"}])
    assert isinstance(first, EmitToolUse)
    assert first.tool_name == "slack_read_channel_messages"

    after_one = next_action(plan, [_assistant_tool_use("slack_read_channel_messages", "t1"), _tool_result("t1", {})])
    assert isinstance(after_one, EmitToolUse)
    assert after_one.tool_name == "resolve_time"


def test_next_action_punts_when_exhausted() -> None:
    plan = _plan()
    messages = [_assistant_tool_use(t, f"t{i}") for i, t in enumerate(_MODAL)]
    action = next_action(plan, messages)
    assert isinstance(action, Punt)
    assert action.reason == "plan_exhausted"


def test_next_action_punts_out_of_envelope() -> None:
    plan = _plan()
    action = next_action(plan, [_assistant_tool_use("unexpected_tool", "t1")])
    assert isinstance(action, Punt)
    assert action.reason == "out_of_envelope"


def test_next_action_punts_on_bad_arg_expression() -> None:
    plan = _plan(args_for_first={"row": {"last_tool_result": {"tool": "nope", "path": "x.y"}}})
    action = next_action(plan, [{"role": "user", "content": "go"}])
    assert isinstance(action, Punt)
    assert action.reason == "arg_expression_failed"


def test_next_action_builds_args_from_expressions() -> None:
    plan = _plan(args_for_first={"channel": {"trigger": "channel"}})
    messages = [{"role": "user", "content": {"channel": "C123"}}]
    action = next_action(plan, messages)
    assert isinstance(action, EmitToolUse)
    assert action.tool_input == {"channel": "C123"}


def test_next_action_sub_judgment_punts_until_t1() -> None:
    plan = {"version": 1, "nodes": [{"type": "sub_judgment"}]}
    action = next_action(plan, [{"role": "user", "content": "go"}])
    assert isinstance(action, Punt)
    assert action.reason == "t1_not_enabled"


def test_next_action_emit_terminal() -> None:
    plan = {"version": 1, "nodes": [{"type": "emit_terminal", "text": "done"}]}
    action = next_action(plan, [{"role": "user", "content": "go"}])
    assert isinstance(action, EmitTerminal)
    assert action.text == "done"


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verify_action_sequence_match_passes() -> None:
    plan = _plan()
    action = EmitToolUse(tool_name="slack_read_channel_messages", tool_input={})
    spec = {"type": "action_sequence_match", "expected_tools": _MODAL}
    assert verify_action(action, spec, plan, [{"role": "user", "content": "go"}]) is True


def test_verify_action_sequence_match_fails_on_wrong_tool() -> None:
    plan = _plan()
    action = EmitToolUse(tool_name="wrong_tool", tool_input={})
    spec = {"type": "action_sequence_match", "expected_tools": _MODAL}
    assert verify_action(action, spec, plan, [{"role": "user", "content": "go"}]) is False
