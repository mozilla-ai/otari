"""Unit tests for the declarative composite-plan interpreter."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.services.composite_interpreter import (
    MAX_MAP_ITEMS,
    EmitTerminal,
    EmitToolUse,
    ExpressionError,
    Punt,
    SubJudgment,
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


def _user_turn() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "go"}]


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


def test_next_action_sub_judgment_yields_subjudgment() -> None:
    node = {"type": "sub_judgment", "for": "decide"}
    plan = {"version": 1, "nodes": [node]}
    action = next_action(plan, [{"role": "user", "content": "go"}])
    assert isinstance(action, SubJudgment)
    assert action.node == node


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


# ---------------------------------------------------------------------------
# Adversarial: malformed data must degrade to a punt, never crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_plan", ["a string", 123, ["a", "list"], None])
def test_next_action_malformed_plan_punts(bad_plan: Any) -> None:
    action = next_action(bad_plan, _user_turn())
    assert isinstance(action, Punt)
    assert matches_envelope(bad_plan, _user_turn()) is False


def test_next_action_non_dict_nodes_punt() -> None:
    assert isinstance(next_action({"nodes": "not-a-list"}, _user_turn()), Punt)
    # Non-dict node entries are ignored, leaving an empty plan -> exhausted.
    action = next_action({"nodes": ["str", 1, None]}, _user_turn())
    assert isinstance(action, Punt)


def test_verify_action_non_dict_verifier_is_safe() -> None:
    action = EmitToolUse(tool_name="x", tool_input={})
    assert verify_action(action, ["not", "a", "dict"], {}, _user_turn()) is True
    assert verify_action(action, None, {}, _user_turn()) is True


def test_unsupported_mixed_shape_punts_not_reserves() -> None:
    # emit, then a non-emit, then another emit is unsupported: serving it by
    # indexing all nodes by executed-tool count would re-serve B after the model
    # already emitted it during the sub_judgment turn. Must punt instead.
    plan = {
        "nodes": [
            {"type": "emit_tool_use", "tool": "A", "args": {}},
            {"type": "sub_judgment"},
            {"type": "emit_tool_use", "tool": "B", "args": {}},
        ]
    }
    # Model already executed A then B.
    messages = [_assistant_tool_use("A", "t1"), _tool_result("t1", {}), _assistant_tool_use("B", "t2")]
    action = next_action(plan, messages)
    assert isinstance(action, Punt)
    assert action.reason == "unsupported_plan_shape"


def test_trailing_sub_judgment_after_emits_yields_subjudgment() -> None:
    plan = {
        "nodes": [
            {"type": "emit_tool_use", "tool": "A", "args": {}},
            {"type": "emit_tool_use", "tool": "B", "args": {}},
            {"type": "sub_judgment"},
        ]
    }
    # First two turns emit A then B in order.
    assert next_action(plan, _user_turn()).tool_name == "A"  # type: ignore[union-attr]
    after_a = [_assistant_tool_use("A", "t1"), _tool_result("t1", {})]
    assert next_action(plan, after_a).tool_name == "B"  # type: ignore[union-attr]
    after_b = [*after_a, _assistant_tool_use("B", "t2")]
    tail = next_action(plan, after_b)
    assert isinstance(tail, SubJudgment)


def test_regex_input_length_capped() -> None:
    huge = {"regex_extract": {"value": {"const": "a" * 5000}, "pattern": "email"}}
    with pytest.raises(ExpressionError):
        evaluate_expression(huge, [])


# ---------------------------------------------------------------------------
# map: serving a variable-length loop
# ---------------------------------------------------------------------------


def _map_plan() -> dict[str, Any]:
    """list -> reply to each message -> terminal."""
    return {
        "nodes": [
            {"type": "emit_tool_use", "tool": "gmail_list", "args": {}},
            {
                "type": "map",
                "over": {"last_tool_result": {"tool": "gmail_list", "path": "messages"}},
                "body": [{"type": "emit_tool_use", "tool": "gmail_reply", "args": {"id": {"item": "id"}}}],
            },
            {"type": "emit_terminal", "text": "done"},
        ]
    }


def _after_list(ids: list[str], replied: int = 0) -> list[dict[str, Any]]:
    """Messages after gmail_list returned ``ids``, with ``replied`` replies done."""
    messages: list[dict[str, Any]] = [
        _assistant_tool_use("gmail_list", "t0"),
        _tool_result("t0", {"messages": [{"id": i} for i in ids]}),
    ]
    for n in range(replied):
        messages.append(_assistant_tool_use("gmail_reply", f"r{n}"))
        messages.append(_tool_result(f"r{n}", {"ok": True}))
    return messages


def test_map_prefix_served_before_list_exists() -> None:
    # Turn 0: the list-producing tool has not run, so the map is never reached
    # and the fixed prefix serves normally (lazy expansion, no premature punt).
    action = next_action(_map_plan(), _user_turn())
    assert isinstance(action, EmitToolUse)
    assert action.tool_name == "gmail_list"


def test_map_iterates_body_once_per_item() -> None:
    plan = _map_plan()
    first = next_action(plan, _after_list(["m1", "m2"], replied=0))
    assert isinstance(first, EmitToolUse)
    assert first.tool_name == "gmail_reply"
    assert first.tool_input == {"id": "m1"}

    second = next_action(plan, _after_list(["m1", "m2"], replied=1))
    assert isinstance(second, EmitToolUse)
    assert second.tool_input == {"id": "m2"}

    # Loop exhausted -> the trailing terminal serves.
    end = next_action(plan, _after_list(["m1", "m2"], replied=2))
    assert isinstance(end, EmitTerminal)
    assert end.text == "done"


def test_map_empty_list_falls_through_to_tail() -> None:
    end = next_action(_map_plan(), _after_list([], replied=0))
    assert isinstance(end, EmitTerminal)


def test_map_envelope_prefix_matches() -> None:
    assert matches_envelope(_map_plan(), _after_list(["m1", "m2"], replied=1)) is True


def test_map_deviation_after_list_punts_out_of_envelope() -> None:
    messages = [
        _assistant_tool_use("gmail_list", "t0"),
        _tool_result("t0", {"messages": [{"id": "m1"}]}),
        _assistant_tool_use("wrong_tool", "t1"),
    ]
    action = next_action(_map_plan(), messages)
    assert isinstance(action, Punt)
    assert action.reason == "out_of_envelope"


def test_map_over_list_too_long_punts() -> None:
    ids = [str(i) for i in range(MAX_MAP_ITEMS + 1)]
    action = next_action(_map_plan(), _after_list(ids, replied=0))
    assert isinstance(action, Punt)
    assert action.reason == "map_expansion_failed"


def test_item_expression_outside_map_raises() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression({"item": "id"}, [])


def test_verify_action_derives_expected_from_plan_without_expected_tools() -> None:
    # A synthesized plan ships an action_sequence_match verifier with no static
    # expected_tools list; the verifier must derive the expected tool from the
    # plan (map-aware) instead of failing every emit.
    plan = _map_plan()
    messages = _after_list(["m1"], replied=0)
    spec = {"type": "action_sequence_match"}
    good = EmitToolUse(tool_name="gmail_reply", tool_input={"id": "m1"})
    assert verify_action(good, spec, plan, messages) is True
    bad = EmitToolUse(tool_name="wrong_tool", tool_input={})
    assert verify_action(bad, spec, plan, messages) is False
