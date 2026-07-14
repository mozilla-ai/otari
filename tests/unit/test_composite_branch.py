"""Unit tests for the branch/filter/predicate additions to the interpreter."""

from __future__ import annotations

import json
from typing import Any

from gateway.services.composite_interpreter import (
    EmitToolUse,
    Punt,
    evaluate_expression,
    evaluate_predicate,
    last_tool_result,
    next_action,
)


def _msgs_after_list(emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "gmail_list_emails", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": {"emails": emails}}]},
    ]


_PLAN = {
    "nodes": [
        {"type": "emit_tool_use", "tool": "gmail_list_emails", "args": {}},
        {
            "type": "branch",
            "predicate": {"non_empty": {"last_tool_result": {"tool": "gmail_list_emails", "path": "emails"}}},
            "then": [
                {
                    "type": "map",
                    "over": {"last_tool_result": {"tool": "gmail_list_emails", "path": "emails"}},
                    "body": [{"type": "emit_tool_use", "tool": "gmail_reply_to_email", "args": {"id": {"item": "id"}}}],
                }
            ],
        },
    ]
}


# -- predicates -------------------------------------------------------------


def test_predicate_non_empty() -> None:
    assert evaluate_predicate({"non_empty": {"const": [1]}}, []) is True
    assert evaluate_predicate({"non_empty": {"const": []}}, []) is False
    assert evaluate_predicate({"non_empty": {"const": ""}}, []) is False


def test_predicate_exists_missing_is_false() -> None:
    assert evaluate_predicate({"exists": {"const": 1}}, []) is True
    # navigating a missing trigger path raises -> exists resolves to False
    assert evaluate_predicate({"exists": {"trigger": "nope"}}, [{"role": "user", "content": {}}]) is False


def test_predicate_equals_not_and_or() -> None:
    assert evaluate_predicate({"equals": {"left": {"const": "a"}, "right": {"const": "a"}}}, []) is True
    assert evaluate_predicate({"not": {"non_empty": {"const": []}}}, []) is True
    assert evaluate_predicate({"and": [{"non_empty": {"const": [1]}}, {"exists": {"const": 1}}]}, []) is True
    assert evaluate_predicate({"or": [{"non_empty": {"const": []}}, {"exists": {"const": 1}}]}, []) is True


# -- filter -----------------------------------------------------------------


def test_filter_keeps_matching_items() -> None:
    expr = {
        "filter": {
            "over": {"const": [{"u": 1}, {"u": 0}, {"u": 1}]},
            "keep": {"equals": {"left": {"item": "u"}, "right": {"const": 1}}},
        }
    }
    assert evaluate_expression(expr, []) == [{"u": 1}, {"u": 1}]


# -- branch in next_action --------------------------------------------------


def test_branch_first_turn_emits_list() -> None:
    action = next_action(_PLAN, [{"role": "user", "content": "go"}])
    assert isinstance(action, EmitToolUse)
    assert action.tool_name == "gmail_list_emails"


def test_branch_taken_when_list_non_empty() -> None:
    action = next_action(_PLAN, _msgs_after_list([{"id": "a"}, {"id": "b"}]))
    assert isinstance(action, EmitToolUse)
    assert action.tool_name == "gmail_reply_to_email"
    assert action.tool_input == {"id": "a"}


def test_branch_not_taken_when_list_empty_punts() -> None:
    # Empty list -> branch else (none) -> body exhausted, no tail -> punt.
    action = next_action(_PLAN, _msgs_after_list([]))
    assert isinstance(action, Punt)


def test_branch_second_reply_uses_second_item() -> None:
    msgs = _msgs_after_list([{"id": "a"}, {"id": "b"}])
    # Append the first reply as executed, so position advances to the 2nd map iter.
    reply = {"type": "tool_use", "id": "t2", "name": "gmail_reply_to_email", "input": {"id": "a"}}
    msgs += [
        {"role": "assistant", "content": [reply]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "sent"}]},
    ]
    action = next_action(_PLAN, msgs)
    assert isinstance(action, EmitToolUse)
    assert action.tool_input == {"id": "b"}


# -- JSON-string tool results (the real-world case) -------------------------


def _msgs_after_list_str(payload: Any) -> list[dict[str, Any]]:
    """Like _msgs_after_list but the tool_result content is a JSON STRING, as real
    integrations return it (not an already-parsed object)."""
    return [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "gmail_list_emails", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": json.dumps(payload)}]},
    ]


def test_last_tool_result_parses_json_string() -> None:
    msgs = _msgs_after_list_str({"emails": [{"id": "a"}]})
    assert last_tool_result(msgs, "gmail_list_emails") == {"emails": [{"id": "a"}]}


def test_last_tool_result_parses_content_block_list() -> None:
    blocks = [{"type": "text", "text": '{"k": 1}'}]
    msgs = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": blocks}]},
    ]
    assert last_tool_result(msgs, "x") == {"k": 1}


def test_last_tool_result_plain_string_unchanged() -> None:
    msgs = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "sent"}]},
    ]
    assert last_tool_result(msgs, "x") == "sent"


def test_branch_and_argmap_serve_over_json_string_result() -> None:
    # The exact real-world failure: branch predicate + arg-map navigate fields of
    # a tool result that arrived as a JSON string. With coercion it now serves.
    action = next_action(_PLAN, _msgs_after_list_str({"emails": [{"id": "a"}, {"id": "b"}]}))
    assert isinstance(action, EmitToolUse)
    assert action.tool_name == "gmail_reply_to_email"
    assert action.tool_input == {"id": "a"}


def test_parse_json_op() -> None:
    assert evaluate_expression({"parse_json": {"const": '{"a": 1}'}}, []) == {"a": 1}
    assert evaluate_expression({"parse_json": {"const": {"a": 1}}}, []) == {"a": 1}


def test_last_tool_result_dollar_root_over_json_string() -> None:
    # Synthesizer commonly emits path "$" (JSONPath root); it means the whole result.
    msgs = _msgs_after_list_str([{"id": "a"}, {"id": "b"}])  # a bare JSON array string
    got = evaluate_expression({"last_tool_result": {"tool": "gmail_list_emails", "path": "$"}}, msgs)
    assert got == [{"id": "a"}, {"id": "b"}]
