"""Unit tests for the branch/filter/predicate additions to the interpreter."""

from __future__ import annotations

from typing import Any

from gateway.services.composite_interpreter import (
    EmitToolUse,
    Punt,
    evaluate_expression,
    evaluate_predicate,
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
