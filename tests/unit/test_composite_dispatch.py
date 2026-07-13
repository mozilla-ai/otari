"""Unit tests for the composite dispatch decision."""

from __future__ import annotations

from typing import Any

from gateway.services.composite_dispatch import (
    NoDispatch,
    Serve,
    Shadow,
    action_matches_model,
    decide,
)
from gateway.services.composite_interpreter import EmitToolUse, Punt

_MODAL = ["slack_read_channel_messages", "resolve_time", "google_sheets_append_row"]


def _composite(status: str, *, automation_key: str = "automation:9cb676db") -> dict[str, Any]:
    return {
        "automation_key": automation_key,
        "name": f"auto:{automation_key}",
        "status": status,
        "plan": {"nodes": [{"type": "emit_tool_use", "tool": t, "args": {}} for t in _MODAL]},
        "verifier_spec": {"type": "action_sequence_match", "expected_tools": _MODAL},
    }


def _user_turn() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "go"}]


def test_no_session_label_no_dispatch() -> None:
    d = decide([_composite("approved")], _user_turn(), session_label=None)
    assert isinstance(d, NoDispatch)
    assert d.reason == "no_session_label"


def test_no_matching_composite_no_dispatch() -> None:
    d = decide([_composite("approved")], _user_turn(), session_label="automation:other")
    assert isinstance(d, NoDispatch)
    assert d.reason == "no_match"


def test_approved_serves_first_action() -> None:
    d = decide([_composite("approved")], _user_turn(), session_label="automation:9cb676db")
    assert isinstance(d, Serve)
    assert isinstance(d.action, EmitToolUse)
    assert d.action.tool_name == "slack_read_channel_messages"


def test_shadow_status_yields_shadow_decision() -> None:
    d = decide([_composite("shadow")], _user_turn(), session_label="automation:9cb676db")
    assert isinstance(d, Shadow)
    assert isinstance(d.action, EmitToolUse)


def test_approved_punts_out_of_envelope() -> None:
    messages = [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "wrong", "input": {}}]}]
    d = decide([_composite("approved")], messages, session_label="automation:9cb676db")
    assert isinstance(d, NoDispatch)
    assert d.reason == "out_of_envelope"


def test_shadow_records_punt_as_shadow() -> None:
    messages = [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "wrong", "input": {}}]}]
    d = decide([_composite("shadow")], messages, session_label="automation:9cb676db")
    assert isinstance(d, Shadow)
    assert isinstance(d.action, Punt)


def test_verifier_failure_blocks_serve() -> None:
    composite = _composite("approved")
    # Break the verifier so the expected tool at index 0 mismatches the plan's tool.
    composite["verifier_spec"] = {"type": "action_sequence_match", "expected_tools": ["different_tool"]}
    d = decide([composite], _user_turn(), session_label="automation:9cb676db")
    assert isinstance(d, NoDispatch)
    assert d.reason == "verifier_failed"


def test_draft_status_not_dispatchable() -> None:
    d = decide([_composite("draft")], _user_turn(), session_label="automation:9cb676db")
    assert isinstance(d, NoDispatch)
    assert d.reason == "not_dispatchable_status"


def test_decide_tolerates_non_dict_composite_entries() -> None:
    # A malformed platform payload (a non-dict entry) must not crash decide().
    composites = ["not-a-dict", _composite("approved")]
    d = decide(composites, _user_turn(), session_label="automation:9cb676db")  # type: ignore[list-item]
    assert isinstance(d, Serve)


def test_decide_malformed_plan_no_dispatch() -> None:
    composite = {"automation_key": "automation:9cb676db", "status": "approved", "plan": "not-a-dict"}
    d = decide([composite], _user_turn(), session_label="automation:9cb676db")
    assert isinstance(d, NoDispatch)


def test_required_tools_gate() -> None:
    composite = _composite("approved")
    composite["recognize_envelope"] = {"required_tools": ["resolve_time", "gmail_task"]}
    label = "automation:9cb676db"
    # Main agent (has both tools) -> serves.
    assert isinstance(
        decide([composite], _user_turn(), session_label=label, tool_names=["resolve_time", "gmail_task"]),
        Serve,
    )
    # Sub-agent (missing resolve_time) -> no dispatch.
    d = decide([composite], _user_turn(), session_label=label, tool_names=["gmail_list_emails"])
    assert isinstance(d, NoDispatch)
    assert d.reason == "tools_mismatch"
    # Tools unknown -> no dispatch (fail safe).
    assert isinstance(decide([composite], _user_turn(), session_label=label, tool_names=None), NoDispatch)


def _role_composite(suffix: str, required: list[str], first_tool: str) -> dict[str, Any]:
    key = "automation:multi"
    return {
        "automation_key": key,
        "name": f"auto:{key}#{suffix}",
        "status": "approved",
        "plan": {"nodes": [{"type": "emit_tool_use", "tool": first_tool, "args": {}}]},
        "verifier_spec": {"type": "action_sequence_match"},
        "recognize_envelope": {"automation_key": key, "required_tools": required},
    }


def test_role_selection_prefers_most_specific() -> None:
    # One automation, two roles under the same label: a main orchestrator and a
    # sub-agent whose tools are a superset. Each request must bind to its own role.
    main = _role_composite("main", ["resolve_time"], "resolve_time")
    sub = _role_composite("sub", ["resolve_time", "gmail_list_emails"], "gmail_list_emails")
    label = "automation:multi"

    # Sub-agent request satisfies BOTH; the more specific (sub) composite wins.
    d_sub = decide(
        [main, sub],
        _user_turn(),
        session_label=label,
        tool_names=["resolve_time", "gmail_list_emails", "gmail_reply_to_email"],
    )
    assert isinstance(d_sub, Serve)
    assert d_sub.action.tool_name == "gmail_list_emails"

    # Main-agent request (no sub tools) satisfies only the main composite.
    d_main = decide([main, sub], _user_turn(), session_label=label, tool_names=["resolve_time", "gmail_task"])
    assert isinstance(d_main, Serve)
    assert d_main.action.tool_name == "resolve_time"


def test_action_matches_model_name_level() -> None:
    action = EmitToolUse(tool_name="resolve_time", tool_input={})
    assert action_matches_model(action, {"type": "tool_use", "name": "resolve_time"}) is True
    assert action_matches_model(action, {"type": "tool_use", "name": "other"}) is False
    assert action_matches_model(action, None) is False
    assert action_matches_model(Punt("x"), {"type": "tool_use", "name": "resolve_time"}) is False
