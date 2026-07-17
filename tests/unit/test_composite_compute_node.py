"""Unit tests for the ``compute`` plan node: a code-exec step that binds a value
derived from prior tool results, folded into the plan via ``materialize_plan``
before the pure interpreter runs.

Models the real Inbox Organizer (automation e7a84d93) sub-agent loop: list unread
emails, then read each and fetch each thread. The list result is unstructured
markdown, so extracting the message ids is a multi-match parse the arg-map DSL
cannot express (``regex_extract`` yields a single group). A compute node produces
the id list; two ``map`` nodes drive the read/thread loop over it.
"""

from __future__ import annotations

from typing import Any

from gateway.services.composite_interpreter import (
    EmitToolUse,
    Punt,
    compute_nodes,
    matches_envelope,
    materialize_plan,
    next_action,
)

_LIST_RESULT = (
    "# Email Search Results\nFound 3 emails matching your criteria.\n\n"
    "- [msg_id:19f65405572bff12] [thread_id:19f65405572bff12] Subject: otari is good\n"
    "- [msg_id:19f653757c83bd78] [thread_id:19f653757c83bd78] Subject: Kutxabank\n"
    "- [msg_id:19f653454a9cebd3] [thread_id:19f653454a9cebd3] Subject: Otari what's that?\n"
)
_IDS = ["19f65405572bff12", "19f653757c83bd78", "19f653454a9cebd3"]


def _plan() -> dict[str, Any]:
    return {
        "version": 1,
        "nodes": [
            {"type": "emit_tool_use", "tool": "gmail_list_emails",
             "args": {"query": {"const": "is:unread"}}},
            {"type": "compute", "bind": "ids",
             "code": "import re\noutput = re.findall(r'\\[msg_id:([0-9a-f]+)\\]', results[-1])"},
            {"type": "map", "over": {"var": "ids"},
             "body": [{"type": "emit_tool_use", "tool": "gmail_read_email",
                       "args": {"message_id": {"item": ""}}}]},
            {"type": "map", "over": {"var": "ids"},
             "body": [{"type": "emit_tool_use", "tool": "gmail_get_thread_replies",
                       "args": {"message_id": {"item": ""}}}]},
        ],
    }


def _assistant(name: str, tool_id: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]}


def _result(tool_use_id: str, content: Any) -> dict[str, Any]:
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}]}


def _run_compute(node: dict[str, Any], results: list[str]) -> Any:
    """Stand in for otari-exec: run the node's code with `results` in scope.

    Mirrors the sandbox contract (the model code sets ``output`` from ``results``);
    here we exec locally so the test needs no live sandbox. The real executor runs
    this identical code in the credential-free otari-exec sandbox.
    """
    ns: dict[str, Any] = {"results": results, "bindings": {}}
    exec(node["code"], ns)  # noqa: S102 - trusted test input mirroring the sandbox
    return ns["output"]


def test_compute_nodes_lists_top_level_compute() -> None:
    nodes = compute_nodes(_plan())
    assert [n["bind"] for n in nodes] == ["ids"]


def test_compute_code_extracts_all_message_ids() -> None:
    # The value the sandbox would produce from the real, unstructured list result.
    assert _run_compute(compute_nodes(_plan())[0], [_LIST_RESULT]) == _IDS


def test_materialized_plan_drives_the_whole_read_then_thread_loop() -> None:
    plan = _plan()
    messages: list[dict[str, Any]] = [{"role": "user", "content": "organize inbox"}]

    # Turn 1: nothing executed yet -> emit the list call (no compute needed).
    action = next_action(materialize_plan(plan, {}), messages)
    assert isinstance(action, EmitToolUse)
    assert action.tool_name == "gmail_list_emails"

    # The list runs; its result becomes visible.
    messages.append(_assistant("gmail_list_emails", "t1", {"query": "is:unread"}))
    messages.append(_result("t1", _LIST_RESULT))

    # The hook resolves the compute node from the visible results, then materializes.
    bindings = {"ids": _run_compute(compute_nodes(plan)[0], [_LIST_RESULT])}
    mat = materialize_plan(plan, bindings)

    # Now the interpreter drives read x3 then thread x3, byte-faithful ids.
    expected = [("gmail_read_email", i) for i in _IDS] + [("gmail_get_thread_replies", i) for i in _IDS]
    tid = 2
    for tool, msg_id in expected:
        action = next_action(mat, messages)
        assert isinstance(action, EmitToolUse), action
        assert (action.tool_name, action.tool_input) == (tool, {"message_id": msg_id})
        messages.append(_assistant(tool, f"t{tid}", action.tool_input))
        messages.append(_result(f"t{tid}", "ok"))
        tid += 1

    # Loop complete -> the plan exhausts and hands off (punt), never mis-serves.
    assert isinstance(next_action(mat, messages), Punt)


def test_matches_envelope_holds_through_the_loop() -> None:
    plan = _plan()
    bindings = {"ids": _IDS}
    mat = materialize_plan(plan, bindings)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "organize inbox"},
        _assistant("gmail_list_emails", "t1", {"query": "is:unread"}),
        _result("t1", _LIST_RESULT),
        _assistant("gmail_read_email", "t2", {"message_id": _IDS[0]}),
        _result("t2", "ok"),
    ]
    assert matches_envelope(mat, messages) is True


def test_unresolved_var_punts_never_serves() -> None:
    # If the compute binding is missing (sandbox unreachable), the var survives
    # materialization, the map's `over` fails to evaluate, and the turn punts.
    plan = _plan()
    mat = materialize_plan(plan, {})  # no bindings resolved
    messages = [
        {"role": "user", "content": "organize inbox"},
        _assistant("gmail_list_emails", "t1", {"query": "is:unread"}),
        _result("t1", _LIST_RESULT),
    ]
    assert isinstance(next_action(mat, messages), Punt)
