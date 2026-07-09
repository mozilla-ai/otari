"""Declarative composite-plan interpreter.

A composite version is a declarative plan (not a program): a state matcher plus
an ordered set of typed action nodes, interpreted by this fixed, vetted code. No
model-authored code runs here (docs/tool-compositor-layer-plan.md sec 4). The
interpreter is a pure function of visible message state: given a matched plan
and the conversation so far (Anthropic-format message dicts), ``next_action``
yields a tool_use to emit, a terminal, or PUNT.

The arg-map / matcher expression language is deliberately tiny: path extraction
over the visible JSON plus a fixed set of pure transforms. There is no general
eval, no I/O, and evaluation is depth-bounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MAX_EXPR_DEPTH = 8

# Whitelisted regex patterns the ``regex_extract`` transform may use. A plan
# cannot supply an arbitrary pattern (avoids catastrophic backtracking and
# keeps evaluation bounded).
_REGEX_WHITELIST: dict[str, str] = {
    "iso_date": r"(\d{4}-\d{2}-\d{2})",
    "leading_int": r"^\s*(\d+)",
    "email": r"([\w.+-]+@[\w-]+\.[\w.-]+)",
}


class ExpressionError(Exception):
    """Raised when an expression cannot be evaluated; forces a punt."""


@dataclass(frozen=True)
class EmitToolUse:
    tool_name: str
    tool_input: dict[str, Any]


@dataclass(frozen=True)
class EmitTerminal:
    text: str


@dataclass(frozen=True)
class Punt:
    reason: str


Action = EmitToolUse | EmitTerminal | Punt


# ---------------------------------------------------------------------------
# Visible-state helpers
# ---------------------------------------------------------------------------


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def executed_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    """Tool names from assistant tool_use blocks, in order."""
    return [
        block["name"]
        for message in messages
        if message.get("role") == "assistant"
        for block in _content_blocks(message)
        if block.get("type") == "tool_use" and isinstance(block.get("name"), str)
    ]


def last_tool_result(messages: list[dict[str, Any]], tool_name: str) -> Any:
    """Content of the most recent tool_result for ``tool_name`` (by matching the
    tool_use id), or None."""
    # Map tool_use_id -> tool name from assistant turns.
    id_to_name: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for block in _content_blocks(message):
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    id_to_name[block["id"]] = block.get("name", "")
    result: Any = None
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in _content_blocks(message):
            if block.get("type") == "tool_result":
                tuid = block.get("tool_use_id")
                if isinstance(tuid, str) and id_to_name.get(tuid) == tool_name:
                    result = block.get("content")
    return result


def _navigate(value: Any, path: str) -> Any:
    """Navigate a dotted path over dicts/lists. Numeric segments index lists."""
    if path == "":
        return value
    current = value
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise ExpressionError(f"missing key {segment!r}")
            current = current[segment]
        elif isinstance(current, list):
            try:
                idx = int(segment)
            except ValueError as exc:
                raise ExpressionError(f"non-int index {segment!r} on list") from exc
            if not -len(current) <= idx < len(current):
                raise ExpressionError(f"index {idx} out of range")
            current = current[idx]
        else:
            raise ExpressionError(f"cannot navigate {segment!r} on {type(current).__name__}")
    return current


# ---------------------------------------------------------------------------
# Expression language
# ---------------------------------------------------------------------------


def evaluate_expression(expr: Any, messages: list[dict[str, Any]], *, depth: int = 0) -> Any:
    """Evaluate a restricted arg-map expression over visible message state.

    Supported ops (each a single-key dict): ``const``, ``trigger``,
    ``last_tool_result``, ``lower``, ``join``, ``regex_extract``. Anything else
    raises ``ExpressionError`` (which the caller converts to a punt).
    """
    if depth > MAX_EXPR_DEPTH:
        raise ExpressionError("expression too deep")
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ExpressionError("expression must be a single-op object")
    (op, arg), = expr.items()

    if op == "const":
        return arg
    if op == "trigger":
        # The trigger payload is the first user message's content when it is a
        # structured block; navigate a dotted path into it.
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        payload = first_user.get("content") if first_user else None
        return _navigate(payload, str(arg))
    if op == "last_tool_result":
        if not isinstance(arg, dict):
            raise ExpressionError("last_tool_result needs {tool, path}")
        content = last_tool_result(messages, str(arg.get("tool", "")))
        return _navigate(content, str(arg.get("path", "")))
    if op == "lower":
        value = evaluate_expression(arg, messages, depth=depth + 1)
        return str(value).lower()
    if op == "join":
        if not isinstance(arg, dict):
            raise ExpressionError("join needs {sep, values}")
        sep = str(arg.get("sep", ""))
        values = arg.get("values", [])
        if not isinstance(values, list):
            raise ExpressionError("join values must be a list")
        parts = [str(evaluate_expression(v, messages, depth=depth + 1)) for v in values]
        return sep.join(parts)
    if op == "regex_extract":
        if not isinstance(arg, dict):
            raise ExpressionError("regex_extract needs {value, pattern}")
        pattern_name = str(arg.get("pattern", ""))
        pattern = _REGEX_WHITELIST.get(pattern_name)
        if pattern is None:
            raise ExpressionError(f"pattern {pattern_name!r} not whitelisted")
        value = str(evaluate_expression(arg["value"], messages, depth=depth + 1))
        match = re.search(pattern, value)
        if match is None:
            raise ExpressionError("regex did not match")
        return match.group(1)
    raise ExpressionError(f"unknown op {op!r}")


# ---------------------------------------------------------------------------
# Matcher, next_action, verifier
# ---------------------------------------------------------------------------


def _actionable_nodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = plan.get("nodes", [])
    return [n for n in nodes if isinstance(n, dict) and n.get("type")]


def matches_envelope(plan: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    """Cheap, modelless recognize check: the tool_use sequence so far must be a
    prefix of the plan's expected sequence. A deviation means the run left the
    validated envelope and must punt."""
    expected = [n.get("tool") for n in _actionable_nodes(plan) if n.get("type") == "emit_tool_use"]
    executed = executed_tool_names(messages)
    if len(executed) > len(expected):
        return False
    return executed == expected[: len(executed)]


def next_action(plan: dict[str, Any], messages: list[dict[str, Any]]) -> Action:
    """Yield the next action for a matched plan given the visible messages."""
    if not matches_envelope(plan, messages):
        return Punt("out_of_envelope")

    nodes = _actionable_nodes(plan)
    index = len(executed_tool_names(messages))
    if index >= len(nodes):
        return Punt("plan_exhausted")

    node = nodes[index]
    node_type = node.get("type")

    if node_type == "emit_tool_use":
        tool = node.get("tool")
        if not isinstance(tool, str):
            return Punt("malformed_node")
        args_spec = node.get("args", {})
        if not isinstance(args_spec, dict):
            return Punt("malformed_args")
        tool_input: dict[str, Any] = {}
        try:
            for arg_name, arg_expr in args_spec.items():
                tool_input[arg_name] = evaluate_expression(arg_expr, messages)
        except ExpressionError:
            return Punt("arg_expression_failed")
        return EmitToolUse(tool_name=tool, tool_input=tool_input)

    if node_type == "emit_terminal":
        return EmitTerminal(text=str(node.get("text", "")))

    if node_type == "sub_judgment":
        # T1 sub-judgment requires a small-model call, handled by the T1 node
        # (a later milestone). Until then, punt so the frontier serves the turn.
        return Punt("t1_not_enabled")

    if node_type == "punt":
        return Punt("plan_punt")

    return Punt("unknown_node_type")


def verify_action(
    action: Action,
    verifier_spec: dict[str, Any],
    plan: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bool:
    """Mechanical check on the produced action. On failure the caller punts.

    ``action_sequence_match``: the emitted tool must equal the expected tool at
    the current position.
    """
    spec_type = verifier_spec.get("type")
    if spec_type == "action_sequence_match":
        if not isinstance(action, EmitToolUse):
            return True  # terminals/punts are not sequence-checked
        expected = verifier_spec.get("expected_tools", [])
        index = len(executed_tool_names(messages))
        if not isinstance(expected, list) or index >= len(expected):
            return False
        return bool(action.tool_name == expected[index])
    # No verifier or unknown type: do not block (the caller may still punt on
    # other signals).
    return True
