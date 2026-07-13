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

# Cap the input length before running a whitelisted regex. The patterns are
# simple but linear-time only for bounded input; the value can come from tenant
# messages (trigger / tool_result), so an unbounded value could stall the async
# event loop. Over-length input punts.
MAX_REGEX_INPUT = 4096

# A ``map`` node expands to one body iteration per element of its runtime list.
# Cap the expansion so a pathologically long list cannot make position
# accounting walk unboundedly; over the cap the interpreter punts to the model.
MAX_MAP_ITEMS = 1000

# Sentinel distinguishing "no current map item" from a real item value of None.
_NO_ITEM: Any = object()

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


def evaluate_expression(
    expr: Any, messages: list[dict[str, Any]], *, depth: int = 0, item: Any = _NO_ITEM
) -> Any:
    """Evaluate a restricted arg-map expression over visible message state.

    Supported ops (each a single-key dict): ``const``, ``trigger``,
    ``last_tool_result``, ``item``, ``lower``, ``join``, ``regex_extract``.
    Anything else raises ``ExpressionError`` (which the caller converts to a
    punt). ``item`` resolves a dotted path into the current ``map`` loop element
    and is only valid inside a ``map`` body (``item`` is ``_NO_ITEM`` elsewhere).
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
    if op == "item":
        if item is _NO_ITEM:
            raise ExpressionError("item reference outside a map body")
        return _navigate(item, str(arg))
    if op == "lower":
        value = evaluate_expression(arg, messages, depth=depth + 1, item=item)
        return str(value).lower()
    if op == "join":
        if not isinstance(arg, dict):
            raise ExpressionError("join needs {sep, values}")
        sep = str(arg.get("sep", ""))
        values = arg.get("values", [])
        if not isinstance(values, list):
            raise ExpressionError("join values must be a list")
        parts = [str(evaluate_expression(v, messages, depth=depth + 1, item=item)) for v in values]
        return sep.join(parts)
    if op == "regex_extract":
        if not isinstance(arg, dict) or "value" not in arg:
            raise ExpressionError("regex_extract needs {value, pattern}")
        pattern_name = str(arg.get("pattern", ""))
        pattern = _REGEX_WHITELIST.get(pattern_name)
        if pattern is None:
            raise ExpressionError(f"pattern {pattern_name!r} not whitelisted")
        value = str(evaluate_expression(arg["value"], messages, depth=depth + 1, item=item))
        if len(value) > MAX_REGEX_INPUT:
            raise ExpressionError("regex input too long")
        match = re.search(pattern, value)
        if match is None:
            raise ExpressionError("regex did not match")
        return match.group(1)
    raise ExpressionError(f"unknown op {op!r}")


# ---------------------------------------------------------------------------
# Matcher, next_action, verifier
# ---------------------------------------------------------------------------


def _actionable_nodes(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    nodes = plan.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    return [n for n in nodes if isinstance(n, dict) and n.get("type")]


@dataclass(frozen=True)
class _Step:
    """One flattened emit position: the tool to emit, the node carrying its arg
    map, and the current ``map`` element (``_NO_ITEM`` outside a map body)."""

    tool: str
    node: dict[str, Any]
    item: Any


def _parse_plan(
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None] | None:
    """Parse nodes as ``[(emit_tool_use | map)*, optional single trailing node]``.

    Returns ``(body_nodes, tail)`` for that supported shape, else ``None``. A
    ``map`` node expands (at ``_flatten`` time) to one body iteration per element
    of its runtime list, so a plan may now serve a variable-length loop, not just
    a fixed sequence. Position is still the count of executed tool_use turns and
    stays in lockstep with the flattened steps; any node type other than
    emit/map may appear only once, as the trailing node (a terminal, a
    sub_judgment, or an explicit punt). Anything else is unsupported and punts.
    """
    body_nodes: list[dict[str, Any]] = []
    tail: dict[str, Any] | None = None
    for node in nodes:
        if node.get("type") in ("emit_tool_use", "map"):
            if tail is not None:
                return None
            body_nodes.append(node)
        else:
            if tail is not None:
                return None
            tail = node
    return body_nodes, tail


def _flatten(
    body_nodes: list[dict[str, Any]], messages: list[dict[str, Any]], need: int
) -> list[_Step] | None:
    """Expand body nodes into an ordered list of emit steps, lazily.

    Expansion stops once ``need`` steps exist (the caller only ever needs up to
    the current position), so a ``map`` is evaluated only once the run has
    actually reached it, by which point its list-producing tool_result is already
    visible. Returns ``None`` on any malformed node or unevaluable ``over``
    expression, which the caller turns into a punt.
    """
    steps: list[_Step] = []
    for node in body_nodes:
        if len(steps) >= need:
            break
        ntype = node.get("type")
        if ntype == "emit_tool_use":
            tool = node.get("tool")
            if not isinstance(tool, str):
                return None
            steps.append(_Step(tool, node, _NO_ITEM))
            continue
        # map: one body iteration per element of the runtime list.
        body = node.get("body")
        if not isinstance(body, list) or not body:
            return None
        for inner in body:
            if (
                not isinstance(inner, dict)
                or inner.get("type") != "emit_tool_use"
                or not isinstance(inner.get("tool"), str)
            ):
                return None
        try:
            over = evaluate_expression(node.get("over"), messages)
        except (ExpressionError, KeyError, TypeError, ValueError, IndexError, RecursionError):
            return None
        if not isinstance(over, list) or len(over) > MAX_MAP_ITEMS:
            return None
        for element in over:
            steps.extend(_Step(inner["tool"], inner, element) for inner in body)
            if len(steps) >= need:
                break
    return steps


def matches_envelope(plan: Any, messages: list[dict[str, Any]]) -> bool:
    """Cheap, modelless recognize check: the tool_use sequence so far must be a
    prefix of the plan's expected emit sequence, and the plan must be a
    supported shape. A deviation (or an unsupported shape) means the run left the
    validated envelope and must punt."""
    if not isinstance(plan, dict):
        return False
    parsed = _parse_plan(_actionable_nodes(plan))
    if parsed is None:
        return False
    body_nodes, _tail = parsed
    executed = executed_tool_names(messages)
    steps = _flatten(body_nodes, messages, len(executed) + 1)
    if steps is None:
        return False
    expected = [s.tool for s in steps]
    if len(executed) > len(expected):
        return False
    return executed == expected[: len(executed)]


def next_action(plan: Any, messages: list[dict[str, Any]]) -> Action:
    """Yield the next action for a matched plan given the visible messages."""
    if not isinstance(plan, dict):
        return Punt("malformed_plan")
    parsed = _parse_plan(_actionable_nodes(plan))
    if parsed is None:
        return Punt("unsupported_plan_shape")
    body_nodes, tail = parsed

    executed = executed_tool_names(messages)
    steps = _flatten(body_nodes, messages, len(executed) + 1)
    if steps is None:
        return Punt("map_expansion_failed")
    expected = [s.tool for s in steps]
    if len(executed) > len(expected) or executed != expected[: len(executed)]:
        return Punt("out_of_envelope")

    index = len(executed)
    if index < len(steps):
        step = steps[index]
        args_spec = step.node.get("args", {})
        if not isinstance(args_spec, dict):
            return Punt("malformed_args")
        tool_input: dict[str, Any] = {}
        try:
            for arg_name, arg_expr in args_spec.items():
                tool_input[arg_name] = evaluate_expression(arg_expr, messages, item=step.item)
        except (ExpressionError, KeyError, TypeError, ValueError, IndexError, RecursionError):
            # A plan is semi-trusted data; any evaluation failure degrades to a
            # punt rather than crashing the request path.
            return Punt("arg_expression_failed")
        return EmitToolUse(tool_name=step.tool, tool_input=tool_input)

    # All body steps served; handle the optional trailing node.
    if tail is None:
        return Punt("plan_exhausted")
    tail_type = tail.get("type")
    if tail_type == "emit_terminal":
        return EmitTerminal(text=str(tail.get("text", "")))
    if tail_type == "sub_judgment":
        # T1 sub-judgment requires a small-model call (a later milestone).
        # Until then, punt so the frontier serves the turn.
        return Punt("t1_not_enabled")
    if tail_type == "punt":
        return Punt("plan_punt")
    return Punt("unknown_node_type")


def verify_action(
    action: Action,
    verifier_spec: Any,
    plan: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bool:
    """Mechanical check on the produced action. On failure the caller punts.

    ``action_sequence_match``: the emitted tool must equal the expected tool at
    the current position. When the spec carries an explicit ``expected_tools``
    list it is used directly; otherwise the expected tool is derived from the
    plan itself (map-aware), so a synthesized plan that ships no static tool list
    is still mechanically checked rather than blocked.
    """
    if not isinstance(verifier_spec, dict):
        return True
    if verifier_spec.get("type") != "action_sequence_match":
        # No verifier or unknown type: do not block (the caller may still punt on
        # other signals).
        return True
    if not isinstance(action, EmitToolUse):
        return True  # terminals/punts are not sequence-checked
    index = len(executed_tool_names(messages))

    expected_tools = verifier_spec.get("expected_tools")
    if isinstance(expected_tools, list) and expected_tools:
        if index >= len(expected_tools):
            return False
        return bool(action.tool_name == expected_tools[index])

    # Derive the expected tool from the plan's own flattened steps.
    parsed = _parse_plan(_actionable_nodes(plan))
    if parsed is None:
        return False
    body_nodes, _tail = parsed
    steps = _flatten(body_nodes, messages, index + 1)
    if steps is None or index >= len(steps):
        return False
    return bool(action.tool_name == steps[index].tool)
