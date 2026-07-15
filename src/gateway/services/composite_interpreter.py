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

import json
import re
from dataclasses import dataclass
from typing import Any

MAX_EXPR_DEPTH = 8

# Tool results arrive on the wire as strings (usually JSON) or as content-block
# lists, not as navigable objects. Cap the size we will attempt to json.loads so
# a pathologically large result cannot stall the event loop; over the cap it stays
# a string and any navigation into it fails to a punt.
MAX_JSON_PARSE_INPUT = 262144

# Cap the input length before running a whitelisted regex. The patterns are
# simple but linear-time only for bounded input; the value can come from tenant
# messages (trigger / tool_result), so an unbounded value could stall the async
# event loop. Over-length input punts.
MAX_REGEX_INPUT = 4096

# A ``map`` node expands to one body iteration per element of its runtime list.
# Cap the expansion so a pathologically long list cannot make position
# accounting walk unboundedly; over the cap the interpreter punts to the model.
MAX_MAP_ITEMS = 1000

# Cap nested control-flow depth (``branch``/``map`` nesting) so plan expansion is
# bounded; over the cap the interpreter punts.
MAX_BRANCH_DEPTH = 8

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


@dataclass(frozen=True)
class SubJudgment:
    """The recognized plan reached a below-frontier judgment step (T1). The hook
    serves it with a cheap model when T1 is enabled, else punts to the frontier.
    ``node`` carries the plan's ``sub_judgment`` node (``for``/``output`` hints)."""

    node: dict[str, Any]


Action = EmitToolUse | EmitTerminal | Punt | SubJudgment


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


def _coerce_result(content: Any) -> Any:
    """Coerce a tool_result payload into a navigable object when possible.

    Real tool results come back as JSON strings (or as a list of Anthropic
    content blocks), so arg-maps/predicates that navigate fields
    (``last_tool_result(tool).field``) would otherwise fail on a bare string.
    A string that parses as a JSON object/array is returned parsed; anything
    else (plain text, already-structured value) is returned unchanged.
    """
    if isinstance(content, list):
        # Anthropic content-block list: concatenate its text blocks.
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            content = "".join(texts)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped[:1] in ("{", "[") and len(stripped) <= MAX_JSON_PARSE_INPUT:
            try:
                return json.loads(stripped)
            except (ValueError, TypeError):
                return content
    return content


def last_tool_result(messages: list[dict[str, Any]], tool_name: str) -> Any:
    """Content of the most recent tool_result for ``tool_name`` (by matching the
    tool_use id), coerced to a navigable object when it is a JSON string, or None."""
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
    return _coerce_result(result)


def _navigate(value: Any, path: str) -> Any:
    """Navigate a dotted path over dicts/lists. Numeric segments index lists.

    ``""`` and ``"$"`` (JSONPath root, which the synthesizer commonly emits) both
    mean the whole value; a leading ``"$."`` is stripped.
    """
    if path in ("", "$"):
        return value
    if path.startswith("$."):
        path = path[2:]
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
        # The trigger payload is the first user message's content; coerce a JSON
        # string to an object, then navigate a dotted path into it.
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        payload = _coerce_result(first_user.get("content") if first_user else None)
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
    if op == "filter":
        # {"filter": {"over": <expr>, "keep": <predicate>}} -> the elements of the
        # list for which the predicate holds (each element bound as ``item``).
        # Lets a ``map`` iterate a filtered list, e.g. reply only to matching mail.
        if not isinstance(arg, dict) or "over" not in arg or "keep" not in arg:
            raise ExpressionError("filter needs {over, keep}")
        over = evaluate_expression(arg["over"], messages, depth=depth + 1, item=item)
        if not isinstance(over, list):
            raise ExpressionError("filter over must be a list")
        if len(over) > MAX_MAP_ITEMS:
            raise ExpressionError("filter list too long")
        return [
            element
            for element in over
            if evaluate_predicate(arg["keep"], messages, depth=depth + 1, item=element)
        ]
    if op == "parse_json":
        # Explicitly parse a nested value that is itself a JSON string (the
        # top-level tool_result is already coerced by last_tool_result).
        value = evaluate_expression(arg, messages, depth=depth + 1, item=item)
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str) and len(value) <= MAX_JSON_PARSE_INPUT:
            try:
                return json.loads(value)
            except (ValueError, TypeError) as exc:
                raise ExpressionError("parse_json: invalid JSON") from exc
        raise ExpressionError("parse_json: not a parseable value")
    raise ExpressionError(f"unknown op {op!r}")


def evaluate_predicate(
    pred: Any, messages: list[dict[str, Any]], *, depth: int = 0, item: Any = _NO_ITEM
) -> bool:
    """Evaluate a restricted boolean predicate over visible state, for ``branch``
    and ``filter``. Single-op object; ops: ``non_empty``, ``exists``, ``equals``,
    ``not``, ``and``, ``or``. No general eval, depth-bounded. Raises
    ``ExpressionError`` when it cannot decide (the caller turns that into a punt),
    except ``exists``/``non_empty`` which resolve a missing value to ``False``.
    """
    if depth > MAX_EXPR_DEPTH:
        raise ExpressionError("predicate too deep")
    if not isinstance(pred, dict) or len(pred) != 1:
        raise ExpressionError("predicate must be a single-op object")
    (op, arg), = pred.items()

    if op == "non_empty":
        try:
            value = evaluate_expression(arg, messages, depth=depth + 1, item=item)
        except ExpressionError:
            return False
        if value is None:
            return False
        if isinstance(value, (list, str, dict)):
            return len(value) > 0
        return bool(value)
    if op == "exists":
        try:
            evaluate_expression(arg, messages, depth=depth + 1, item=item)
        except ExpressionError:
            return False
        return True
    if op == "equals":
        if not isinstance(arg, dict) or "left" not in arg or "right" not in arg:
            raise ExpressionError("equals needs {left, right}")
        left = evaluate_expression(arg["left"], messages, depth=depth + 1, item=item)
        right = evaluate_expression(arg["right"], messages, depth=depth + 1, item=item)
        return bool(left == right)
    if op == "not":
        return not evaluate_predicate(arg, messages, depth=depth + 1, item=item)
    if op == "and":
        if not isinstance(arg, list):
            raise ExpressionError("and needs a list of predicates")
        return all(evaluate_predicate(p, messages, depth=depth + 1, item=item) for p in arg)
    if op == "or":
        if not isinstance(arg, list):
            raise ExpressionError("or needs a list of predicates")
        return any(evaluate_predicate(p, messages, depth=depth + 1, item=item) for p in arg)
    raise ExpressionError(f"unknown predicate op {op!r}")


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
    """Parse nodes as ``[(emit_tool_use | map | branch)*, optional trailing node]``.

    Returns ``(body_nodes, tail)``. A ``map`` node expands (at ``_flatten`` time)
    to one body iteration per element of its runtime list, so a plan may serve a
    variable-length loop, not just a fixed sequence. Position is the count of
    executed tool_use turns and stays in lockstep with the flattened steps.

    The body is the leading run of emit/map/branch nodes; the first node of any
    other type becomes the tail (a terminal, a sub_judgment, or an explicit punt).
    Nodes after that first tail are unreachable, a terminal or punt ends the run
    and a sub_judgment is the hand-off boundary where a composite stops serving
    (post-judgment mechanical work is not yet expressible in one plan), so they are
    truncated rather than rejecting the whole plan. This makes a plan always
    degrade to its maximal servable prefix instead of failing wholesale, which is
    both safe (the tail hands off to the frontier) and strictly better for
    coverage than punting the entire automation. Returns ``None`` only for a
    non-list input, which the caller treats as an unsupported shape.
    """
    if not isinstance(nodes, list):
        return None
    body_nodes: list[dict[str, Any]] = []
    tail: dict[str, Any] | None = None
    for node in nodes:
        if tail is not None:
            break
        if node.get("type") in ("emit_tool_use", "map", "branch"):
            body_nodes.append(node)
        else:
            tail = node
    return body_nodes, tail


def _expand(
    nodes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    need: int,
    steps: list[_Step],
    *,
    item: Any,
    depth: int,
) -> bool:
    """Expand a node list into emit steps in ``steps``, lazily and recursively.

    Handles ``emit_tool_use`` (one step), ``map`` (one body iteration per element
    of its runtime list, element bound as ``item``), and ``branch`` (evaluate the
    predicate over visible state and expand the ``then`` or ``else`` list). Stops
    early once ``need`` steps exist. Returns ``False`` on any malformed node or
    unevaluable expression, which the caller turns into a punt.
    """
    if depth > MAX_BRANCH_DEPTH:
        return False
    for node in nodes:
        if len(steps) >= need:
            return True
        if not isinstance(node, dict):
            return False
        ntype = node.get("type")
        if ntype == "emit_tool_use":
            tool = node.get("tool")
            if not isinstance(tool, str):
                return False
            steps.append(_Step(tool, node, item))
        elif ntype == "map":
            body = node.get("body")
            if not isinstance(body, list) or not body:
                return False
            try:
                over = evaluate_expression(node.get("over"), messages, item=item)
            except (ExpressionError, KeyError, TypeError, ValueError, IndexError, RecursionError):
                return False
            if not isinstance(over, list) or len(over) > MAX_MAP_ITEMS:
                return False
            for element in over:
                if not _expand(body, messages, need, steps, item=element, depth=depth + 1):
                    return False
                if len(steps) >= need:
                    return True
        elif ntype == "branch":
            try:
                took = evaluate_predicate(node.get("predicate"), messages, item=item)
            except (ExpressionError, KeyError, TypeError, ValueError, IndexError, RecursionError):
                return False
            chosen = node.get("then") if took else node.get("else")
            chosen = chosen or []
            if not isinstance(chosen, list):
                return False
            if not _expand(chosen, messages, need, steps, item=item, depth=depth + 1):
                return False
        else:
            return False
    return True


def _flatten(
    body_nodes: list[dict[str, Any]], messages: list[dict[str, Any]], need: int
) -> list[_Step] | None:
    """Expand body nodes into an ordered list of emit steps, lazily.

    Wraps ``_expand``; returns ``None`` (the caller punts) on any malformed node,
    unevaluable ``over``/``predicate``, or over-deep nesting.
    """
    steps: list[_Step] = []
    if not _expand(body_nodes, messages, need, steps, item=_NO_ITEM, depth=0):
        return None
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
        # A below-frontier judgment step: the hook serves it with a cheap model
        # (T1) when enabled, otherwise punts to the frontier.
        return SubJudgment(node=tail)
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
