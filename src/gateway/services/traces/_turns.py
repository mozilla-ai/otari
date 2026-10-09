"""What a request says about the agent loop it is part of, read from its own body.

Two facts, and nothing else is kept:

- whether the request opens a turn: its last input is the user's own text, so a
  new prompt starts here, rather than a tool result the agent is handing back;
- which tool calls it answers: the results at the end of its input, each named
  by the tool call in the history that asked for it, and whether it failed.

The second is how tools the client ran itself (a shell, a file read) enter the
trace: the request that carries a result is the one that saw the tool finish.
Only the call id, the tool's name and the error flag are read; no argument or
result content is. The body is the client's, so every field is type-checked
before use: a malformed one reads as absent, never as an error.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Dialect = Literal["chat", "messages", "responses"]


@dataclass(frozen=True)
class AnsweredCall:
    """A tool call the client ran, answered in this request."""

    call_id: str
    name: str | None
    is_error: bool


@dataclass(frozen=True)
class TurnFacts:
    opens_turn: bool
    answered: tuple[AnsweredCall, ...] = ()


def read_turn(dialect: Dialect, input_items: Any) -> TurnFacts:
    """Read a request's messages (Chat, Messages) or input items (Responses)."""
    if dialect == "responses":
        return _responses(input_items)
    if not isinstance(input_items, list) or not input_items:
        return TurnFacts(opens_turn=False)
    return _chat(input_items) if dialect == "chat" else _messages(input_items)


def _dicts(items: Any) -> list[Mapping[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _chat(messages: list[Any]) -> TurnFacts:
    items = _dicts(messages)
    if not items:
        return TurnFacts(opens_turn=False)
    if items[-1].get("role") == "user":
        return TurnFacts(opens_turn=True)
    trailing: list[Mapping[str, Any]] = []
    for message in reversed(items):
        if message.get("role") != "tool":
            break
        trailing.append(message)
    names: dict[str, str | None] = {}
    for message in items:
        if message.get("role") == "assistant":
            for call in _dicts(message.get("tool_calls")):
                if isinstance(call.get("id"), str):
                    function = call.get("function")
                    names[call["id"]] = _str(function.get("name")) if isinstance(function, Mapping) else None
    answered = tuple(
        AnsweredCall(call_id=message["tool_call_id"], name=names.get(message["tool_call_id"]), is_error=False)
        for message in reversed(trailing)
        if isinstance(message.get("tool_call_id"), str)
    )
    return TurnFacts(opens_turn=False, answered=answered)


def _messages(messages: list[Any]) -> TurnFacts:
    items = _dicts(messages)
    if not items or items[-1].get("role") != "user":
        return TurnFacts(opens_turn=False)
    content = items[-1].get("content")
    if isinstance(content, str):
        return TurnFacts(opens_turn=True)
    blocks = _dicts(content)
    results = [block for block in blocks if block.get("type") == "tool_result"]
    if not results:
        return TurnFacts(opens_turn=any(block.get("type") == "text" for block in blocks))
    names: dict[str, str | None] = {}
    for message in items:
        if message.get("role") == "assistant":
            for block in _dicts(message.get("content")):
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    names[block["id"]] = _str(block.get("name"))
    answered = tuple(
        AnsweredCall(
            call_id=block["tool_use_id"], name=names.get(block["tool_use_id"]), is_error=bool(block.get("is_error"))
        )
        for block in results
        if isinstance(block.get("tool_use_id"), str)
    )
    return TurnFacts(opens_turn=False, answered=answered)


_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})


def _responses(input_items: Any) -> TurnFacts:
    if isinstance(input_items, str):
        return TurnFacts(opens_turn=bool(input_items))
    if not isinstance(input_items, list):
        return TurnFacts(opens_turn=False)
    items = _dicts(input_items)
    if not items:
        return TurnFacts(opens_turn=False)
    last = items[-1]
    if _str(last.get("type", "message")) == "message" and last.get("role") == "user":
        return TurnFacts(opens_turn=True)
    trailing: list[Mapping[str, Any]] = []
    for item in reversed(items):
        if _str(item.get("type")) not in _OUTPUT_TYPES:
            break
        trailing.append(item)
    names = {
        item["call_id"]: _str(item.get("name"))
        for item in items
        if _str(item.get("type")) in _CALL_TYPES and isinstance(item.get("call_id"), str)
    }
    answered = tuple(
        AnsweredCall(call_id=item["call_id"], name=names.get(item["call_id"]), is_error=False)
        for item in reversed(trailing)
        if isinstance(item.get("call_id"), str)
    )
    return TurnFacts(opens_turn=False, answered=answered)
