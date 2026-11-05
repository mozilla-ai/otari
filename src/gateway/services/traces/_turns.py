"""What a request says about the agent loop it is part of, read from its own body.

Two facts, and nothing else is kept:

- whether the request opens a turn: its last input is the user's own text, so a
  new prompt starts here, rather than a tool result the agent is handing back;
- which tool calls it answers: the results at the end of its input, each named
  by the tool call in the history that asked for it, and whether it failed.

The second is how tools the client ran itself (a shell, a file read) enter the
trace: the request that carries a result is the one that saw the tool finish.
Only the call id, the tool's name and the error flag are read. The prompt, and
each answered call's arguments and result, are held as the body's own values,
unread, for :mod:`._turn_content` to read where the request keeps content. The
body is the client's, so every field is type-checked before use: a malformed
one reads as absent, never as an error.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Dialect = Literal["chat", "messages", "responses"]


@dataclass(frozen=True)
class AnsweredCall:
    """A tool call the client ran, answered in this request.

    ``position`` is the index in the input of the item carrying the result. A
    retried request repeats it, while a later call reusing the same id answers
    further down a longer history, so the two stay apart.
    """

    call_id: str
    name: str | None
    is_error: bool
    position: int = 0
    arguments: Any = field(default=None, compare=False, repr=False)
    result: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class TurnFacts:
    opens_turn: bool
    answered: tuple[AnsweredCall, ...] = ()
    prompt: Any = field(default=None, compare=False, repr=False)


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
        return TurnFacts(opens_turn=True, prompt=items[-1].get("content"))
    trailing: list[tuple[int, Mapping[str, Any]]] = []
    for position in range(len(items) - 1, -1, -1):
        if items[position].get("role") != "tool":
            break
        trailing.append((position, items[position]))
    calls: dict[str, Mapping[str, Any]] = {}
    for message in items:
        if message.get("role") == "assistant":
            for call in _dicts(message.get("tool_calls")):
                if isinstance(call.get("id"), str):
                    function = call.get("function")
                    calls[call["id"]] = function if isinstance(function, Mapping) else {}
    answered = tuple(
        AnsweredCall(
            call_id=message["tool_call_id"],
            name=_str(calls.get(message["tool_call_id"], {}).get("name")),
            is_error=False,
            position=position,
            arguments=calls.get(message["tool_call_id"], {}).get("arguments"),
            result=message.get("content"),
        )
        for position, message in reversed(trailing)
        if isinstance(message.get("tool_call_id"), str)
    )
    return TurnFacts(opens_turn=False, answered=answered)


def _messages(messages: list[Any]) -> TurnFacts:
    items = _dicts(messages)
    # A mid-conversation system message is the harness's context, not the user's input, so the input is under it.
    last = len(items) - 1
    while last >= 0 and items[last].get("role") == "system":
        last -= 1
    if last < 0 or items[last].get("role") != "user":
        return TurnFacts(opens_turn=False)
    content = items[last].get("content")
    if isinstance(content, str):
        return TurnFacts(opens_turn=True, prompt=content)
    blocks = _dicts(content)
    results = [block for block in blocks if block.get("type") == "tool_result"]
    if not results:
        opens_turn = any(block.get("type") == "text" for block in blocks)
        return TurnFacts(opens_turn=opens_turn, prompt=content if opens_turn else None)
    uses: dict[str, Mapping[str, Any]] = {}
    for message in items:
        if message.get("role") == "assistant":
            for block in _dicts(message.get("content")):
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    uses[block["id"]] = block
    answered = tuple(
        AnsweredCall(
            call_id=block["tool_use_id"],
            name=_str(uses.get(block["tool_use_id"], {}).get("name")),
            is_error=bool(block.get("is_error")),
            position=last,
            arguments=uses.get(block["tool_use_id"], {}).get("input"),
            result=block.get("content"),
        )
        for block in results
        if isinstance(block.get("tool_use_id"), str)
    )
    return TurnFacts(opens_turn=False, answered=answered)


_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})


def _responses(input_items: Any) -> TurnFacts:
    if isinstance(input_items, str):
        return TurnFacts(opens_turn=bool(input_items), prompt=input_items)
    if not isinstance(input_items, list):
        return TurnFacts(opens_turn=False)
    items = _dicts(input_items)
    if not items:
        return TurnFacts(opens_turn=False)
    last = items[-1]
    if _str(last.get("type", "message")) == "message" and last.get("role") == "user":
        return TurnFacts(opens_turn=True, prompt=last.get("content"))
    trailing: list[tuple[int, Mapping[str, Any]]] = []
    for position in range(len(items) - 1, -1, -1):
        if _str(items[position].get("type")) not in _OUTPUT_TYPES:
            break
        trailing.append((position, items[position]))
    calls = {
        item["call_id"]: item
        for item in items
        if _str(item.get("type")) in _CALL_TYPES and isinstance(item.get("call_id"), str)
    }
    answered = tuple(
        AnsweredCall(
            call_id=item["call_id"],
            name=_str(calls.get(item["call_id"], {}).get("name")),
            is_error=False,
            position=position,
            arguments=_call_arguments(calls.get(item["call_id"], {})),
            result=item.get("output"),
        )
        for position, item in reversed(trailing)
        if isinstance(item.get("call_id"), str)
    )
    return TurnFacts(opens_turn=False, answered=answered)


def _call_arguments(call: Mapping[str, Any]) -> Any:
    """A Responses call's arguments: a function call's ``arguments``, a custom tool call's ``input``."""
    return call.get("arguments") if call.get("type") == "function_call" else call.get("input")
