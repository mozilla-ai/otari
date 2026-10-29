"""What a routed request's own body says its step was about, for a session that keeps content.

The counterpart of :mod:`._turns`, which reads only ids and flags: this reads the
text, and runs only once the request, its deployment and its workspace have all
chosen to keep content. Two things are read:

- the prompt, where the request opens a turn: the text of the user's new input;
- for each tool call the request answers, the call's arguments, from the history
  that asked for it, and the result the client hands back.

Nothing else in the body is kept: not the replayed history, the system prompt or
an attachment's bytes. The body is the client's, so a malformed field reads as
absent, never as an error.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gateway.log_config import logger
from gateway.services.traces._turns import AnsweredCall, Dialect, read_turn

_TEXT_PARTS = frozenset({"text", "input_text", "output_text"})
_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})


@dataclass(frozen=True)
class ToolContent:
    """One answered tool call's arguments and result, as text; None where the body held none."""

    arguments: str | None
    result: str | None


@dataclass(frozen=True)
class TurnContent:
    """A request's prompt, and each answered call's content by ``(call id, position)``, as ``AnsweredCall`` has them."""

    prompt: str | None = None
    calls: Mapping[tuple[str, int], ToolContent] = field(default_factory=dict)


def text_of(value: Any) -> str | None:
    """The text in a message's content: a string, or the text parts of a list of parts, joined by blank lines."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) and value.get("type", "text") in _TEXT_PARTS and text else None
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        parts = [part for item in value if (part := text_of(item)) is not None]
        return "\n\n".join(parts) or None
    return None


def arguments_of(value: Any) -> str | None:
    """A tool call's arguments as text: a string as sent, anything else as JSON."""
    if isinstance(value, str):
        return value or None
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _dicts(items: Any) -> list[Mapping[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _chat(items: list[Mapping[str, Any]], answered: tuple[AnsweredCall, ...], opens_turn: bool) -> TurnContent:
    prompt = text_of(items[-1].get("content")) if opens_turn and items else None
    arguments: dict[str, str | None] = {}
    for message in items:
        if message.get("role") == "assistant":
            for call in _dicts(message.get("tool_calls")):
                function = call.get("function")
                if isinstance(call.get("id"), str) and isinstance(function, Mapping):
                    arguments[call["id"]] = arguments_of(function.get("arguments"))
    calls = {
        (call.call_id, call.position): ToolContent(
            arguments=arguments.get(call.call_id), result=text_of(items[call.position].get("content"))
        )
        for call in answered
        if call.position < len(items)
    }
    return TurnContent(prompt=prompt, calls=calls)


def _messages(items: list[Mapping[str, Any]], answered: tuple[AnsweredCall, ...], opens_turn: bool) -> TurnContent:
    last = len(items) - 1
    while last >= 0 and items[last].get("role") == "system":
        last -= 1
    prompt = text_of(items[last].get("content")) if opens_turn and last >= 0 else None
    arguments: dict[str, str | None] = {}
    for message in items:
        if message.get("role") == "assistant":
            for block in _dicts(message.get("content")):
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    arguments[block["id"]] = arguments_of(block.get("input"))
    results: dict[str, str | None] = {}
    if last >= 0:
        for block in _dicts(items[last].get("content")):
            if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                results[block["tool_use_id"]] = text_of(block.get("content"))
    calls = {
        (call.call_id, call.position): ToolContent(
            arguments=arguments.get(call.call_id), result=results.get(call.call_id)
        )
        for call in answered
    }
    return TurnContent(prompt=prompt, calls=calls)


def _responses(input_items: Any, answered: tuple[AnsweredCall, ...], opens_turn: bool) -> TurnContent:
    if isinstance(input_items, str):
        return TurnContent(prompt=input_items or None)
    items = _dicts(input_items) if isinstance(input_items, list) else []
    prompt = text_of(items[-1].get("content")) if opens_turn and items else None
    arguments: dict[str, str | None] = {}
    for item in items:
        if item.get("type") in _CALL_TYPES and isinstance(item.get("call_id"), str):
            arguments[item["call_id"]] = arguments_of(
                item.get("arguments") if item.get("type") == "function_call" else item.get("input")
            )
    calls = {
        (call.call_id, call.position): ToolContent(
            arguments=arguments.get(call.call_id), result=text_of(items[call.position].get("output"))
        )
        for call in answered
        if call.position < len(items)
    }
    return TurnContent(prompt=prompt, calls=calls)


def read_turn_content(dialect: Dialect, input_items: Any) -> TurnContent | None:
    """Read a request's messages (Chat, Messages) or input items (Responses). None when it cannot be read."""
    try:
        facts = read_turn(dialect, input_items)
        if dialect == "responses":
            return _responses(input_items, facts.answered, facts.opens_turn)
        # The same objects ``read_turn`` indexed, so an answered call's position finds its result.
        items = _dicts(input_items) if isinstance(input_items, list) else []
        reader = _chat if dialect == "chat" else _messages
        return reader(items, facts.answered, facts.opens_turn)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Request content not read (%s)", type(exc).__name__)
        return None
