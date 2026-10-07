"""The content a request carries, read from its own body, for a workspace that captures it.

Everything comes from the request the gateway is already serving, so nothing is
fetched and no response stream is parsed:

- **input**: the messages the client added since the model last spoke (the new
  prompt, or the tool results it hands back);
- **prior output**: the model's last message as the client replays it, which is
  the output of the previous step in the session;
- **tool calls**: for each tool result this request carries, the arguments the
  model asked with (from the replayed assistant message) and the result.

Each field is capped, so one large file read cannot become one large row. The
body is the client's, so a malformed field reads as absent, never as an error.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gateway.services.traces._turns import Dialect

MAX_FIELD_CHARS = 16_000


def cap(text: str) -> str:
    """Bound one captured field, saying so when it was cut."""
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "\n[truncated]"


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _text(content: Any) -> str:
    """The readable text of a message's content, whatever shape the dialect gives it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    parts.append(_text(block.get("content")))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool call {block.get('name')}] {_render(block.get('input'))}")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return "" if content is None else _render(content)


@dataclass(frozen=True)
class ToolIO:
    arguments: str = field(repr=False)
    result: str = field(repr=False)


@dataclass(frozen=True)
class RequestContent:
    """A request's captured content. Never printed: a repr shows the field names only."""

    input: str = field(default="", repr=False)
    prior_output: str = field(default="", repr=False)
    tool_calls: Mapping[str, ToolIO] = field(default_factory=dict, repr=False)


def _dicts(items: Any) -> list[Mapping[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _after_last(
    items: list[Mapping[str, Any]], is_model: Any
) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
    """The model's last item, and everything the client sent after it."""
    for index in range(len(items) - 1, -1, -1):
        if is_model(items[index]):
            return items[index], items[index + 1 :]
    return None, items


def _function_arguments(call: Mapping[str, Any]) -> Any:
    function = call.get("function")
    return function.get("arguments") if isinstance(function, Mapping) else None


def read_content(dialect: Dialect, input_items: Any) -> RequestContent:
    if dialect == "responses":
        return _responses(input_items)
    if not isinstance(input_items, list):
        return RequestContent()
    items = _dicts(input_items)
    last, new = _after_last(items, lambda message: message.get("role") == "assistant")
    tool_calls: dict[str, ToolIO] = {}
    if dialect == "chat":
        arguments = {
            call["id"]: _render(_function_arguments(call))
            for call in _dicts((last or {}).get("tool_calls"))
            if isinstance(call.get("id"), str)
        }
        for message in new:
            if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
                call_id = message["tool_call_id"]
                tool_calls[call_id] = ToolIO(cap(arguments.get(call_id, "")), cap(_text(message.get("content"))))
        prior = _text((last or {}).get("content"))
        if last and last.get("tool_calls"):
            prior = "\n".join(filter(None, [prior, *(f"[tool call] {arguments[key]}" for key in arguments)]))
    else:
        uses = {
            block["id"]: _render(block.get("input"))
            for block in _dicts((last or {}).get("content"))
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str)
        }
        for message in new:
            for block in _dicts(message.get("content")):
                if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                    call_id = block["tool_use_id"]
                    tool_calls[call_id] = ToolIO(cap(uses.get(call_id, "")), cap(_text(block.get("content"))))
        prior = _text((last or {}).get("content"))
    return RequestContent(
        input=cap("\n\n".join(_text(message.get("content")) for message in new)),
        prior_output=cap(prior),
        tool_calls=tool_calls,
    )


_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})


def _type(item: Mapping[str, Any], default: str | None = None) -> str | None:
    value = item.get("type", default)
    return value if isinstance(value, str) else None


_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})


def _responses(input_items: Any) -> RequestContent:
    if isinstance(input_items, str):
        return RequestContent(input=cap(input_items))
    if not isinstance(input_items, list):
        return RequestContent()
    items = _dicts(input_items)
    last, new = _after_last(items, lambda item: item.get("role") == "assistant" or _type(item) in _CALL_TYPES)
    arguments = {
        item["call_id"]: _render(item.get("arguments") or item.get("input"))
        for item in items
        if _type(item) in _CALL_TYPES and isinstance(item.get("call_id"), str)
    }
    tool_calls = {
        item["call_id"]: ToolIO(cap(arguments.get(item["call_id"], "")), cap(_render(item.get("output"))))
        for item in new
        if _type(item) in _OUTPUT_TYPES and isinstance(item.get("call_id"), str)
    }
    return RequestContent(
        input=cap("\n\n".join(_text(item.get("content")) for item in new if _type(item, "message") == "message")),
        prior_output=cap(_text((last or {}).get("content"))),
        tool_calls=tool_calls,
    )
