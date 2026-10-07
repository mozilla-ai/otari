"""The model's output for one request, read back from the response the gateway sent.

Only for a workspace that keeps everything (``full``), and only from bytes the
middleware already passed to the client: a JSON body, or the SSE events of a
stream. The text is the assistant's own (its message, and the tool calls it
asked for); everything else in the response is left. Malformed or truncated
input reads as whatever text was found before it, never as an error.
"""

import json
from collections.abc import Iterator, Mapping
from typing import Any

from gateway.services.traces._content_reading import cap
from gateway.services.traces._turns import Dialect

# The raw response kept per request. Larger than a field's cap, because SSE
# framing and JSON escaping are most of a stream's bytes.
MAX_OUTPUT_BYTES = 256 * 1024


def dialect_of(endpoint: str) -> Dialect:
    if endpoint.endswith("/messages"):
        return "messages"
    if endpoint.endswith("/responses"):
        return "responses"
    return "chat"


def _events(body: bytes) -> Iterator[Mapping[str, Any]]:
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, Mapping):
            yield event


def _index(event: Mapping[str, Any]) -> int:
    value = event.get("index")
    return value if isinstance(value, int) else 0


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _dicts(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _chat_message(body: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for choice in _dicts(body.get("choices")):
        message = choice.get("message")
        if not isinstance(message, Mapping):
            continue
        parts.append(_str(message.get("content")))
        for call in _dicts(message.get("tool_calls")):
            function = call.get("function")
            if isinstance(function, Mapping):
                parts.append(f"[tool call {_str(function.get('name'))}] {_str(function.get('arguments'))}")
    return "\n".join(part for part in parts if part)


def _chat_stream(body: bytes) -> str:
    text: list[str] = []
    calls: dict[int, list[str]] = {}
    for event in _events(body):
        for choice in _dicts(event.get("choices")):
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            text.append(_str(delta.get("content")))
            for call in _dicts(delta.get("tool_calls")):
                index = _index(call)
                function = call.get("function")
                if isinstance(function, Mapping):
                    parts = calls.setdefault(index, [f"[tool call {_str(function.get('name'))}] "])
                    parts.append(_str(function.get("arguments")))
    return "\n".join(filter(None, ["".join(text), *("".join(parts) for parts in calls.values())]))


def _messages_blocks(blocks: Any) -> str:
    parts: list[str] = []
    for block in _dicts(blocks):
        if block.get("type") == "text":
            parts.append(_str(block.get("text")))
        elif block.get("type") == "tool_use":
            parts.append(f"[tool call {_str(block.get('name'))}] {json.dumps(block.get('input'), default=str)}")
    return "\n".join(part for part in parts if part)


def _messages_stream(body: bytes) -> str:
    blocks: dict[int, list[str]] = {}
    for event in _events(body):
        index = _index(event)
        if event.get("type") == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                blocks[index] = [f"[tool call {_str(block.get('name'))}] "]
            else:
                blocks.setdefault(index, [])
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, Mapping):
                piece = _str(delta.get("text")) or _str(delta.get("partial_json"))
                blocks.setdefault(index, []).append(piece)
    return "\n".join(filter(None, ("".join(parts) for _, parts in sorted(blocks.items()))))


def _responses_output(items: Any) -> str:
    parts: list[str] = []
    for item in _dicts(items):
        if item.get("type") == "message":
            parts.extend(_str(block.get("text")) for block in _dicts(item.get("content")))
        elif item.get("type") in ("function_call", "custom_tool_call"):
            parts.append(f"[tool call {_str(item.get('name'))}] {_str(item.get('arguments') or item.get('input'))}")
    return "\n".join(part for part in parts if part)


def _responses_stream(body: bytes) -> str:
    for event in _events(body):
        # The completed event repeats the whole output, which is the cheap reading.
        response = event.get("response")
        if event.get("type") == "response.completed" and isinstance(response, Mapping):
            return _responses_output(response.get("output"))
    return "".join(
        _str(event.get("delta")) for event in _events(body) if event.get("type") == "response.output_text.delta"
    )


def read_output(dialect: Dialect, body: bytes, *, is_stream: bool) -> str:
    """The assistant's output in one response, capped like every captured field."""
    try:
        if is_stream:
            reader = {"chat": _chat_stream, "messages": _messages_stream, "responses": _responses_stream}[dialect]
            return cap(reader(body))
        parsed = json.loads(body)
        if not isinstance(parsed, Mapping):
            return ""
        if dialect == "chat":
            return cap(_chat_message(parsed))
        if dialect == "messages":
            return cap(_messages_blocks(parsed.get("content")))
        return cap(_responses_output(parsed.get("output")))
    except (ValueError, TypeError, RecursionError):
        return ""
