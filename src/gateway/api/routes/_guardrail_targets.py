"""What a request's guardrails read, from each wire format, in one shape.

Each completion endpoint carries the same content in its own words: the
conversation's turns and the results of tools the client ran. Each piece a
guardrail reads is one :class:`RequestTarget`, in conversation order.

A check on input reads the user turns that arrived since the model's last turn,
since older input was checked when it arrived. A check on tool results reads
every result, since a client resends the results it holds and one a guardrail
withheld has to be withheld again wherever it reappears.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from gateway.api.routes._helpers import text_from_content
from gateway.services.guardrails import GuardrailTarget


@dataclass(frozen=True)
class RequestTarget:
    """One piece of a request a guardrail reads.

    ``new`` is content that arrived since the model's last turn, whose verdicts
    are reported. ``withhold`` replaces a tool result whole.
    """

    target: GuardrailTarget
    new: bool
    withhold: Callable[[str], None] | None = None


@dataclass(frozen=True)
class AnswerPart:
    """One piece of the model's answer a guardrail reads: its text, or a tool call it hands the client.

    ``text`` is a tool call's arguments as JSON.
    """

    kind: Literal["text", "tool_call"]
    text: str
    tool: str | None = None


def _setter(container: dict[str, Any], key: str, *, also: dict[str, Any] | None = None) -> Callable[[str], None]:
    def put(text: str) -> None:
        container[key] = text
        container.update(also or {})

    return put


def _last_turn(items: Sequence[Any], is_model_turn: Callable[[dict[str, Any]], bool]) -> int:
    """The index of the model's last turn, or -1. A final assistant message is a caller's prefill, not a turn."""
    turns = (i for i, item in enumerate(items[:-1]) if isinstance(item, dict) and is_model_turn(item))
    return max(turns, default=-1)


class _Collector:
    """The targets of one request."""

    def __init__(self) -> None:
        self.targets: list[RequestTarget] = []

    def input(self, text: str) -> None:
        if text:
            self.targets.append(RequestTarget(GuardrailTarget("before_model", text), new=True))

    def result(self, text: str, tool: str | None, withhold: Callable[[str], None], *, new: bool) -> None:
        if text:
            self.targets.append(
                RequestTarget(GuardrailTarget("after_tool", text, tool=tool), new=new, withhold=withhold)
            )

    def finish(self, last: Any) -> list[RequestTarget]:
        """The targets, with the last message as the input where there is neither a user turn nor a tool result."""
        if not self.targets and isinstance(last, dict):
            self.input(text_from_content(last.get("content")))
        return self.targets


def chat_targets(messages: Sequence[Any]) -> list[RequestTarget]:
    """The targets of a Chat Completions ``messages`` list."""
    last_turn = _last_turn(messages, lambda m: m.get("role") == "assistant")
    names = {
        call.get("id"): call["function"].get("name")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    }
    found = _Collector()
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role, new = message.get("role"), i > last_turn
        if role in ("tool", "function"):
            tool = names.get(message.get("tool_call_id")) if role == "tool" else message.get("name")
            found.result(text_from_content(message.get("content")), tool, _setter(message, "content"), new=new)
        elif role == "user" and new:
            found.input(text_from_content(message.get("content")))
    return found.finish(messages[-1] if messages else None)


def messages_targets(request: Any) -> list[RequestTarget]:
    """The targets of an Anthropic Messages request. A tool result is a block of a user turn."""
    messages: Sequence[Any] = request.messages
    last_turn = _last_turn(messages, lambda m: m.get("role") == "assistant")
    names = {
        block.get("id"): block.get("name")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    }
    found = _Collector()
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        new, content = i > last_turn, message.get("content")
        if not isinstance(content, list):
            if new:
                found.input(text_from_content(content))
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                found.result(
                    text_from_content(block.get("content")),
                    names.get(block.get("tool_use_id")),
                    _setter(block, "content", also={"is_error": True}),
                    new=new,
                )
            elif block.get("type") == "text" and isinstance(block.get("text"), str) and new:
                found.input(block["text"])
    return found.finish(messages[-1] if messages else None)


# Items the model produced. Everything after the last of them is new.
_RESPONSES_MODEL_ITEMS = frozenset({"function_call", "custom_tool_call", "reasoning"})
_RESPONSES_TOOL_OUTPUTS = frozenset({"function_call_output", "custom_tool_call_output"})


def responses_targets(request: Any) -> list[RequestTarget]:
    """The targets of a Responses request's ``input``: a bare string or a list of items."""
    found = _Collector()
    value = request.input
    if isinstance(value, str):
        found.input(value)
        return found.targets
    if not isinstance(value, list):
        return found.targets
    last_turn = _last_turn(
        value, lambda item: item.get("role") == "assistant" or item.get("type") in _RESPONSES_MODEL_ITEMS
    )
    names = {
        item.get("call_id"): item.get("name")
        for item in value
        if isinstance(item, dict) and item.get("type") in ("function_call", "custom_tool_call")
    }
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        new, kind = i > last_turn, item.get("type")
        if kind in _RESPONSES_TOOL_OUTPUTS:
            found.result(
                text_from_content(item.get("output")),
                names.get(item.get("call_id")),
                _setter(item, "output"),
                new=new,
            )
        elif item.get("role") == "user" and kind in (None, "message") and new:
            found.input(text_from_content(item.get("content")))
    return found.finish(value[-1] if value else None)
