"""Automatic recall at the edges of each wire format: the turn it searches for, and where what it found goes.

Recall searches only a turn the user wrote: a turn that hands back tool results
asks nothing new, so a client's own agent loop is searched once per question
rather than once per step. What it finds is added to that turn, at the end of
the prompt, so the prefix a provider caches is the same with recall or without.

The user's turns before it go with it: a follow-up ("what did we decide?")
often names only what the conversation is about, which an earlier turn set.
"""

from collections.abc import Iterable
from typing import Any

from gateway.api.routes._helpers import text_from_content

_TOOL_RESULT_PARTS = frozenset({"tool_result", "function_call_output"})
# User turns before the newest that recall searches with it, the latest ones.
RECALL_EARLIER_TURNS = 2


def _asks(content: Any) -> str | None:
    """The text of a user turn's content, or None when it hands back tool results or holds no text."""
    if isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") in _TOOL_RESULT_PARTS for part in content
    ):
        return None
    return text_from_content(content).strip() or None


def _earlier(contents: Iterable[Any]) -> str | None:
    """The latest of the user turns' contents that ask something, oldest first, as one text."""
    asked = [text for content in contents if (text := _asks(content)) is not None]
    return "\n\n".join(asked[-RECALL_EARLIER_TURNS:]) or None


def _with_text(content: Any, text: str, part_type: str) -> Any:
    if isinstance(content, str):
        return f"{content}\n\n{text}"
    if isinstance(content, list):
        return [*content, {"type": part_type, "text": text}]
    return content


def recall_query(messages: Any) -> str | None:
    """The newest turn's text, in a chat or Anthropic messages list, when the user wrote it."""
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    return _asks(last.get("content"))


def recall_context(messages: Any) -> str | None:
    """The user's turns before the newest in a chat or Anthropic messages list, where recall searches the newest."""
    if recall_query(messages) is None:
        return None
    return _earlier(
        message.get("content")
        for message in messages[:-1]
        if isinstance(message, dict) and message.get("role") == "user"
    )


def with_recall(messages: list[Any], recall: str | None) -> list[Any]:
    """``messages`` with ``recall`` added to the newest turn, which :func:`recall_query` searched for."""
    if recall is None or recall_query(messages) is None:
        return messages
    last = messages[-1]
    return [*messages[:-1], {**last, "content": _with_text(last.get("content"), recall, "text")}]


def _is_user_message(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") == "user"


def _last_input_message(items: list[Any]) -> dict[str, Any] | None:
    last = items[-1] if items else None
    return last if _is_user_message(last) else None


def responses_recall_query(input_value: Any) -> str | None:
    """The newest turn's text, in a Responses ``input``, when the user wrote it."""
    if isinstance(input_value, str):
        return input_value.strip() or None
    if not isinstance(input_value, list):
        return None
    last = _last_input_message(input_value)
    return _asks(last.get("content")) if last is not None else None


def responses_recall_context(input_value: Any) -> str | None:
    """The user's turns before the newest in a Responses ``input``, where recall searches the newest."""
    if not isinstance(input_value, list) or responses_recall_query(input_value) is None:
        return None
    return _earlier(item.get("content") for item in input_value[:-1] if _is_user_message(item))


def responses_with_recall(input_value: Any, recall: str | None) -> Any:
    """A Responses ``input`` with ``recall`` added to its newest turn."""
    if recall is None or responses_recall_query(input_value) is None:
        return input_value
    if isinstance(input_value, str):
        return f"{input_value}\n\n{recall}"
    last = _last_input_message(input_value)
    assert last is not None  # responses_recall_query found it
    return [*input_value[:-1], {**last, "content": _with_text(last.get("content"), recall, "input_text")}]
