"""Automatic recall at the edges of each wire format: the turn it searches for, and where what it found goes.

Recall searches only a turn the user wrote: a turn that hands back tool results
asks nothing new, so a client's own agent loop is searched once per question
rather than once per step. What it finds is added to that turn, at the end of
the prompt, so the prefix a provider caches is the same with recall or without.
"""

from typing import Any

from gateway.api.routes._helpers import text_from_content

_TOOL_RESULT_PARTS = frozenset({"tool_result", "function_call_output"})


def _asks(content: Any) -> str | None:
    """The text of a user turn's content, or None when it hands back tool results or holds no text."""
    if isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") in _TOOL_RESULT_PARTS for part in content
    ):
        return None
    return text_from_content(content).strip() or None


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


def with_recall(messages: list[Any], recall: str | None) -> list[Any]:
    """``messages`` with ``recall`` added to the newest turn, which :func:`recall_query` searched for."""
    if recall is None or recall_query(messages) is None:
        return messages
    last = messages[-1]
    return [*messages[:-1], {**last, "content": _with_text(last.get("content"), recall, "text")}]


def _last_input_message(items: list[Any]) -> dict[str, Any] | None:
    last = items[-1] if items else None
    if isinstance(last, dict) and last.get("type", "message") == "message" and last.get("role") == "user":
        return last
    return None


def responses_recall_query(input_value: Any) -> str | None:
    """The newest turn's text, in a Responses ``input``, when the user wrote it."""
    if isinstance(input_value, str):
        return input_value.strip() or None
    if not isinstance(input_value, list):
        return None
    last = _last_input_message(input_value)
    return _asks(last.get("content")) if last is not None else None


def responses_with_recall(input_value: Any, recall: str | None) -> Any:
    """A Responses ``input`` with ``recall`` added to its newest turn."""
    if recall is None or responses_recall_query(input_value) is None:
        return input_value
    if isinstance(input_value, str):
        return f"{input_value}\n\n{recall}"
    last = _last_input_message(input_value)
    assert last is not None  # responses_recall_query found it
    return [*input_value[:-1], {**last, "content": _with_text(last.get("content"), recall, "input_text")}]
