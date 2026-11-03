"""What a routed request's own body says its step was about, for a session that keeps content.

The counterpart of :mod:`._turns`, which finds what to read and reads only ids
and flags: this reads the text, and runs only once the request, its deployment
and its workspace have all chosen to keep content. Two things are read:

- the prompt, where the request opens a turn: the text of the user's new input;
- for each tool call the request answers, the call's arguments, from the history
  that asked for it, and, only where tool output is kept, the result the client
  hands back.

Nothing else in the body is kept: not the replayed history, the system prompt or
an attachment's bytes. The body is the client's, so a malformed field reads as
absent, never as an error.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gateway.log_config import logger
from gateway.services.traces._turns import TurnFacts

_TEXT_PARTS = frozenset({"text", "input_text", "output_text"})


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


def read_turn_content(turn: TurnFacts, *, tool_output: bool) -> TurnContent | None:
    """Read what ``turn`` found: its prompt, each answered call's arguments, and its result only with ``tool_output``.

    Only the answered calls are read, so the cost follows the turn rather than the
    length of the history. None when the content cannot be read.
    """
    try:
        return TurnContent(
            prompt=text_of(turn.prompt) if turn.opens_turn else None,
            calls={
                (call.call_id, call.position): ToolContent(
                    arguments=arguments_of(call.arguments), result=text_of(call.result) if tool_output else None
                )
                for call in turn.answered
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Request content not read (%s)", type(exc).__name__)
        return None
