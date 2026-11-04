"""How much of a model request's input each kind of content accounts for, as character counts.

The gateway has no tokenizer, so a request's composition is kept as weights: the
characters in the system prompt, the tool definitions, the user and assistant
turns, the tool results, and a flat allowance per attachment. Only the integers
are kept. The dashboard scales them to the provider's exact input-token count,
so the total is exact and the split is an estimate.

The body is the client's, so every field is type-checked before use: a malformed
one weighs nothing, and reading never raises into the request path.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gateway.log_config import logger
from gateway.services.traces._turns import Dialect

# An image, audio clip, file or document is billed by its size and format, not by
# its bytes (a base64 payload says nothing about it), so each counts for a flat
# 4,000 characters, about a thousand tokens at the gateway's four characters per token.
FILE_CHARS = 4_000

_FILE_TYPES = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "audio",
        "input_audio",
        "video_url",
        "file",
        "input_file",
        "document",
    }
)
# Keys that name or tie a part together rather than say anything, so they weigh nothing.
_STRUCTURAL_KEYS = frozenset(
    {"type", "id", "role", "tool_use_id", "tool_call_id", "call_id", "cache_control", "status", "index"}
)
# Blobs a provider hands back to be echoed, not text anyone reads: a thinking block's
# signature, a redacted thinking block, a reasoning item's encrypted content. The
# readable `thinking` text of a thinking block still counts.
_OPAQUE_BLOCKS = frozenset({"redacted_thinking"})
_OPAQUE_KEYS = {"thinking": frozenset({"signature"}), "reasoning": frozenset({"encrypted_content"})}
# Deep enough for a tool result carrying a nested API payload; only a guard against
# a pathological body, never a limit a real request reaches.
_MAX_DEPTH = 32

_SYSTEM_ROLES = frozenset({"system", "developer"})
_RESULT_ROLES = frozenset({"tool", "function"})
_RESULT_ITEMS = frozenset({"function_call_output", "custom_tool_call_output"})

# The span attributes the weights are stored under, in the order ``ContextWeights`` declares them.
CONTEXT_ATTRIBUTE_PREFIX = "otari.context."


@dataclass(frozen=True)
class ContextWeights:
    """Characters per kind of input, plus how many tool definitions and messages there were."""

    system_chars: int = 0
    tools_chars: int = 0
    user_chars: int = 0
    assistant_chars: int = 0
    tool_result_chars: int = 0
    file_chars: int = 0
    tool_count: int = 0
    message_count: int = 0

    def attributes(self) -> dict[str, str | int | float | bool]:
        """The weights as span attributes, ``otari.context.<field>``."""
        return {
            f"{CONTEXT_ATTRIBUTE_PREFIX}{name}": value
            for name, value in (
                ("system_chars", self.system_chars),
                ("tools_chars", self.tools_chars),
                ("user_chars", self.user_chars),
                ("assistant_chars", self.assistant_chars),
                ("tool_result_chars", self.tool_result_chars),
                ("file_chars", self.file_chars),
                ("tool_count", self.tool_count),
                ("message_count", self.message_count),
            )
        }


@dataclass
class _Tally:
    system: int = 0
    user: int = 0
    assistant: int = 0
    tool_result: int = 0
    files: int = 0

    def add(self, bucket: str, value: Any) -> None:
        """Weigh ``value`` into ``bucket``, setting aside the attachments inside it."""
        text, files = _weigh(value)
        setattr(self, bucket, getattr(self, bucket) + text)
        self.files += files


def _weigh(value: Any, depth: int = 0) -> tuple[int, int]:
    """The characters of text in a value and the characters allowed for the attachments inside it."""
    if isinstance(value, str):
        return len(value), 0
    if depth >= _MAX_DEPTH:
        return 0, 0
    if isinstance(value, Mapping):
        # A `type` that is not a string (JSON Schema allows a list) is a property name, not a block type.
        kind = value.get("type")
        kind = kind if isinstance(kind, str) else ""
        if kind in _FILE_TYPES:
            return 0, FILE_CHARS
        if kind in _OPAQUE_BLOCKS:
            return 0, 0
        opaque = _OPAQUE_KEYS.get(kind, frozenset())
        text = files = 0
        for key, item in value.items():
            if key in _STRUCTURAL_KEYS or key in opaque:
                continue
            item_text, item_files = _weigh(item, depth + 1)
            text += item_text
            files += item_files
        return text, files
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        text = files = 0
        for item in value:
            item_text, item_files = _weigh(item, depth + 1)
            text += item_text
            files += item_files
        return text, files
    return 0, 0


def _role_bucket(role: Any) -> str:
    if not isinstance(role, str):
        return "user"
    if role in _SYSTEM_ROLES:
        return "system"
    if role == "assistant":
        return "assistant"
    if role in _RESULT_ROLES:
        return "tool_result"
    return "user"


def _chat(tally: _Tally, messages: Sequence[Any]) -> None:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        bucket = _role_bucket(message.get("role"))
        tally.add(bucket, message.get("content"))
        if bucket == "assistant":
            tally.add("assistant", message.get("tool_calls"))


def _messages(tally: _Tally, messages: Sequence[Any]) -> None:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        bucket = _role_bucket(message.get("role"))
        content = message.get("content")
        if bucket != "user" or not isinstance(content, Sequence) or isinstance(content, str):
            tally.add(bucket, content)
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_result":
                tally.add("tool_result", block.get("content"))
            else:
                tally.add("user", block)


def _responses(tally: _Tally, items: Any) -> int:
    if isinstance(items, str):
        tally.add("user", items)
        return 1
    if not isinstance(items, Sequence):
        return 0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type", "message")
        if not isinstance(kind, str):
            kind = ""
        if kind == "message":
            tally.add(_role_bucket(item.get("role")), item.get("content"))
        elif kind in _RESULT_ITEMS:
            tally.add("tool_result", item.get("output"))
        elif kind != "item_reference":
            # A tool call's arguments, a reasoning item, or a built-in tool's own call.
            tally.add("assistant", item)
    return len(items)


def _schema_chars(value: Any, depth: int = 0) -> int:
    """The characters of a tool definition: its names and its text, without JSON's punctuation.

    A schema's keys are property names the model reads, unlike a message's keys, so
    they count; quotes, braces and commas do not, which keeps a definition on the
    same scale as the messages beside it. Walked rather than serialized, since an
    agent's tool list can run to a hundred kilobytes on every request.
    """
    if isinstance(value, str):
        return len(value)
    if isinstance(value, bool | int | float):
        return len(str(value))
    if depth >= _MAX_DEPTH:
        return 0
    if isinstance(value, Mapping):
        return sum(len(str(key)) + _schema_chars(item, depth + 1) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return sum(_schema_chars(item, depth + 1) for item in value)
    return 0


def _definitions(tools: Any) -> tuple[int, int]:
    """The size of the tool definitions and how many there are."""
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes) or not tools:
        return 0, 0
    return _schema_chars(tools), len(tools)


def read_context(dialect: Dialect, messages: Any, *, tools: Any = None, system: Any = None) -> ContextWeights | None:
    """Weigh a request's messages (Chat, Messages) or input items (Responses).

    ``system`` is Anthropic's ``system`` parameter or the Responses ``instructions``;
    a system or developer message inside ``messages`` counts as system too.
    Returns None when the request cannot be read, so the span simply carries no weights.
    """
    try:
        tally = _Tally()
        tally.add("system", system)
        if dialect == "responses":
            count = _responses(tally, messages)
        else:
            items = messages if isinstance(messages, Sequence) and not isinstance(messages, str) else ()
            (_chat if dialect == "chat" else _messages)(tally, items)
            count = len(items)
        tools_chars, tool_count = _definitions(tools)
        return ContextWeights(
            system_chars=tally.system,
            tools_chars=tools_chars,
            user_chars=tally.user,
            assistant_chars=tally.assistant,
            tool_result_chars=tally.tool_result,
            file_chars=tally.files,
            tool_count=tool_count,
            message_count=count,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Request context not weighed (%s)", type(exc).__name__)
        return None


def context_of(read: Callable[[], ContextWeights | None] | None) -> ContextWeights | None:
    """What a route's reader returns, or None when there is none or it failed."""
    if read is None:
        return None
    try:
        return read()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Request context not weighed (%s)", type(exc).__name__)
        return None
