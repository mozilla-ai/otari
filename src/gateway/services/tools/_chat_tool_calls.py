"""Chat Completions' ``otari_tool_calls``: the gateway's report of the web tool calls it ran (decision 67).

Chat Completions has no vocabulary for a call the server ran, so the gateway adds a
field of its own at the top level of the response and of each streamed chunk, beside
``choices``. A chunk's ``delta`` is not used: the OpenAI Python SDK's stream helper
merges a list there by each item's ``index`` and raises on an item without one, so a
field there would break every app using the helper, whether or not it knows the field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from gateway.schemas.tools import OtariToolCall, OtariToolCallError
from gateway.services.tools._web_tool_failure import CALL_LIMIT_TAG, WebToolFailure

if TYPE_CHECKING:
    from gateway.services.tools._native import NativeCall

OTARI_TOOL_CALLS_FIELD = "otari_tool_calls"


def chat_entry(
    call: NativeCall,
    *,
    id_prefix: str,
    kind: Literal["web_search", "web_fetch"],
    status: Literal["in_progress", "completed", "failed"],
    **fields: Any,
) -> dict[str, Any]:
    """The wire entry for ``call``, with an ID from its token and only the ``fields`` that are not ``None``.

    The same token gives the started and the ended entry of one call the same ID.
    """
    present = {name: value for name, value in fields.items() if value is not None}
    entry = OtariToolCall(id=f"{id_prefix}{call.token}", type=kind, status=status, **present)
    return entry.model_dump(mode="json", exclude_unset=True)


def call_limit_error() -> OtariToolCallError:
    """The ``error`` of a call refused because the request used up its calls, which reached no provider."""
    return OtariToolCallError.model_validate(WebToolFailure.of(None, None, CALL_LIMIT_TAG).as_dict())


def without_otari_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``messages`` with the field taken off every assistant message.

    A client may store the entries it was sent on the assistant message it sends
    back. A provider rejects a field it does not know, and the entries tell the
    model nothing its tool results do not.
    """
    return [
        {key: value for key, value in message.items() if key != OTARI_TOOL_CALLS_FIELD}
        if message.get("role") == "assistant" and OTARI_TOOL_CALLS_FIELD in message
        else message
        for message in messages
    ]
