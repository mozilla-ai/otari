"""Synthesize provider-shaped responses for a served composite turn.

When the interpreter yields an action, the gateway returns a response whose
shape is byte-identical to a real provider turn, so the tenant executes the
tool_use exactly as a model-emitted one (docs/tool-compositor-layer-plan.md
sec 5.3). These builders construct the Anthropic-format non-stream message and
the equivalent SSE event sequence; the serving wiring (dispatch hook) returns
them without calling the provider.

Usage is composite-metered (synthetic), not provider usage.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
    ToolUseBlock,
)


def _message_id() -> str:
    return f"msg_comp_{uuid.uuid4().hex}"


def _tool_use_id() -> str:
    return f"toolu_comp_{uuid.uuid4().hex}"


def _usage(input_tokens: int, output_tokens: int) -> MessageUsage:
    return MessageUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def tool_use_response(
    *,
    model: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str | None = None,
    message_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> MessageResponse:
    """Non-stream synthetic response carrying a single tool_use block."""
    return MessageResponse(
        id=message_id or _message_id(),
        type="message",
        role="assistant",
        model=model,
        content=[
            ToolUseBlock(
                type="tool_use",
                id=tool_use_id or _tool_use_id(),
                name=tool_name,
                input=tool_input,
            )
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=_usage(input_tokens, output_tokens),
    )


def terminal_response(
    *,
    model: str,
    text: str,
    message_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> MessageResponse:
    """Non-stream synthetic response carrying a final assistant text (end_turn)."""
    return MessageResponse(
        id=message_id or _message_id(),
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=_usage(input_tokens, output_tokens),
    )


def tool_use_stream_events(
    *,
    model: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str | None = None,
    message_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> list[MessageStreamEvent]:
    """The SSE event sequence equivalent to :func:`tool_use_response`.

    message_start, content_block_start (tool_use), input_json_delta (the args),
    content_block_stop, message_delta (stop_reason=tool_use), message_stop.
    """
    mid = message_id or _message_id()
    tuid = tool_use_id or _tool_use_id()
    start_message = MessageResponse(
        id=mid,
        type="message",
        role="assistant",
        model=model,
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=_usage(input_tokens, 0),
    )
    return [
        MessageStartEvent(type="message_start", message=start_message),
        ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=ToolUseBlock(type="tool_use", id=tuid, name=tool_name, input={}),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json=json.dumps(tool_input)),
        ),
        ContentBlockStopEvent(type="content_block_stop", index=0),
        MessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason="tool_use", stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=output_tokens),
        ),
        MessageStopEvent(type="message_stop"),
    ]
