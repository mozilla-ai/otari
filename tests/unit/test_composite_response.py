"""Unit tests for synthetic composite response builders."""

from __future__ import annotations

import json

from gateway.services.composite_response import (
    terminal_response,
    tool_use_response,
    tool_use_stream_events,
)


def test_tool_use_response_shape() -> None:
    resp = tool_use_response(
        model="claude-haiku-4-5",
        tool_name="google_sheets_append_row",
        tool_input={"sheet": "Log", "row": ["a", "b"]},
    )
    dumped = resp.model_dump(exclude_none=True)
    assert dumped["type"] == "message"
    assert dumped["role"] == "assistant"
    assert dumped["stop_reason"] == "tool_use"
    assert dumped["model"] == "claude-haiku-4-5"
    assert len(dumped["content"]) == 1
    block = dumped["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "google_sheets_append_row"
    assert block["input"] == {"sheet": "Log", "row": ["a", "b"]}
    assert block["id"].startswith("toolu_comp_")
    assert dumped["usage"]["input_tokens"] == 0
    assert dumped["usage"]["output_tokens"] == 0


def test_terminal_response_shape() -> None:
    resp = terminal_response(model="claude-haiku-4-5", text="all done")
    dumped = resp.model_dump(exclude_none=True)
    assert dumped["stop_reason"] == "end_turn"
    assert dumped["content"][0]["type"] == "text"
    assert dumped["content"][0]["text"] == "all done"


def test_tool_use_stream_event_sequence() -> None:
    events = tool_use_stream_events(
        model="claude-haiku-4-5",
        tool_name="resolve_time",
        tool_input={"tz": "UTC"},
    )
    types = [e.type for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    start = events[0]
    assert start.message.content == []
    assert start.message.role == "assistant"

    block_start = events[1]
    assert block_start.content_block.type == "tool_use"
    assert block_start.content_block.name == "resolve_time"

    delta = events[2]
    assert delta.delta.type == "input_json_delta"
    assert json.loads(delta.delta.partial_json) == {"tz": "UTC"}

    message_delta = events[4]
    assert message_delta.delta.stop_reason == "tool_use"


def test_stream_and_nonstream_agree_on_tool() -> None:
    kwargs = {"model": "m", "tool_name": "resolve_time", "tool_input": {"tz": "UTC"}}
    resp = tool_use_response(**kwargs)
    events = tool_use_stream_events(**kwargs)
    assert resp.content[0].name == events[1].content_block.name
