"""Regression tests: streaming tool loops close the upstream stream.

The Anthropic messages and OpenAI Responses streaming loops stop consuming
the upstream event stream at their terminal event (``message_stop`` /
``response.completed``). Before the shared engine in
:mod:`gateway.services._tool_loop`, that early exit left the async stream
(and any underlying connection) to garbage collection; the engine now closes
it deterministically, both when the loop exits and between iterations.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from any_llm.types.messages import (
    ContentBlockDeltaEvent as MsgContentBlockDeltaEvent,
)
from any_llm.types.messages import (
    ContentBlockStartEvent as MsgContentBlockStartEvent,
)
from any_llm.types.messages import (
    InputJSONDelta,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageStopEvent,
    MessageStreamEvent,
    TextBlock,
    ToolUseBlock,
)
from any_llm.types.responses import Response
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.services import mcp_loop_messages as messages_loop_module
from gateway.services import mcp_loop_responses as responses_loop_module
from gateway.services.mcp_loop_messages import anthropic_tool_loop_stream
from gateway.services.mcp_loop_responses import responses_tool_loop_stream


class _ClosableStream:
    """Async iterator that records whether ``aclose`` was called."""

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> "_ClosableStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


class _FakePool:
    def __init__(self, tool_names: list[str], results: dict[str, str] | None = None) -> None:
        self._tool_names = set(tool_names)
        self._results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
            for n in sorted(self._tool_names)
        ]

    def owns_tool(self, name: str) -> bool:
        return name in self._tool_names

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self._results.get(name, f"ran {name}")


def _msg_delta_event(stop_reason: str) -> MessageDeltaEvent:
    return MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, stop_reason), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=1,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )


def _tool_use_block_start(index: int, tool_id: str, name: str) -> MsgContentBlockStartEvent:
    return MsgContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=cast(Any, ToolUseBlock(type="tool_use", id=tool_id, name=name, input={})),
    )


def _input_json_delta(index: int, partial: str) -> MsgContentBlockDeltaEvent:
    return MsgContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=cast(Any, InputJSONDelta(type="input_json_delta", partial_json=partial)),
    )


def _response_completed() -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        type="response.completed",
        response=Response(
            id="resp_test",
            created_at=0.0,
            model="fake",
            object="response",
            status="completed",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            usage=ResponseUsage(
                input_tokens=1,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=1,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=2,
            ),
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata=None,
            temperature=None,
            top_p=None,
        ),
        sequence_number=0,
    )


@pytest.mark.asyncio
async def test_messages_stream_closes_upstream_on_terminal_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop breaks out of the stream at ``message_stop``; the stream must be closed."""
    stream = _ClosableStream([_msg_delta_event("end_turn"), MessageStopEvent(type="message_stop")])

    async def fake_amessages(**kwargs: Any) -> _ClosableStream:
        return stream

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    events: list[MessageStreamEvent] = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
            pool=cast(Any, _FakePool(tool_names=[])),
            max_iterations=3,
        )
    ]
    assert [getattr(e, "type", None) for e in events] == ["message_delta", "message_stop"]
    assert stream.closed


@pytest.mark.asyncio
async def test_messages_stream_closes_upstream_between_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-loop continuation must also close the iteration's upstream stream."""
    first = _ClosableStream(
        [
            _tool_use_block_start(0, "tu_1", "fetch_url"),
            _input_json_delta(0, "{}"),
            _msg_delta_event("tool_use"),
            MessageStopEvent(type="message_stop"),
        ]
    )
    second = _ClosableStream([_msg_delta_event("end_turn"), MessageStopEvent(type="message_stop")])
    streams = iter([first, second])

    async def fake_amessages(**kwargs: Any) -> _ClosableStream:
        return next(streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    async for _event in anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 10},
        pool=cast(Any, pool),
        max_iterations=3,
    ):
        pass
    assert pool.calls == [("fetch_url", {})]
    assert first.closed
    assert second.closed


@pytest.mark.asyncio
async def test_responses_stream_closes_upstream_on_terminal_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop breaks out of the stream at ``response.completed``; the stream must be closed."""
    stream = _ClosableStream([_response_completed()])

    async def fake_aresponses(**kwargs: Any) -> _ClosableStream:
        return stream

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    types = [
        event.type
        async for event in responses_tool_loop_stream(
            completion_kwargs={"model": "fake", "input_data": "hi"},
            pool=cast(Any, _FakePool(tool_names=[])),
            max_iterations=3,
        )
    ]
    assert types == ["response.completed"]
    assert stream.closed


def _text_block_start(index: int) -> MsgContentBlockStartEvent:
    return MsgContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=cast(Any, TextBlock(type="text", text="", citations=None)),
    )


def _responses_text_delta(delta: str) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id="msg_1",
        output_index=0,
        content_index=0,
        delta=delta,
        sequence_number=0,
        logprobs=[],
    )


@pytest.mark.asyncio
async def test_messages_stream_downstream_close_propagates_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the tool-loop generator mid-stream must close the upstream stream too."""
    stream = _ClosableStream([_text_block_start(0), _text_block_start(1)])

    async def fake_amessages(**kwargs: Any) -> _ClosableStream:
        return stream

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    agen = anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        pool=cast(Any, _FakePool(tool_names=[])),
        max_iterations=3,
    )
    first = await anext(agen)
    assert getattr(first, "type", None) == "content_block_start"
    await agen.aclose()
    assert stream.closed


@pytest.mark.asyncio
async def test_responses_stream_downstream_close_propagates_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the tool-loop generator mid-stream must close the upstream stream too."""
    stream = _ClosableStream([_responses_text_delta("hi"), _responses_text_delta(" there")])

    async def fake_aresponses(**kwargs: Any) -> _ClosableStream:
        return stream

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    agen = responses_tool_loop_stream(
        completion_kwargs={"model": "fake", "input_data": "hi"},
        pool=cast(Any, _FakePool(tool_names=[])),
        max_iterations=3,
    )
    first = await anext(agen)
    assert getattr(first, "type", None) == "response.output_text.delta"
    await agen.aclose()
    assert stream.closed
