"""Unit tests for the Anthropic-shaped MCP tool-use loop and tool-format helpers.

Mirrors :mod:`tests.unit.test_mcp_loop` semantically: same scenarios, expressed
in Anthropic content-block / streaming-event shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
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
    TextDelta,
    ToolUseBlock,
)

from gateway.services import mcp_loop_messages as messages_loop_module
from gateway.services.mcp_loop_messages import (
    MaxToolIterationsExceeded,
    anthropic_tool_loop,
    anthropic_tool_loop_stream,
)
from gateway.services.tool_format import (
    inject_purpose_hints_anthropic,
    openai_to_anthropic_tools,
)


class _FakePool:
    """Stand-in for MCPClientPool that satisfies the loop's protocol.

    Duck-types the same surface as :class:`tests.unit.test_mcp_loop._FakePool`
    so the same scenarios apply.
    """

    def __init__(
        self,
        tool_names: list[str],
        purpose_hints: list[tuple[str, str]] | None = None,
        results: dict[str, str] | None = None,
    ):
        self._tool_names = set(tool_names)
        self._hints = purpose_hints or []
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
        return list(self._hints)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name not in self._results:
            return f"ran {name}"
        return self._results[name]


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text, citations=None)


def _tool_use(id: str, name: str, input: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input)


def _message_response(
    *,
    stop_reason: str,
    content: list[Any] | None = None,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="fake",
        content=content or [],
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


# ---------- pure helpers ----------


def test_openai_to_anthropic_tools_converts_function_shape() -> None:
    out = openai_to_anthropic_tools(
        [{"type": "function", "function": {"name": "fetch", "description": "d", "parameters": {"type": "object"}}}]
    )
    assert out == [{"name": "fetch", "description": "d", "input_schema": {"type": "object"}}]


def test_openai_to_anthropic_tools_supplies_empty_schema_when_parameters_missing() -> None:
    out = openai_to_anthropic_tools([{"type": "function", "function": {"name": "noop"}}])
    assert out == [{"name": "noop", "input_schema": {"type": "object", "properties": {}}}]


def test_openai_to_anthropic_tools_passes_unknown_shapes_through() -> None:
    odd = {"type": "custom", "spec": {"foo": "bar"}}
    out = openai_to_anthropic_tools([odd])
    assert out == [odd]


def test_inject_purpose_hints_anthropic_no_hints_returns_unchanged() -> None:
    kwargs: dict[str, Any] = {"system": "be helpful"}
    out = inject_purpose_hints_anthropic(kwargs, [])
    assert out["system"] == "be helpful"


def test_inject_purpose_hints_anthropic_inserts_when_no_system() -> None:
    kwargs: dict[str, Any] = {}
    out = inject_purpose_hints_anthropic(kwargs, [("calendar", "for scheduling")])
    assert "calendar" in out["system"]
    assert "for scheduling" in out["system"]


def test_inject_purpose_hints_anthropic_prepends_existing_string_system() -> None:
    kwargs: dict[str, Any] = {"system": "be helpful"}
    out = inject_purpose_hints_anthropic(kwargs, [("cal", "use it")])
    assert "cal" in out["system"]
    assert "be helpful" in out["system"]
    assert out["system"].index("cal") < out["system"].index("be helpful")


def test_inject_purpose_hints_anthropic_prepends_existing_list_system() -> None:
    kwargs: dict[str, Any] = {
        "system": [{"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}]
    }
    out = inject_purpose_hints_anthropic(kwargs, [("cal", "use it")])
    assert isinstance(out["system"], list)
    assert out["system"][0]["type"] == "text"
    assert "cal" in out["system"][0]["text"]
    # The original list entry is preserved after the prepended hint block.
    assert out["system"][1]["cache_control"] == {"type": "ephemeral"}


# ---------- non-streaming loop ----------


@pytest.mark.asyncio
async def test_loop_returns_immediately_when_model_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs)
        return _message_response(stop_reason="end_turn", content=[_text_block("hi there")])

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"])
    out = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert isinstance(out.content[0], TextBlock)
    assert out.content[0].text == "hi there"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_loop_executes_owned_tool_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _message_response(
                stop_reason="tool_use",
                content=[_tool_use("tu_1", "fetch_url", {"u": "x"})],
            ),
            _message_response(stop_reason="end_turn", content=[_text_block("fetched: ok")]),
        ]
    )
    captured_messages: list[list[dict[str, Any]]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured_messages.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    out = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "fetch x"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    )

    assert out.stop_reason == "end_turn"
    assert pool.calls == [("fetch_url", {"u": "x"})]
    # second call should have assistant tool_use msg and user tool_result msg appended
    second_msgs = captured_messages[1]
    assert second_msgs[-2]["role"] == "assistant"
    assert any(block.get("type") == "tool_use" for block in second_msgs[-2]["content"])
    assert second_msgs[-1]["role"] == "user"
    assert second_msgs[-1]["content"][0]["type"] == "tool_result"
    assert second_msgs[-1]["content"][0]["tool_use_id"] == "tu_1"
    assert second_msgs[-1]["content"][0]["content"] == "ok"


@pytest.mark.asyncio
async def test_loop_accumulates_usage_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _message_response(
                stop_reason="tool_use",
                content=[_tool_use("tu_1", "fetch_url", {})],
                input_tokens=10,
                output_tokens=2,
            ),
            _message_response(
                stop_reason="end_turn",
                content=[_text_block("done")],
                input_tokens=12,
                output_tokens=3,
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return next(responses)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    out = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
        max_iterations=5,
    )
    assert out.usage is not None
    assert out.usage.input_tokens == 22
    assert out.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_loop_max_iter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _message_response(stop_reason="tool_use", content=[_tool_use("tu", "fetch_url", {})])

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    with pytest.raises(MaxToolIterationsExceeded):
        await anthropic_tool_loop(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
            pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
            max_iterations=2,
        )


@pytest.mark.asyncio
async def test_loop_foreign_tool_returns_to_caller_without_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _message_response(stop_reason="tool_use", content=[_tool_use("tu", "user_tool", {})])

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"])  # doesn't own user_tool
    out = await anthropic_tool_loop(
        completion_kwargs={
            "model": "fake",
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": 100,
            "tools": [{"name": "user_tool", "input_schema": {"type": "object"}}],
        },
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert out.stop_reason == "tool_use"
    assert pool.calls == []
    assert any(getattr(block, "type", None) == "tool_use" for block in out.content)


@pytest.mark.asyncio
async def test_loop_mixed_tools_executes_owned_and_returns_only_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed batch: gateway executes the owned subset and filters it from the
    returned content. The client only sees what it can dispatch itself.
    """

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _message_response(
            stop_reason="tool_use",
            content=[
                _tool_use("owned_id", "fetch_url", {}),
                _tool_use("foreign_id", "user_tool", {}),
            ],
        )

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    out = await anthropic_tool_loop(
        completion_kwargs={
            "model": "fake",
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": 100,
            "tools": [{"name": "user_tool", "input_schema": {"type": "object"}}],
        },
        pool=cast(Any, pool),
        max_iterations=5,
    )
    # Owned tool was executed internally.
    assert pool.calls == [("fetch_url", {})]
    # Returned content only carries the foreign tool_use.
    tool_use_blocks = [b for b in out.content if getattr(b, "type", None) == "tool_use"]
    tool_use_ids = [getattr(b, "id", None) for b in tool_use_blocks]
    assert tool_use_ids == ["foreign_id"]


@pytest.mark.asyncio
async def test_loop_tool_execution_failure_appears_as_tool_result_message(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _message_response(stop_reason="tool_use", content=[_tool_use("tu", "fetch_url", {})]),
            _message_response(stop_reason="end_turn", content=[_text_block("recovered")]),
        ]
    )
    captured: list[list[dict[str, Any]]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    class FailingPool(_FakePool):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            raise RuntimeError("upstream down")

    pool = FailingPool(tool_names=["fetch_url"])
    out = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert isinstance(out.content[0], TextBlock)
    assert out.content[0].text == "recovered"
    tool_result_msg = captured[1][-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert "tool error" in tool_result_msg["content"][0]["content"]
    assert "upstream down" in tool_result_msg["content"][0]["content"]


@pytest.mark.asyncio
async def test_loop_exits_when_stop_reason_isnt_tool_use_even_with_tool_use_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the model emits tool_use blocks but stops with a non-``tool_use`` reason
    (e.g. ``end_turn`` because ``max_tokens`` was hit mid-tool-call), the loop
    must exit rather than try to execute them.
    """

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _message_response(stop_reason="end_turn", content=[_tool_use("tu", "fetch_url", {})])

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"])
    out = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert out.stop_reason == "end_turn"
    assert pool.calls == []  # never executed


# ---------- on_first_response (lock-in callback) ----------


@pytest.mark.asyncio
async def test_loop_fires_on_first_response_after_first_amessages_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``on_first_response`` is invoked exactly once, right after the first
    upstream call returns successfully.
    """
    responses = iter(
        [
            _message_response(stop_reason="tool_use", content=[_tool_use("tu_1", "fetch_url", {})]),
            _message_response(stop_reason="end_turn", content=[_text_block("done")]),
        ]
    )
    fire_order: list[str] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        fire_order.append("amessages")
        return next(responses)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    def _on_first() -> None:
        fire_order.append("on_first_response")

    await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
        max_iterations=5,
        on_first_response=_on_first,
    )
    # Fires after the first amessages but before any tool-loop continuation.
    assert fire_order[0] == "amessages"
    assert fire_order[1] == "on_first_response"
    # Only ever fires once, even across multiple iterations.
    assert fire_order.count("on_first_response") == 1


@pytest.mark.asyncio
async def test_loop_does_not_fire_on_first_response_when_initial_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the first ``amessages`` call raises before returning, the callback
    must not fire — callers depend on that to know whether the attempt locked
    in (and therefore can or can't fall through to a fallback provider).
    """
    fired = False

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    def _on_first() -> None:
        nonlocal fired
        fired = True

    with pytest.raises(RuntimeError, match="upstream 500"):
        await anthropic_tool_loop(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
            pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
            max_iterations=5,
            on_first_response=_on_first,
        )
    assert fired is False


@pytest.mark.asyncio
async def test_loop_is_backward_compatible_without_on_first_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback is optional — callers that don't need lock-in (standalone
    mode) can omit it without changing behavior.
    """

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _message_response(stop_reason="end_turn", content=[_text_block("hi")])

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    out = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
        max_iterations=5,
    )
    assert isinstance(out.content[0], TextBlock)
    assert out.content[0].text == "hi"


# ---------- streaming loop ----------


async def _async_iter(*events: MessageStreamEvent) -> AsyncIterator[MessageStreamEvent]:
    for e in events:
        yield e


def _msg_start_event() -> MessageStartEvent:
    return MessageStartEvent(
        type="message_start",
        message=cast(
            Any,
            _message_response(stop_reason="end_turn", content=[], input_tokens=1, output_tokens=0),
        ),
    )


def _msg_delta_event(stop_reason: str, output_tokens: int = 1) -> MessageDeltaEvent:
    return MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, stop_reason), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )


def _msg_stop_event() -> MessageStopEvent:
    return MessageStopEvent(type="message_stop")


def _text_block_start(index: int, text: str = "") -> ContentBlockStartEvent:
    return ContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=cast(Any, TextBlock(type="text", text=text, citations=None)),
    )


def _text_delta(index: int, text: str) -> ContentBlockDeltaEvent:
    return ContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=cast(Any, TextDelta(type="text_delta", text=text)),
    )


def _content_block_stop(index: int) -> ContentBlockStopEvent:
    return ContentBlockStopEvent(type="content_block_stop", index=index)


def _tool_use_block_start(index: int, tool_id: str, name: str) -> ContentBlockStartEvent:
    return ContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=cast(Any, ToolUseBlock(type="tool_use", id=tool_id, name=name, input={})),
    )


def _input_json_delta(index: int, partial: str) -> ContentBlockDeltaEvent:
    return ContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=cast(Any, InputJSONDelta(type="input_json_delta", partial_json=partial)),
    )


@pytest.mark.asyncio
async def test_stream_passes_text_events_through_and_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return _async_iter(
            _msg_start_event(),
            _text_block_start(0),
            _text_delta(0, "hi"),
            _text_delta(0, " there"),
            _content_block_stop(0),
            _msg_delta_event("end_turn"),
            _msg_stop_event(),
        )

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    types: list[str] = []
    async for event in anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, _FakePool(tool_names=[])),
        max_iterations=3,
    ):
        types.append(event.type)
    # All events forwarded — single-iteration path, terminal events included.
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_stream_runs_owned_tool_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iteration 1 emits a tool_use block; the loop executes it server-side and
    drops the intermediate ``message_delta`` / ``message_stop``. Iteration 2 runs
    and its terminal events ARE forwarded.
    """
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_1", "fetch_url"),
                _input_json_delta(0, '{"u":'),
                _input_json_delta(0, ' "x"}'),
                _content_block_stop(0),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "done"),
                _content_block_stop(0),
                _msg_delta_event("end_turn"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    terminal_types: list[str] = []
    async for event in anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    ):
        if event.type in {"message_delta", "message_stop"}:
            terminal_types.append(event.type)
    # Only the FINAL iteration's terminal events are forwarded. The intermediate
    # ``tool_use`` iteration's terminal events are dropped.
    assert terminal_types == ["message_delta", "message_stop"]
    assert pool.calls == [("fetch_url", {"u": "x"})]


@pytest.mark.asyncio
async def test_stream_forwards_terminal_when_model_emits_foreign_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the model asks for a foreign tool, the loop terminates and the
    terminal events MUST reach the client so it knows to dispatch.
    """
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu", "user_tool"),
                _content_block_stop(0),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"])  # doesn't own user_tool
    terminal_types: list[str] = []
    async for event in anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    ):
        if event.type in {"message_delta", "message_stop"}:
            terminal_types.append(event.type)
    assert terminal_types == ["message_delta", "message_stop"]
    assert pool.calls == []


@pytest.mark.asyncio
async def test_stream_input_json_delta_accumulates_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The streaming loop must concatenate partial_json across multiple
    input_json_delta events before parsing the tool input.
    """
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_1", "fetch_url"),
                _input_json_delta(0, '{"u":'),
                _input_json_delta(0, ' "x",'),
                _input_json_delta(0, ' "n": 3}'),
                _content_block_stop(0),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "done"),
                _content_block_stop(0),
                _msg_delta_event("end_turn"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    async for _event in anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    ):
        pass
    assert pool.calls == [("fetch_url", {"u": "x", "n": 3})]
