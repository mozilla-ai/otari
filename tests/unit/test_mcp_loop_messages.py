"""Unit tests for the Anthropic-shaped MCP tool-use loop and tool-format helpers.

Mirrors :mod:`tests.unit.test_mcp_loop` semantically: same scenarios, expressed
in Anthropic content-block / streaming-event shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from any_llm.types.messages import (
    CompactionBlock,
    CompactionDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDeltaEvent,
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
from gateway.services.mcp_client import MCPToolCallOutcome
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


class _ActivityPool(_FakePool):
    """MCP pool stand-in with server metadata and controllable execution."""

    def __init__(self, *, content: str = "ok", is_error: bool = False) -> None:
        super().__init__(["fetch_url"])
        self.content = content
        self.is_error = is_error
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def server_name_for_tool(self, name: str) -> str | None:
        return "fixture-server" if self.owns_tool(name) else None

    async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
        self.calls.append((name, arguments))
        self.started.set()
        await self.release.wait()
        return MCPToolCallOutcome(content=self.content, is_error=self.is_error)


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
    iterations: list[dict[str, Any]] | None = None,
) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="fake",
        content=content or [],
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=MessageUsage.model_validate(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "iterations": iterations,
            }
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
async def test_loop_replays_compaction_block_with_context_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_management = {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}]}
    responses = iter(
        [
            MessageResponse.model_validate(
                {
                    "id": "msg_compaction_tool",
                    "type": "message",
                    "role": "assistant",
                    "model": "fake",
                    "content": [
                        {"type": "compaction", "content": "Conversation summary"},
                        {"type": "tool_use", "id": "tu_1", "name": "fetch_url", "input": {"u": "x"}},
                    ],
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ),
            _message_response(stop_reason="end_turn", content=[_text_block("done")]),
        ]
    )
    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    out = await anthropic_tool_loop(
        completion_kwargs={
            "model": "fake",
            "messages": [{"role": "user", "content": "fetch x"}],
            "max_tokens": 100,
            "context_management": context_management,
            "betas": ["compact-2026-01-12"],
        },
        pool=cast(Any, _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})),
        max_iterations=5,
    )

    assert out.stop_reason == "end_turn"
    assert calls[1]["context_management"] == context_management
    assert calls[1]["betas"] == ["compact-2026-01-12"]
    assistant_content = calls[1]["messages"][-2]["content"]
    assert assistant_content[0] == {"type": "compaction", "content": "Conversation summary"}
    assert assistant_content[1]["type"] == "tool_use"


@pytest.mark.asyncio
async def test_loop_accumulates_usage_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _message_response(
                stop_reason="tool_use",
                content=[_tool_use("tu_1", "fetch_url", {})],
                input_tokens=10,
                output_tokens=2,
                iterations=[
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                    {
                        "type": "message",
                        "model": "fake",
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                ],
            ),
            _message_response(
                stop_reason="end_turn",
                content=[_text_block("done")],
                input_tokens=12,
                output_tokens=3,
                iterations=[
                    {
                        "type": "message",
                        "model": "fake",
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                ],
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
    assert [iteration.type for iteration in out.usage.iterations or []] == [
        "compaction",
        "message",
        "message",
    ]


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


def _msg_delta_event(
    stop_reason: str,
    output_tokens: int = 1,
    *,
    iterations: list[dict[str, Any]] | None = None,
    applied_edits: list[dict[str, Any]] | None = None,
) -> MessageDeltaEvent:
    return MessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": None,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "server_tool_use": None,
                "iterations": iterations,
            },
            "context_management": {"applied_edits": applied_edits} if applied_edits is not None else None,
        }
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


def _compaction_block_start(index: int) -> ContentBlockStartEvent:
    return ContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=cast(Any, CompactionBlock(type="compaction", content=None)),
    )


def _compaction_delta(index: int, content: str) -> ContentBlockDeltaEvent:
    return ContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=cast(Any, CompactionDelta(type="compaction_delta", content=content)),
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

    types = [
        event.type
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
            pool=cast(Any, _FakePool(tool_names=[])),
            max_iterations=3,
        )
    ]
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
    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
            pool=cast(Any, pool),
            max_iterations=5,
        )
    ]

    # The FULL sequence, not just the terminal events: a client accumulating this
    # stream must see one well-formed message. Asserting only the terminal events
    # is what let a duplicated ``message_start`` and a reused block index ship.
    assert [e.type for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # Exactly one envelope, even though the loop consumed two upstream messages.
    assert [e.type for e in events].count("message_start") == 1
    # The gateway's own tool_use block never reaches the client: it could never be
    # sent the matching tool_result, because the loop consumed it.
    assert not any(getattr(getattr(e, "content_block", None), "type", None) == "tool_use" for e in events)
    assert pool.calls == [("fetch_url", {"u": "x"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(("content", "is_error"), [("fixture result", False), ("fixture error", True)])
async def test_stream_emits_live_mcp_activity_around_execution(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    is_error: bool,
) -> None:
    """The MCP start reaches the client before execution, then a paired completion follows."""
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_internal", "fetch_url"),
                _input_json_delta(0, '{"url": "https://example.test"}'),
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
    pool = _ActivityPool(content=content, is_error=is_error)
    stream = anthropic_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
        pool=cast(Any, pool),
        max_iterations=5,
    )

    assert (await anext(stream)).type == "message_start"
    activity_start = await anext(stream)
    assert activity_start.type == "content_block_start"
    use = cast(Any, activity_start).content_block
    assert use.type == "mcp_tool_use"
    assert use.id.startswith("otari_mcptoolu_")
    assert use.name == "fetch_url"
    assert use.server_name == "fixture-server"
    assert use.input == {"url": "https://example.test"}
    assert not pool.started.is_set(), "execution must not precede the client-visible start"

    assert (await anext(stream)).type == "content_block_stop"
    pending_completion = asyncio.create_task(anext(stream))
    await asyncio.wait_for(pool.started.wait(), timeout=1)
    assert not pending_completion.done(), "the completion must wait for the MCP call"
    pool.release.set()

    completion_start = await pending_completion
    assert completion_start.type == "content_block_start"
    result = cast(Any, completion_start).content_block
    assert result.type == "mcp_tool_result"
    assert result.tool_use_id == use.id
    assert result.content == content
    assert result.is_error is is_error
    assert cast(Any, completion_start).index == cast(Any, activity_start).index + 1

    remaining = [event async for event in stream]
    assert [event.type for event in remaining] == [
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert cast(Any, remaining[1]).index == cast(Any, completion_start).index + 1
    assert pool.calls == [("fetch_url", {"url": "https://example.test"})]


@pytest.mark.asyncio
async def test_stream_mcp_exception_emits_error_without_logging_detail(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_internal", "fetch_url"),
                _input_json_delta(0, '{"secret_input": "do-not-log"}'),
                _content_block_stop(0),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "recovered"),
                _content_block_stop(0),
                _msg_delta_event("end_turn"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    class FailingActivityPool(_ActivityPool):
        async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
            self.calls.append((name, arguments))
            raise RuntimeError("credential-detail-do-not-log")

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)
    pool = FailingActivityPool()
    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
            pool=cast(Any, pool),
            max_iterations=5,
        )
    ]

    result = next(
        cast(Any, event).content_block
        for event in events
        if event.type == "content_block_start"
        and getattr(cast(Any, event).content_block, "type", None) == "mcp_tool_result"
    )
    assert result.is_error is True
    assert result.content == "[tool error] MCP tool execution failed"
    assert "credential-detail-do-not-log" not in caplog.text
    assert "do-not-log" not in caplog.text


@pytest.mark.asyncio
async def test_stream_replays_compaction_content_when_tool_loop_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_management = {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}]}
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _compaction_block_start(0),
                _compaction_delta(0, "Conversation "),
                _compaction_delta(0, "summary"),
                _content_block_stop(0),
                _tool_use_block_start(1, "tu_1", "fetch_url"),
                _input_json_delta(1, '{"u": "x"}'),
                _content_block_stop(1),
                _msg_delta_event(
                    "tool_use",
                    output_tokens=3,
                    iterations=[
                        {
                            "type": "compaction",
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        }
                    ],
                    applied_edits=[
                        {
                            "type": "clear_tool_uses_20250919",
                            "cleared_input_tokens": 42,
                            "cleared_tool_uses": 2,
                        }
                    ],
                ),
                _msg_stop_event(),
            ),
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "done"),
                _content_block_stop(0),
                _msg_delta_event(
                    "end_turn",
                    output_tokens=5,
                    iterations=[
                        {
                            "type": "message",
                            "model": "fake",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        }
                    ],
                    applied_edits=[
                        {
                            "type": "clear_thinking_20251015",
                            "cleared_input_tokens": 21,
                            "cleared_thinking_turns": 1,
                        }
                    ],
                ),
                _msg_stop_event(),
            ),
        ]
    )
    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        calls.append(kwargs)
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={
                "model": "fake",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "context_management": context_management,
                "betas": ["compact-2026-01-12"],
            },
            pool=cast(Any, _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})),
            max_iterations=5,
        )
    ]

    assistant_content = calls[1]["messages"][-2]["content"]
    assert assistant_content[0] == {"type": "compaction", "content": "Conversation summary"}
    assert assistant_content[1]["type"] == "tool_use"
    assert calls[1]["context_management"] == context_management
    assert calls[1]["betas"] == ["compact-2026-01-12"]

    final_delta = next(event for event in events if event.type == "message_delta")
    assert final_delta.usage.output_tokens == 8
    assert [iteration.type for iteration in final_delta.usage.iterations or []] == ["compaction", "message"]
    assert final_delta.context_management is not None
    assert [edit.type for edit in final_delta.context_management.applied_edits] == [
        "clear_tool_uses_20250919",
        "clear_thinking_20251015",
    ]


@pytest.mark.asyncio
async def test_stream_renumbers_blocks_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content-block indices are continuous across a tool-loop round trip.

    The model speaks before it calls the tool ("let me look that up"), so
    iteration 1 forwards a text block at upstream index 0 and iteration 2 opens
    its own text block, also at upstream index 0. The client sees one message, so
    the second block has to arrive as index 1; reusing 0 would make an accumulator
    overwrite the first block.
    """
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "let me look"),
                _content_block_stop(0),
                _tool_use_block_start(1, "tu_1", "fetch_url"),
                _input_json_delta(1, '{"u": "x"}'),
                _content_block_stop(1),
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
    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
            pool=cast(Any, pool),
            max_iterations=5,
        )
    ]

    # The stream event type is a union; only the content_block_* members carry an
    # index, so read it positionally rather than narrowing 6 variants per element.
    indexed = [(e.type, getattr(e, "index")) for e in events if getattr(e, "index", None) is not None]
    assert indexed == [
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
        # Iteration 2's block arrives at 1, not at its upstream index of 0.
        ("content_block_start", 1),
        ("content_block_delta", 1),
        ("content_block_stop", 1),
    ]
    assert [e.type for e in events].count("message_start") == 1
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
    terminal_types = [
        event.type
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
            pool=cast(Any, pool),
            max_iterations=5,
        )
        if event.type in {"message_delta", "message_stop"}
    ]
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


@pytest.mark.asyncio
async def test_stream_mixed_batch_hides_and_still_runs_the_gateway_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed batch shows only the caller's tool, and still runs the gateway's.

    The loop exits so the caller can dispatch its own tool. The gateway's ordinary
    tool block is withheld, but the client receives the server-owned MCP activity
    pair while Otari executes it.
    """
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_owned", "fetch_url"),
                _input_json_delta(0, '{"u": "x"}'),
                _content_block_stop(0),
                _tool_use_block_start(1, "tu_foreign", "user_tool"),
                _input_json_delta(1, "{}"),
                _content_block_stop(1),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    pool = _ActivityPool(content="ok")
    pool.release.set()
    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 100},
            pool=cast(Any, pool),
            max_iterations=5,
        )
    ]

    shown = [
        getattr(getattr(e, "content_block", None), "name", None)
        for e in events
        if getattr(getattr(e, "content_block", None), "type", None) == "tool_use"
    ]
    assert shown == ["user_tool"]
    # Renumbered so the caller's block is index 0, with no hole where the hidden
    # raw call was. Server-owned MCP activity follows at indices 1 and 2.
    starts = [getattr(e, "index") for e in events if e.type == "content_block_start"]
    assert starts == [0, 1, 2]
    activity_types = [
        getattr(getattr(e, "content_block", None), "type", None) for e in events if e.type == "content_block_start"
    ]
    assert activity_types == ["tool_use", "mcp_tool_use", "mcp_tool_result"]
    assert pool.calls == [("fetch_url", {"u": "x"})]


# ---------- native web-search blocks ----------
#
# A caller that declared web search in Anthropic's native vocabulary gets
# ``server_tool_use`` / ``web_search_tool_result`` blocks describing the searches
# the gateway ran, so a citations panel has something to render. Every other
# caller keeps the historical behavior (the gateway's calls stay invisible).


class _FakeSearchPool(_FakePool):
    """A pool that owns ``web_search`` and exposes structured hits like the real backend."""

    def __init__(
        self,
        *,
        results: list[dict[str, Any]] | None = None,
        fail: bool = False,
    ) -> None:
        super().__init__(tool_names=["web_search"], results={"web_search": "[1] Result\nhttps://a"})
        self._structured = results if results is not None else [{"url": "https://a", "title": "A"}]
        self._fail = fail
        self._taken = False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._fail:
            self.calls.append((name, arguments))
            raise RuntimeError("backend down")
        return await super().call_tool(name, arguments)

    def take_last_results(self) -> list[dict[str, Any]]:
        self._taken = True
        return list(self._structured)


def _search_use(block_id: str = "tu_1", query: str = "python release") -> ToolUseBlock:
    return _tool_use(block_id, "web_search", {"query": query})


def _two_round_responses(query: str = "python release") -> list[MessageResponse]:
    return [
        _message_response(stop_reason="tool_use", content=[_search_use(query=query)]),
        _message_response(stop_reason="end_turn", content=[_text_block("Python 3.14.")]),
    ]


def _fake_amessages_for(responses: list[MessageResponse]) -> Any:
    it = iter(responses)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return next(it)

    return fake_amessages


@pytest.mark.asyncio
async def test_native_blocks_prepended_to_final_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for(_two_round_responses()))
    pool = _FakeSearchPool(results=[{"url": "https://python.org", "title": "Python", "published_date": "2026-01-02"}])

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
        emit_native_web_search=True,
    )

    # The pair comes before the model's answer, because the search happened first.
    assert [b.type for b in result.content] == ["server_tool_use", "web_search_tool_result", "text"]
    server_use, tool_result, _text = (cast(Any, block) for block in result.content)
    assert server_use.name == "web_search"
    assert server_use.input == {"query": "python release"}
    # The result block is paired to its server_tool_use by id, as a client expects.
    assert tool_result.tool_use_id == server_use.id
    assert server_use.id.startswith("srvtoolu_")
    citation = tool_result.content[0]
    assert citation.url == "https://python.org"
    assert citation.title == "Python"
    assert citation.page_age == "2026-01-02"
    # Empty rather than forged: only Anthropic can sign this blob.
    assert citation.encrypted_content == ""


@pytest.mark.asyncio
async def test_no_native_blocks_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The historical shape: the gateway's search is invisible in the response."""
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for(_two_round_responses()))
    pool = _FakeSearchPool()

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
    )

    assert [b.type for b in result.content] == ["text"]


@pytest.mark.asyncio
async def test_failed_search_contributes_no_native_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search that never returned results has nothing to cite."""
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for(_two_round_responses()))
    pool = _FakeSearchPool(fail=True)

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
        emit_native_web_search=True,
    )

    assert [b.type for b in result.content] == ["text"]


@pytest.mark.asyncio
async def test_hits_without_a_url_are_dropped_from_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for(_two_round_responses()))
    pool = _FakeSearchPool(results=[{"title": "no url"}, {"url": "https://ok", "title": "ok"}])

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
        emit_native_web_search=True,
    )

    citations = cast(Any, result.content[1]).content
    assert [c.url for c in citations] == ["https://ok"]


@pytest.mark.asyncio
async def test_non_web_search_tool_gets_no_native_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MCP or sandbox call has no Anthropic block that would be honest to emit."""
    responses = [
        _message_response(stop_reason="tool_use", content=[_tool_use("tu_1", "fetch_url", {"u": "x"})]),
        _message_response(stop_reason="end_turn", content=[_text_block("done")]),
    ]
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for(responses))
    pool = _FakePool(tool_names=["fetch_url"])

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
        emit_native_web_search=True,
    )

    assert [b.type for b in result.content] == ["text"]


@pytest.mark.asyncio
async def test_native_blocks_for_each_of_several_searches(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _message_response(stop_reason="tool_use", content=[_search_use("tu_1", "first")]),
        _message_response(stop_reason="tool_use", content=[_search_use("tu_2", "second")]),
        _message_response(stop_reason="end_turn", content=[_text_block("both")]),
    ]
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for(responses))

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, _FakeSearchPool()),
        max_iterations=5,
        emit_native_web_search=True,
    )

    assert [b.type for b in result.content] == [
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert [b.input["query"] for b in result.content if b.type == "server_tool_use"] == ["first", "second"]
    # Each pair is independently addressable.
    ids = [b.id for b in result.content if b.type == "server_tool_use"]
    assert len(set(ids)) == 2


@pytest.mark.asyncio
async def test_stream_emits_native_blocks_with_gapless_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthetic blocks take the swallowed tool_use events' place on the wire.

    Indices must continue the client-visible sequence: an SDK accumulator indexes
    its snapshot array by the index it is handed.
    """
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_1", "web_search"),
                _input_json_delta(0, '{"query": "python"}'),
                _content_block_stop(0),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "Python 3.14."),
                _content_block_stop(0),
                _msg_delta_event("end_turn"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)
    pool = _FakeSearchPool(results=[{"url": "https://python.org", "title": "Python"}])

    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
            pool=cast(Any, pool),
            max_iterations=5,
            emit_native_web_search=True,
        )
    ]

    assert [e.type for e in events] == [
        "message_start",
        # The synthetic pair, in place of the tool_use events the loop swallowed.
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        # Then the second iteration's real text block.
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    starts = [e for e in events if e.type == "content_block_start"]
    assert [e.content_block.type for e in starts] == ["server_tool_use", "web_search_tool_result", "text"]
    # Gapless and in order, so an accumulator can index straight into its array.
    assert [e.index for e in events if e.type == "content_block_start"] == [0, 1, 2]
    assert [e.index for e in events if e.type == "content_block_stop"] == [0, 1, 2]
    # The query survives without any input_json_delta: the start event carries the
    # complete block, and the SDK only overwrites ``input`` when a delta arrives.
    assert cast(Any, starts[0].content_block).input == {"query": "python"}
    # The gateway's own tool_use block still never reaches the client.
    assert not any(getattr(getattr(e, "content_block", None), "type", None) == "tool_use" for e in events)


@pytest.mark.asyncio
async def test_stream_emits_no_native_blocks_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    iter_streams = iter(
        [
            _async_iter(
                _msg_start_event(),
                _tool_use_block_start(0, "tu_1", "web_search"),
                _input_json_delta(0, '{"query": "python"}'),
                _content_block_stop(0),
                _msg_delta_event("tool_use"),
                _msg_stop_event(),
            ),
            _async_iter(
                _msg_start_event(),
                _text_block_start(0),
                _text_delta(0, "Python 3.14."),
                _content_block_stop(0),
                _msg_delta_event("end_turn"),
                _msg_stop_event(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)

    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
            pool=cast(Any, _FakeSearchPool()),
            max_iterations=5,
        )
    ]

    assert [e.type for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_mixed_batch_still_emits_native_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search alongside a caller's tool still gets its pair.

    The round exits so the caller can dispatch its own tool, but the gateway did run
    the search, so a native client is owed the blocks describing it.
    """
    mixed = _message_response(
        stop_reason="tool_use",
        content=[_search_use("tu_1", "python"), _tool_use("tu_2", "get_weather", {"city": "Lisbon"})],
    )
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for([mixed]))
    pool = _FakeSearchPool(results=[{"url": "https://python.org", "title": "Python"}])

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, pool),
        max_iterations=5,
        emit_native_web_search=True,
    )

    types = [b.type for b in result.content]
    # The pair is prepended; the caller's tool_use survives so it can dispatch it.
    assert types == ["server_tool_use", "web_search_tool_result", "tool_use"]
    assert cast(Any, result.content[2]).name == "get_weather"
    # The gateway's own tool_use was filtered out: the client can never answer it.
    assert not any(getattr(b, "name", None) == "web_search" and b.type == "tool_use" for b in result.content)
    assert pool.calls == [("web_search", {"query": "python"})]


@pytest.mark.asyncio
async def test_mixed_batch_emits_no_native_blocks_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed = _message_response(
        stop_reason="tool_use",
        content=[_search_use("tu_1", "python"), _tool_use("tu_2", "get_weather", {})],
    )
    monkeypatch.setattr(messages_loop_module, "amessages", _fake_amessages_for([mixed]))

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        pool=cast(Any, _FakeSearchPool()),
        max_iterations=5,
    )

    assert [b.type for b in result.content] == ["tool_use"]


@pytest.mark.asyncio
async def test_stream_mixed_batch_emits_native_blocks_before_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming mixed batch: the pair goes out before message_delta / message_stop."""

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return _async_iter(
            _msg_start_event(),
            _tool_use_block_start(0, "tu_1", "web_search"),
            _input_json_delta(0, '{"query": "python"}'),
            _content_block_stop(0),
            _tool_use_block_start(1, "tu_2", "get_weather"),
            _input_json_delta(1, "{}"),
            _content_block_stop(1),
            _msg_delta_event("tool_use"),
            _msg_stop_event(),
        )

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)
    pool = _FakeSearchPool(results=[{"url": "https://python.org", "title": "Python"}])

    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
            pool=cast(Any, pool),
            max_iterations=5,
            emit_native_web_search=True,
        )
    ]

    types = [e.type for e in events]
    # The caller's tool_use is forwarded (renumbered to 0), then the gateway's pair,
    # then the terminal events.
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    starts = [e for e in events if e.type == "content_block_start"]
    assert [cast(Any, e.content_block).type for e in starts] == [
        "tool_use",
        "server_tool_use",
        "web_search_tool_result",
    ]
    # Gapless indices, and the terminal events stay last.
    assert [e.index for e in starts] == [0, 1, 2]
    assert types[-2:] == ["message_delta", "message_stop"]
    # The search really ran.
    assert pool.calls == [("web_search", {"query": "python"})]
