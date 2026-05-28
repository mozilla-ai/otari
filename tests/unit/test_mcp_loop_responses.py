"""Unit tests for the Responses API MCP tool-use loop and tool-format helpers.

Mirrors :mod:`tests.unit.test_mcp_loop` and
:mod:`tests.unit.test_mcp_loop_messages` semantically — same scenarios
expressed in the Responses API ``output`` items + ``function_call_output``
input items shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from any_llm.types.responses import Response, ResponseStreamEvent
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.services import mcp_loop_responses as responses_loop_module
from gateway.services.mcp_loop_responses import (
    MaxToolIterationsExceeded,
    responses_tool_loop,
    responses_tool_loop_stream,
)
from gateway.services.tool_format import (
    inject_purpose_hints_responses,
    openai_to_responses_tools,
)


class _FakePool:
    """Stand-in for MCPClientPool — same duck-typed surface as the other tests' _FakePool."""

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


def _function_call(
    call_id: str,
    name: str,
    arguments: str,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def _response(
    *,
    output: list[Any] | None = None,
    status: str = "completed",
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> Response:
    """Build a minimal Response. Many required fields are filled with neutral defaults."""
    return Response(
        id="resp_test",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, status),
        output=output or [],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens + output_tokens,
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


# ---------- pure helpers ----------


def test_openai_to_responses_tools_flattens_function_shape() -> None:
    out = openai_to_responses_tools(
        [{"type": "function", "function": {"name": "fetch", "description": "d", "parameters": {"type": "object"}}}]
    )
    assert out == [{"type": "function", "name": "fetch", "description": "d", "parameters": {"type": "object"}}]


def test_openai_to_responses_tools_omits_missing_optional_fields() -> None:
    out = openai_to_responses_tools([{"type": "function", "function": {"name": "noop"}}])
    assert out == [{"type": "function", "name": "noop"}]


def test_openai_to_responses_tools_passes_already_flat_shapes_through() -> None:
    flat = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
    out = openai_to_responses_tools([flat])
    assert out == [flat]


def test_inject_purpose_hints_responses_no_hints_returns_unchanged() -> None:
    kwargs: dict[str, Any] = {"instructions": "be helpful"}
    assert inject_purpose_hints_responses(kwargs, [])["instructions"] == "be helpful"


def test_inject_purpose_hints_responses_inserts_when_no_instructions() -> None:
    kwargs: dict[str, Any] = {}
    out = inject_purpose_hints_responses(kwargs, [("calendar", "for scheduling")])
    assert "calendar" in out["instructions"]
    assert "for scheduling" in out["instructions"]


def test_inject_purpose_hints_responses_prepends_existing_instructions() -> None:
    kwargs: dict[str, Any] = {"instructions": "be helpful"}
    out = inject_purpose_hints_responses(kwargs, [("cal", "use it")])
    assert "cal" in out["instructions"]
    assert "be helpful" in out["instructions"]
    assert out["instructions"].index("cal") < out["instructions"].index("be helpful")


# ---------- non-streaming loop ----------


@pytest.mark.asyncio
async def test_loop_returns_immediately_when_no_function_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_aresponses(**kwargs: Any) -> Response:
        calls.append(kwargs)
        return _response(output=[], status="completed")

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"])
    out = await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": [{"role": "user", "content": "hi"}]},
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert out.status == "completed"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_loop_executes_owned_function_call_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response(output=[_function_call("call_1", "fetch_url", '{"u":"x"}')]),
            _response(output=[], status="completed"),
        ]
    )
    captured_inputs: list[Any] = []

    async def fake_aresponses(**kwargs: Any) -> Response:
        # Snapshot the input list — the loop mutates it across iterations.
        captured_inputs.append(list(kwargs["input_data"]))
        return next(responses)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    out = await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": [{"role": "user", "content": "fetch x"}]},
        pool=cast(Any, pool),
        max_iterations=5,
    )

    assert pool.calls == [("fetch_url", {"u": "x"})]
    # Second call's input must include the function_call item + the
    # matching function_call_output item appended to the running list.
    second_input = captured_inputs[1]
    types = [item.get("type") for item in second_input if isinstance(item, dict)]
    assert "function_call" in types
    assert "function_call_output" in types
    output_item = next(
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert output_item["call_id"] == "call_1"
    assert output_item["output"] == "ok"
    assert out.status == "completed"


@pytest.mark.asyncio
async def test_loop_accumulates_usage_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response(output=[_function_call("c1", "fetch_url", "{}")], input_tokens=10, output_tokens=2),
            _response(output=[], status="completed", input_tokens=12, output_tokens=3),
        ]
    )

    async def fake_aresponses(**kwargs: Any) -> Response:
        return next(responses)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    out = await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
        max_iterations=5,
    )
    assert out.usage is not None
    assert out.usage.input_tokens == 22
    assert out.usage.output_tokens == 5
    assert out.usage.total_tokens == 27


@pytest.mark.asyncio
async def test_loop_max_iter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_aresponses(**kwargs: Any) -> Response:
        return _response(output=[_function_call("c", "fetch_url", "{}")])

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    with pytest.raises(MaxToolIterationsExceeded):
        await responses_tool_loop(
            completion_kwargs={"model": "fake", "input_data": "go"},
            pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
            max_iterations=2,
        )


@pytest.mark.asyncio
async def test_loop_foreign_function_call_returns_to_caller_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aresponses(**kwargs: Any) -> Response:
        return _response(output=[_function_call("c", "user_tool", "{}")])

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"])  # doesn't own user_tool
    out = await responses_tool_loop(
        completion_kwargs={
            "model": "fake",
            "input_data": "go",
            "tools": [{"type": "function", "name": "user_tool", "parameters": {}}],
        },
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert pool.calls == []
    types = [getattr(item, "type", None) for item in out.output]
    assert "function_call" in types


@pytest.mark.asyncio
async def test_loop_mixed_calls_executes_owned_and_returns_only_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aresponses(**kwargs: Any) -> Response:
        return _response(
            output=[
                _function_call("owned_id", "fetch_url", "{}"),
                _function_call("foreign_id", "user_tool", "{}"),
            ],
        )

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    out = await responses_tool_loop(
        completion_kwargs={
            "model": "fake",
            "input_data": "go",
            "tools": [{"type": "function", "name": "user_tool", "parameters": {}}],
        },
        pool=cast(Any, pool),
        max_iterations=5,
    )
    assert pool.calls == [("fetch_url", {})]
    remaining_call_ids = [
        getattr(item, "call_id", None)
        for item in out.output
        if getattr(item, "type", None) == "function_call"
    ]
    assert remaining_call_ids == ["foreign_id"]


@pytest.mark.asyncio
async def test_loop_tool_failure_appears_as_function_call_output(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response(output=[_function_call("c", "fetch_url", "{}")]),
            _response(output=[], status="completed"),
        ]
    )
    captured_inputs: list[Any] = []

    async def fake_aresponses(**kwargs: Any) -> Response:
        # Snapshot the input list — the loop mutates it across iterations.
        captured_inputs.append(list(kwargs["input_data"]))
        return next(responses)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    class FailingPool(_FakePool):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            raise RuntimeError("upstream down")

    pool = FailingPool(tool_names=["fetch_url"])
    await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=cast(Any, pool),
        max_iterations=5,
    )
    second_input = captured_inputs[1]
    output_item = next(
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert "tool error" in output_item["output"]
    assert "upstream down" in output_item["output"]


@pytest.mark.asyncio
async def test_loop_coerces_string_input_into_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Responses API accepts a string as ``input`` (treated as a single user
    message). To continue past a tool round we need a list; the loop must
    normalise on entry so the next call sees the full appended transcript.
    """
    responses = iter(
        [
            _response(output=[_function_call("c1", "fetch_url", "{}")]),
            _response(output=[], status="completed"),
        ]
    )
    captured_inputs: list[Any] = []

    async def fake_aresponses(**kwargs: Any) -> Response:
        # Snapshot via list() — the loop reuses the same input_items list
        # across iterations, so capturing the reference would show only the
        # final state.
        captured_inputs.append(list(kwargs["input_data"]))
        return next(responses)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"])
    await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": "hi"},
        pool=cast(Any, pool),
        max_iterations=5,
    )
    # First call was normalised to a list with just the user message.
    assert isinstance(captured_inputs[0], list)
    assert len(captured_inputs[0]) == 1
    assert captured_inputs[0][0]["role"] == "user"
    # Second call has the function_call + function_call_output appended.
    assert len(captured_inputs[1]) > len(captured_inputs[0])


# ---------- on_first_response (lock-in callback) ----------


@pytest.mark.asyncio
async def test_loop_fires_on_first_response_after_first_aresponses_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``on_first_response`` is invoked exactly once, right after the first
    upstream call returns successfully.
    """
    responses = iter(
        [
            _response(output=[_function_call("c1", "fetch_url", "{}")]),
            _response(output=[], status="completed"),
        ]
    )
    fire_order: list[str] = []

    async def fake_aresponses(**kwargs: Any) -> Response:
        fire_order.append("aresponses")
        return next(responses)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    def _on_first() -> None:
        fire_order.append("on_first_response")

    await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=cast(Any, _FakePool(tool_names=["fetch_url"])),
        max_iterations=5,
        on_first_response=_on_first,
    )
    assert fire_order[0] == "aresponses"
    assert fire_order[1] == "on_first_response"
    assert fire_order.count("on_first_response") == 1


@pytest.mark.asyncio
async def test_loop_does_not_fire_on_first_response_when_initial_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired = False

    async def fake_aresponses(**kwargs: Any) -> Response:
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    def _on_first() -> None:
        nonlocal fired
        fired = True

    with pytest.raises(RuntimeError, match="upstream 500"):
        await responses_tool_loop(
            completion_kwargs={"model": "fake", "input_data": "go"},
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

    async def fake_aresponses(**kwargs: Any) -> Response:
        return _response(output=[], status="completed")

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    out = await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": "hi"},
        pool=cast(Any, _FakePool(tool_names=[])),
        max_iterations=5,
    )
    assert out.status == "completed"


# ---------- streaming loop ----------


async def _async_iter(*events: ResponseStreamEvent) -> AsyncIterator[ResponseStreamEvent]:
    for e in events:
        yield e


def _output_item_added(output_index: int, item: Any, seq: int = 0) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent(
        type="response.output_item.added",
        item=item,
        output_index=output_index,
        sequence_number=seq,
    )


def _output_item_done(output_index: int, item: Any, seq: int = 0) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent(
        type="response.output_item.done",
        item=item,
        output_index=output_index,
        sequence_number=seq,
    )


def _function_call_args_delta(
    output_index: int, item_id: str, delta: str, seq: int = 0
) -> ResponseFunctionCallArgumentsDeltaEvent:
    return ResponseFunctionCallArgumentsDeltaEvent(
        type="response.function_call_arguments.delta",
        item_id=item_id,
        output_index=output_index,
        delta=delta,
        sequence_number=seq,
    )


def _function_call_args_done(
    output_index: int, item_id: str, name: str, arguments: str, seq: int = 0
) -> ResponseFunctionCallArgumentsDoneEvent:
    return ResponseFunctionCallArgumentsDoneEvent(
        type="response.function_call_arguments.done",
        item_id=item_id,
        output_index=output_index,
        name=name,
        arguments=arguments,
        sequence_number=seq,
    )


def _text_delta(item_id: str, output_index: int, delta: str, seq: int = 0) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        delta=delta,
        sequence_number=seq,
        logprobs=[],
    )


def _response_completed(seq: int = 0) -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        type="response.completed",
        response=_response(output=[], status="completed"),
        sequence_number=seq,
    )


@pytest.mark.asyncio
async def test_stream_passes_text_events_through_and_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_aresponses(**kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        return _async_iter(
            _text_delta("msg_1", 0, "hi"),
            _text_delta("msg_1", 0, " there"),
            _response_completed(),
        )

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    types: list[str] = []
    async for event in responses_tool_loop_stream(
        completion_kwargs={"model": "fake", "input_data": "hi"},
        pool=cast(Any, _FakePool(tool_names=[])),
        max_iterations=3,
    ):
        types.append(event.type)
    assert types == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]


@pytest.mark.asyncio
async def test_stream_runs_owned_function_call_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iteration 1 emits a function_call output item; the loop executes it and
    drops the intermediate response.completed. Iteration 2 runs and its
    terminal response.completed IS forwarded.
    """
    fc = _function_call("call_1", "fetch_url", "")
    iter_streams = iter(
        [
            _async_iter(
                _output_item_added(0, fc),
                _function_call_args_delta(0, "fc_item_1", '{"u":'),
                _function_call_args_delta(0, "fc_item_1", ' "x"}'),
                _function_call_args_done(0, "fc_item_1", "fetch_url", '{"u": "x"}'),
                _output_item_done(0, _function_call("call_1", "fetch_url", '{"u": "x"}')),
                _response_completed(),
            ),
            _async_iter(
                _text_delta("msg_1", 0, "done"),
                _response_completed(),
            ),
        ]
    )

    async def fake_aresponses(**kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    completed_count = 0
    async for event in responses_tool_loop_stream(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=cast(Any, pool),
        max_iterations=5,
    ):
        if event.type == "response.completed":
            completed_count += 1
    # Only the final iteration's response.completed is forwarded — the
    # intermediate one is dropped because the loop kept iterating.
    assert completed_count == 1
    assert pool.calls == [("fetch_url", {"u": "x"})]


@pytest.mark.asyncio
async def test_stream_foreign_function_call_forwards_terminal_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model emits a function_call for a foreign tool, the loop must
    forward response.completed so the client knows to dispatch.
    """
    fc = _function_call("call_1", "user_tool", "")
    iter_streams = iter(
        [
            _async_iter(
                _output_item_added(0, fc),
                _function_call_args_done(0, "fc_item_1", "user_tool", "{}"),
                _response_completed(),
            ),
        ]
    )

    async def fake_aresponses(**kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"])  # doesn't own user_tool
    types: list[str] = []
    async for event in responses_tool_loop_stream(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=cast(Any, pool),
        max_iterations=5,
    ):
        types.append(event.type)
    assert "response.completed" in types
    assert pool.calls == []


@pytest.mark.asyncio
async def test_stream_function_call_arguments_accumulate_across_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial ``arguments`` deltas across multiple events must concatenate
    before the tool input is parsed.
    """
    fc = _function_call("call_1", "fetch_url", "")
    iter_streams = iter(
        [
            _async_iter(
                _output_item_added(0, fc),
                _function_call_args_delta(0, "fc_item_1", '{"u":'),
                _function_call_args_delta(0, "fc_item_1", ' "x",'),
                _function_call_args_delta(0, "fc_item_1", ' "n": 3}'),
                # Deliberately omit the done event to confirm the loop can
                # work off the accumulated deltas alone.
                _response_completed(),
            ),
            _async_iter(
                _text_delta("msg_1", 0, "done"),
                _response_completed(),
            ),
        ]
    )

    async def fake_aresponses(**kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        return next(iter_streams)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    async for _event in responses_tool_loop_stream(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=cast(Any, pool),
        max_iterations=5,
    ):
        pass
    assert pool.calls == [("fetch_url", {"u": "x", "n": 3})]
