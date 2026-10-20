"""One tool loop over several backends: the gateway's web search, its sandbox and an MCP server.

Each backend here is a separate fake, combined the way the pipeline combines the real
ones, so what is asserted is that every call reaches the backend that owns it, that a
native rendering reads the buffer of the backend that ran the call, and that the
per-request caps and the failure wording follow the tool rather than the pool.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from any_llm.types.messages import (
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
from any_llm.types.responses import Response, ResponseStreamEvent
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.exceptions.tools_exceptions import McpServerDeclarationError
from gateway.services import mcp_loop as chat_loop_module
from gateway.services import mcp_loop_messages as messages_loop_module
from gateway.services import mcp_loop_responses as responses_loop_module
from gateway.services._tool_loop import tool_failure_detail
from gateway.services.mcp_client import MCPToolCallOutcome
from gateway.services.mcp_loop import mcp_tool_loop, mcp_tool_loop_stream
from gateway.services.mcp_loop_messages import anthropic_tool_loop, anthropic_tool_loop_stream
from gateway.services.mcp_loop_responses import responses_tool_loop, responses_tool_loop_stream
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, CodeExecution
from gateway.services.tools import (
    MAX_USES_EXCEEDED_ERROR,
    WEB_SEARCH_TOOL_NAME,
    CombinedToolBackend,
    ToolUseBudget,
    owning_backend,
    refuse_shadowed_builtin_tools,
)
from gateway.types.code_execution import ResultBlock

_MCP_TOOL = "lookup"
_NATIVE = frozenset({WEB_SEARCH_TOOL_NAME, CODE_EXECUTION_TOOL_NAME})


class _Backend:
    """The surface the loop drives, over a fixed set of tool names."""

    def __init__(self, *names: str, fail: bool = False) -> None:
        self._names = names
        self._fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in self._names]

    def owns_tool(self, name: str) -> bool:
        return name in self._names

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(name, f"use {name}") for name in self._names]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if self._fail:
            raise RuntimeError("secret://internal-host")
        return f"{name} result"


class _Search(_Backend):
    """The gateway's search: it keeps the hits a native rendering cites."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(WEB_SEARCH_TOOL_NAME, fail=fail)

    def take_last_results(self) -> list[dict[str, Any]]:
        return [{"url": "https://python.org", "title": "Python"}]


def _exec_result() -> ResultBlock:
    return ResultBlock.model_validate(
        {
            "type": "code_execution_tool_result",
            "content": {"type": "code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0},
        }
    )


class _Sandbox(_Backend):
    """The gateway's sandbox: it keeps the executions a native rendering reports."""

    container_id = "otari_cntr_test"
    files_base_url = None

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(CODE_EXECUTION_TOOL_NAME, fail=fail)
        self._executions: list[CodeExecution] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        code = str(arguments.get("code") or "")
        try:
            text = await super().call_tool(name, arguments)
        except RuntimeError:
            self._executions.append(CodeExecution(code=code, result=None))
            raise
        self._executions.append(CodeExecution(code=code, result=_exec_result()))
        return text

    def take_executions(self) -> list[CodeExecution]:
        taken, self._executions = self._executions, []
        return taken


class _Mcp(_Backend):
    """An MCP pool, with the server metadata the Messages stream reports activity from."""

    def __init__(self, *names: str, fail: bool = False) -> None:
        super().__init__(*(names or (_MCP_TOOL,)), fail=fail)

    def server_name_for_tool(self, name: str) -> str | None:
        return "docs-server" if self.owns_tool(name) else None

    async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
        text = await self.call_tool(name, arguments)
        return MCPToolCallOutcome(content=text, activity_content=text, is_error=False)


def _combined(*backends: _Backend) -> CombinedToolBackend:
    return CombinedToolBackend(cast(Any, backends))


# ---------- the combined pool ----------


@pytest.mark.asyncio
async def test_each_call_runs_on_the_backend_that_owns_it() -> None:
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()
    pool = _combined(search, sandbox, mcp)

    assert [tool["function"]["name"] for tool in pool.openai_tools] == ["web_search", "code_execution", "lookup"]
    assert [name for name, _ in pool.purpose_hints()] == ["web_search", "code_execution", "lookup"]
    assert all(pool.owns_tool(name) for name in ("web_search", "code_execution", "lookup"))
    assert not pool.owns_tool("caller_function")

    assert await pool.call_tool("lookup", {"q": "x"}) == "lookup result"
    assert await pool.call_tool("web_search", {"query": "y"}) == "web_search result"
    assert mcp.calls == [("lookup", {"q": "x"})]
    assert search.calls == [("web_search", {"query": "y"})]
    assert sandbox.calls == []


@pytest.mark.asyncio
async def test_a_call_no_backend_owns_is_refused() -> None:
    with pytest.raises(KeyError):
        await _combined(_Search(), _Mcp()).call_tool("caller_function", {})


@pytest.mark.asyncio
async def test_the_first_backend_owning_a_name_runs_it() -> None:
    """The pipeline puts the gateway's backends ahead of MCP, so a shadowing server never answers."""
    search, mcp = _Search(), _Mcp(WEB_SEARCH_TOOL_NAME)

    await _combined(search, mcp).call_tool(WEB_SEARCH_TOOL_NAME, {"query": "q"})

    assert search.calls and not mcp.calls


def test_owning_backend_unwraps_only_a_combined_pool() -> None:
    search, mcp = _Search(), _Mcp()
    pool = _combined(search, mcp)

    assert owning_backend(pool, WEB_SEARCH_TOOL_NAME) is search
    assert owning_backend(pool, _MCP_TOOL) is mcp
    assert owning_backend(pool, "caller_function") is pool
    assert owning_backend(cast(Any, search), WEB_SEARCH_TOOL_NAME) is search


def test_failure_wording_follows_the_tool_not_the_pool() -> None:
    pool = _combined(_Search(), _Mcp())

    assert tool_failure_detail(owning_backend(pool, _MCP_TOOL), _MCP_TOOL) == "MCP tool execution failed"
    assert (
        tool_failure_detail(owning_backend(pool, WEB_SEARCH_TOOL_NAME), WEB_SEARCH_TOOL_NAME)
        == "Gateway tool execution failed"
    )


def test_an_mcp_tool_named_like_a_declared_builtin_is_refused() -> None:
    with pytest.raises(McpServerDeclarationError, match="'web_search'"):
        refuse_shadowed_builtin_tools(cast(Any, _Mcp("web_search", "lookup")), [CODE_EXECUTION_TOOL_NAME, "web_search"])


def test_an_mcp_tool_named_like_an_undeclared_builtin_is_left_alone() -> None:
    refuse_shadowed_builtin_tools(cast(Any, _Mcp("web_search")), [CODE_EXECUTION_TOOL_NAME])


# ---------- Messages ----------


def _tool_use(block_id: str, name: str, arguments: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=block_id, name=name, input=arguments)


def _message(stop_reason: str, content: list[Any]) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="fake",
        content=content,
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=1, output_tokens=1),
        container=None,
    )


def _all_three_round() -> list[ToolUseBlock]:
    return [
        _tool_use("tu_1", WEB_SEARCH_TOOL_NAME, {"query": "python"}),
        _tool_use("tu_2", CODE_EXECUTION_TOOL_NAME, {"code": "print(6 * 7)"}),
        _tool_use("tu_3", _MCP_TOOL, {"q": "docs"}),
    ]


def _scripted_messages(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake_amessages(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return next(remaining)

    monkeypatch.setattr(messages_loop_module, "amessages", fake_amessages)
    return sent


@pytest.mark.asyncio
async def test_messages_runs_every_backend_in_one_loop_and_announces_the_builtins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _scripted_messages(
        monkeypatch,
        [
            _message("tool_use", _all_three_round()),
            _message("end_turn", [TextBlock(type="text", text="done", citations=None)]),
        ],
    )
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 10},
        pool=_combined(search, sandbox, mcp),
        max_iterations=5,
        native_tools=_NATIVE,
    )

    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 1, 1]
    # The MCP call has no honest Anthropic block, so only the gateway's own two are announced.
    assert [block.type for block in result.content] == [
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "code_execution_tool_result",
        "text",
    ]
    assert cast(Any, result.content[1]).content[0].url == "https://python.org"
    assert cast(Any, result.content[3]).content.stdout == "42\n"
    # Every call's result went back to the model, keyed to its own tool_use.
    results = {block["tool_use_id"]: block["content"] for block in sent[1]["messages"][-1]["content"]}
    assert results == {"tu_1": "web_search result", "tu_2": "code_execution result", "tu_3": "lookup result"}
    # The model was offered every backend's tools at once.
    assert {tool["name"] for tool in sent[0]["tools"]} == {"web_search", "code_execution", "lookup"}


@pytest.mark.asyncio
async def test_messages_caps_one_tool_while_the_others_keep_running(monkeypatch: pytest.MonkeyPatch) -> None:
    round_ = [
        _tool_use("tu_s", WEB_SEARCH_TOOL_NAME, {"query": "python"}),
        _tool_use("tu_c", CODE_EXECUTION_TOOL_NAME, {"code": "print(1)"}),
        _tool_use("tu_m", _MCP_TOOL, {"q": "docs"}),
    ]
    sent = _scripted_messages(
        monkeypatch,
        [
            _message("tool_use", round_),
            _message("tool_use", round_),
            _message("end_turn", [TextBlock(type="text", text="done", citations=None)]),
        ],
    )
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    result = await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 10},
        pool=_combined(search, sandbox, mcp),
        max_iterations=5,
        native_tools=_NATIVE,
        use_budget=ToolUseBudget({WEB_SEARCH_TOOL_NAME: 1}),
    )

    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 2, 2]
    second = {block["tool_use_id"]: block["content"] for block in sent[2]["messages"][-1]["content"]}
    assert second == {"tu_s": MAX_USES_EXCEEDED_ERROR, "tu_c": "code_execution result", "tu_m": "lookup result"}
    search_results = [cast(Any, b).content for b in result.content if b.type == "web_search_tool_result"]
    assert [getattr(content, "error_code", None) for content in search_results] == [None, "max_uses_exceeded"]


@pytest.mark.asyncio
async def test_messages_failures_are_worded_by_the_tool_that_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _scripted_messages(
        monkeypatch,
        [
            _message(
                "tool_use",
                [
                    _tool_use("tu_m", _MCP_TOOL, {"q": "docs"}),
                    _tool_use("tu_c", CODE_EXECUTION_TOOL_NAME, {"code": "print(1)"}),
                ],
            ),
            _message("end_turn", [TextBlock(type="text", text="done", citations=None)]),
        ],
    )

    await anthropic_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 10},
        pool=_combined(_Sandbox(fail=True), _Mcp(fail=True)),
        max_iterations=5,
    )

    results = {block["tool_use_id"]: block["content"] for block in sent[1]["messages"][-1]["content"]}
    assert results == {
        "tu_m": "[tool error] MCP tool execution failed",
        "tu_c": "[tool error] Gateway tool execution failed",
    }


async def _events(*events: Any) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _streamed_tool_round(calls: list[tuple[str, str, dict[str, Any]]]) -> AsyncIterator[MessageStreamEvent]:
    events: list[Any] = [MessageStartEvent(type="message_start", message=cast(Any, _message("end_turn", [])))]
    for index, (block_id, name, arguments) in enumerate(calls):
        events += [
            ContentBlockStartEvent(
                type="content_block_start",
                index=index,
                content_block=cast(Any, ToolUseBlock(type="tool_use", id=block_id, name=name, input={})),
            ),
            ContentBlockDeltaEvent(
                type="content_block_delta",
                index=index,
                delta=cast(Any, InputJSONDelta(type="input_json_delta", partial_json=json.dumps(arguments))),
            ),
            ContentBlockStopEvent(type="content_block_stop", index=index),
        ]
    events += [_message_delta("tool_use"), MessageStopEvent(type="message_stop")]
    return _events(*events)


def _message_delta(stop_reason: str) -> MessageDeltaEvent:
    return MessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )


def _streamed_text_round() -> AsyncIterator[MessageStreamEvent]:
    return _events(
        MessageStartEvent(type="message_start", message=cast(Any, _message("end_turn", []))),
        ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=cast(Any, TextBlock(type="text", text="", citations=None)),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta", index=0, delta=cast(Any, TextDelta(type="text_delta", text="done"))
        ),
        ContentBlockStopEvent(type="content_block_stop", index=0),
        _message_delta("end_turn"),
        MessageStopEvent(type="message_stop"),
    )


@pytest.mark.asyncio
async def test_messages_stream_runs_every_backend_and_announces_each_in_its_own_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [(block.id, block.name, dict(block.input)) for block in _all_three_round()]
    _scripted_messages(monkeypatch, [_streamed_tool_round(calls), _streamed_text_round()])
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    events = [
        event
        async for event in anthropic_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}], "max_tokens": 10},
            pool=_combined(search, sandbox, mcp),
            max_iterations=5,
            native_tools=_NATIVE,
            emit_native_mcp=True,
        )
    ]

    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 1, 1]
    started = [
        cast(Any, event).content_block.type for event in events if getattr(event, "type", None) == "content_block_start"
    ]
    # The MCP call is reported live around its execution; the built-ins after the round, then the answer.
    assert started == [
        "mcp_tool_use",
        "mcp_tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "code_execution_tool_result",
        "text",
    ]
    # No gateway-owned tool_use leaks to the client.
    assert "tool_use" not in started
    indices = [cast(Any, event).index for event in events if getattr(event, "type", None) == "content_block_start"]
    assert indices == list(range(len(indices)))


# ---------- Responses ----------


def _function_call(call_id: str, name: str, arguments: dict[str, Any]) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(type="function_call", call_id=call_id, name=name, arguments=json.dumps(arguments))


def _response(output: list[Any]) -> Response:
    return Response(
        id="resp_1",
        created_at=0.0,
        model="fake",
        object="response",
        status="completed",
        output=output,
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
    )


_RESPONSES_ROUND = [
    ("c1", WEB_SEARCH_TOOL_NAME, {"query": "python"}),
    ("c2", CODE_EXECUTION_TOOL_NAME, {"code": "print(6 * 7)"}),
    ("c3", _MCP_TOOL, {"q": "docs"}),
]


def _scripted_responses(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake_aresponses(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return next(remaining)

    monkeypatch.setattr(responses_loop_module, "aresponses", fake_aresponses)
    return sent


@pytest.mark.asyncio
async def test_responses_runs_every_backend_and_announces_the_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _scripted_responses(
        monkeypatch,
        [_response([_function_call(*call) for call in _RESPONSES_ROUND]), _response([])],
    )
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    result = await responses_tool_loop(
        completion_kwargs={"model": "fake", "input_data": "go"},
        pool=_combined(search, sandbox, mcp),
        max_iterations=5,
        native_tools=_NATIVE,
    )

    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 1, 1]
    assert [item.type for item in result.output] == ["web_search_call", "code_interpreter_call"]
    assert cast(Any, result.output[1]).container_id == _Sandbox.container_id
    outputs = {
        item["call_id"]: item["output"] for item in sent[1]["input_data"] if item.get("type") == "function_call_output"
    }
    assert outputs == {"c1": "web_search result", "c2": "code_execution result", "c3": "lookup result"}


@pytest.mark.asyncio
async def test_responses_stream_runs_every_backend_and_announces_the_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    first: list[Any] = []
    for index, (call_id, name, arguments) in enumerate(_RESPONSES_ROUND):
        item = _function_call(call_id, name, arguments)
        first += [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=_function_call(call_id, name, {}),
                output_index=index,
                sequence_number=0,
            ),
            ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                item_id=f"fc_{index}",
                output_index=index,
                name=name,
                arguments=item.arguments,
                sequence_number=0,
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done", item=item, output_index=index, sequence_number=0
            ),
        ]
    completed = ResponseCompletedEvent(type="response.completed", response=_response([]), sequence_number=0)
    _scripted_responses(monkeypatch, [_events(*first, completed), _events(completed)])
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    events: list[ResponseStreamEvent] = [
        event
        async for event in responses_tool_loop_stream(
            completion_kwargs={"model": "fake", "input_data": "go"},
            pool=_combined(search, sandbox, mcp),
            max_iterations=5,
            native_tools=_NATIVE,
        )
    ]

    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 1, 1]
    done = [
        cast(Any, event).item.type for event in events if getattr(event, "type", None) == "response.output_item.done"
    ]
    assert done == ["web_search_call", "code_interpreter_call"]


# ---------- Chat Completions ----------


def _completion(calls: list[tuple[str, str, dict[str, Any]]] | None) -> ChatCompletion:
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=call_id, type="function", function=Function(name=name, arguments=json.dumps(arguments))
        )
        for call_id, name, arguments in calls or []
    ]
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model="fake",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None if calls else "done", tool_calls=cast(Any, tool_calls or None)
                ),
                finish_reason="tool_calls" if calls else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.asyncio
async def test_chat_runs_every_backend_and_caps_only_the_capped_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []
    remaining = iter([_completion(_RESPONSES_ROUND), _completion(_RESPONSES_ROUND), _completion(None)])

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        sent.append(kwargs)
        return next(remaining)

    monkeypatch.setattr(chat_loop_module, "acompletion", fake_acompletion)
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    result = await mcp_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
        pool=_combined(search, sandbox, mcp),
        max_iterations=5,
        use_budget=ToolUseBudget({WEB_SEARCH_TOOL_NAME: 1}),
    )

    assert result.choices[0].message.content == "done"
    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 2, 2]
    second = {m["tool_call_id"]: m["content"] for m in sent[2]["messages"] if m.get("role") == "tool"}
    assert second["c1"] == MAX_USES_EXCEEDED_ERROR


@pytest.mark.asyncio
async def test_chat_stream_runs_every_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk.model_validate(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fake",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    tool_round = [
        chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            }
        )
        for index, (call_id, name, arguments) in enumerate(_RESPONSES_ROUND)
    ]
    rounds = iter([_events(*tool_round, chunk({}, "tool_calls")), _events(chunk({"content": "done"}, "stop"))])

    async def fake_acompletion(**kwargs: Any) -> Any:
        return next(rounds)

    monkeypatch.setattr(chat_loop_module, "acompletion", fake_acompletion)
    search, sandbox, mcp = _Search(), _Sandbox(), _Mcp()

    chunks = [
        c
        async for c in mcp_tool_loop_stream(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
            pool=_combined(search, sandbox, mcp),
            max_iterations=5,
        )
    ]

    assert [len(b.calls) for b in (search, sandbox, mcp)] == [1, 1, 1]
    assert not any(c.choices and c.choices[0].delta.tool_calls for c in chunks), "a gateway call leaked to the client"
    assert chunks[-1].choices[0].finish_reason == "stop"
