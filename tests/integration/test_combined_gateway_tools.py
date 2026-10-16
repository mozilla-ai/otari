"""One request combining the gateway's web search, web fetch, code execution sandbox and MCP servers.

The code execution port is faked at the composition root and the MCP pool at the
pipeline, while the web retrieval backend runs with only its search and its page
fetch patched, so the real admission, backends, combined pool and tool loops run.
The provider is scripted to call every declared tool in one round and answer in the next.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
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
    MessageUsage,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from any_llm.types.responses import Response
from fastapi.testclient import TestClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from sqlalchemy.orm import Session

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.models.usage import UsageLog
from gateway.ports.code_execution_port import SandboxFileEntry
from gateway.services.mcp_client import MCPToolCallOutcome
from gateway.services.mcp_loop_messages import MCP_ACTIVITY_ID_PREFIX
from gateway.services.tool_usage import TOOL_METER_NAMESPACE
from gateway.services.tools import (
    CODE_EXECUTION_HEADER,
    MAX_USES_EXCEEDED_ERROR,
    SERVER_TOOL_USE_ID_PREFIX,
    WEB_FETCH_HEADER,
    WEB_SEARCH_HEADER,
)
from gateway.services.web_fetch_service import WebFetchResult, WebFetchService
from gateway.services.web_retrieval_backend import WebRetrievalBackend
from gateway.services.web_retrieval_policy import canonicalize_web_url
from gateway.types.code_execution import ResultBlock

from .conftest import MODEL_NAME, app_for

_MCP_TOOL = "lookup"
_MCP_SERVER = {"name": "docs", "url": "http://127.0.0.1:9999/mcp"}
_SEARCH = {"type": "otari_web_search"}
_FETCH = {"type": "otari_web_fetch"}
_CODE = {"type": "otari_code_execution"}
_ARGUMENTS: dict[str, dict[str, object]] = {
    "web_search": {"query": "python"},
    "web_fetch": {"url": "https://example.com/"},
    "code_execution": {"code": "print(6 * 7)"},
    _MCP_TOOL: {"q": "docs"},
}


class _Session:
    """A sandbox session whose every run prints 42 and writes ``chart.png``."""

    session_id = "sbx-1"
    holds_across_requests = False

    def __init__(self, port: _Port) -> None:
        self._port = port
        self._files: list[SandboxFileEntry] = []

    def discard(self) -> None:
        return None

    async def execute(self, code: str, *, timeout_s: float) -> ResultBlock:
        self._port.executed.append(code)
        self._files = [SandboxFileEntry(path="chart.png", size_bytes=3, modified_at=1.0)]
        return ResultBlock.model_validate(
            {
                "type": "code_execution_tool_result",
                "content": {"type": "code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0},
            }
        )

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        return None

    async def list_files(self) -> list[SandboxFileEntry]:
        return list(self._files)

    async def read_file(self, path: str, *, budget_bytes: int) -> AsyncIterator[bytes]:
        yield b"png"


class _Port:
    label = "fake-sandbox"

    def __init__(self) -> None:
        self.opened = 0
        self.executed: list[str] = []

    @asynccontextmanager
    async def open_session(self, **kwargs: Any) -> AsyncIterator[_Session]:
        self.opened += 1
        yield _Session(self)


@dataclass
class _Backends:
    """What each of the request's backends was asked to do."""

    port: _Port = field(default_factory=_Port)
    mcp_tools: tuple[str, ...] = (_MCP_TOOL,)
    mcp_calls: list[str] = field(default_factory=list)
    searches: AsyncMock = field(
        default_factory=lambda: AsyncMock(
            return_value=[{"url": "https://python.org", "title": "Python", "content": "3.14"}]
        )
    )
    fetches: AsyncMock = field(default_factory=lambda: AsyncMock(return_value=_page()))


def _page() -> WebFetchResult:
    return WebFetchResult(
        text="page text",
        content_type="text/html",
        content_kind="html",
        requested_url=canonicalize_web_url("https://example.com/"),
        final_url=canonicalize_web_url("https://example.com/"),
        redirect_count=0,
    )


def _fake_mcp_pool(backends: _Backends) -> Callable[..., Any]:
    class _Pool:
        """The MCP pool's surface, recording on the request's tally as the real one does."""

        def __init__(self, configs: Any, *, tally: Any = None, trace: Any = None) -> None:
            self._tally = tally

        async def __aenter__(self) -> _Pool:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        @property
        def openai_tools(self) -> list[dict[str, Any]]:
            return [
                {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
                for name in backends.mcp_tools
            ]

        def owns_tool(self, name: str) -> bool:
            return name in backends.mcp_tools

        def purpose_hints(self) -> list[tuple[str, str]]:
            return []

        def server_name_for_tool(self, name: str) -> str | None:
            return "docs" if self.owns_tool(name) else None

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            backends.mcp_calls.append(name)
            if self._tally is not None:
                self._tally.record_result(name, "mcp result")
            return "mcp result"

        async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
            text = await self.call_tool(name, arguments)
            return MCPToolCallOutcome(content=text, activity_content=text, is_error=False)

    return _Pool


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig, tmp_path: Path) -> Iterator[_Backends]:
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://web-search.invalid")
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)
    # The local file store writes the sandbox's output files under the working directory.
    monkeypatch.chdir(tmp_path)
    seen = _Backends()
    with (
        patch("gateway.container.build_code_execution_port", new=lambda _config: seen.port),
        patch("gateway.api.routes._pipeline.MCPClientPool", new=_fake_mcp_pool(seen)),
        patch.object(WebRetrievalBackend, "_search", new=seen.searches),
        patch.object(WebFetchService, "fetch", new=seen.fetches),
        # A search reads each hit's page through the same service; these hits keep their snippets.
        patch.object(WebFetchService, "fetch_for_search", new=AsyncMock(side_effect=OSError)),
    ):
        yield seen


def _ran(backends: _Backends) -> set[str]:
    ran: set[str] = set()
    if backends.searches.await_count:
        ran.add("web_search")
    if backends.fetches.await_count:
        ran.add("web_fetch")
    if backends.port.executed:
        ran.add("code_execution")
    ran.update(backends.mcp_calls)
    return ran


# ---------- Messages ----------


def _message(stop_reason: str, content: list[Any]) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="fake",
        content=content,
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _done_message() -> MessageResponse:
    return _message("end_turn", [TextBlock(type="text", text="done", citations=None)])


def _tool_uses(names: list[str]) -> list[ToolUseBlock]:
    return [
        ToolUseBlock(type="tool_use", id=f"tu_{index}", name=name, input=_ARGUMENTS[name])
        for index, name in enumerate(names)
    ]


@contextmanager
def _scripted(target: str, responses: list[Any]) -> Iterator[list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return next(remaining)

    with patch(target, new=fake):
        yield sent


_COMBINATIONS = {
    "search and code": ([_SEARCH, _CODE], False, ["web_search", "code_execution"]),
    "fetch and code": ([_FETCH, _CODE], False, ["web_fetch", "code_execution"]),
    "search and MCP": ([_SEARCH], True, ["web_search", _MCP_TOOL]),
    "code and MCP": ([_CODE], True, ["code_execution", _MCP_TOOL]),
    "all four": ([_SEARCH, _FETCH, _CODE], True, ["web_search", "web_fetch", "code_execution", _MCP_TOOL]),
}


def _messages_body(tools: list[dict[str, Any]], with_mcp: bool, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "research and compute"}],
        "max_tokens": 100,
        "tools": tools,
        **extra,
    }
    if with_mcp:
        body["mcp_servers"] = [_MCP_SERVER]
    return body


@pytest.mark.parametrize("combination", list(_COMBINATIONS))
def test_messages_runs_each_combination_in_one_loop(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends, combination: str
) -> None:
    tools, with_mcp, names = _COMBINATIONS[combination]
    with _scripted(
        "gateway.services.mcp_loop_messages.amessages", [_message("tool_use", _tool_uses(names)), _done_message()]
    ) as sent:
        response = client.post(f"{API_ROOT}/messages", json=_messages_body(tools, with_mcp), headers=api_key_header)

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(names)
    assert {tool["name"] for tool in sent[0]["tools"]} == set(names)
    results = sent[1]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in results] == [f"tu_{index}" for index in range(len(names))]
    assert not any(block["content"].startswith("[tool error]") for block in results)


def test_messages_announces_each_builtin_natively_and_meters_every_tool(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    backends: _Backends,
    db_session_factory: Callable[[], Session],
) -> None:
    names = ["web_search", "web_fetch", "code_execution", _MCP_TOOL]
    with _scripted(
        "gateway.services.mcp_loop_messages.amessages", [_message("tool_use", _tool_uses(names)), _done_message()]
    ) as sent:
        response = client.post(f"{API_ROOT}/messages", json=_native_body(), headers=_native_headers(api_key_header))

    assert response.status_code == 200, response.text
    content = response.json()["content"]
    assert [block["type"] for block in content] == [
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_fetch_tool_result",
        "server_tool_use",
        "code_execution_tool_result",
        "text",
    ]
    assert content[1]["content"][0]["url"] == "https://python.org"
    assert content[3]["content"]["url"] == "https://example.com/"
    # The file the run wrote was stored, and the native result names it by the id a caller downloads.
    stored = [ref["file_id"] for ref in content[5]["content"]["content"]]
    assert len(stored) == 1 and stored[0].startswith("file")
    code_result = next(block for block in sent[1]["messages"][-1]["content"] if block["tool_use_id"] == "tu_2")
    assert f"chart.png (file_id: {stored[0]})" in code_result["content"]

    tools = (_latest_row(db_session_factory, api_key_obj["user_id"]).billing_meters or {})[TOOL_METER_NAMESPACE]
    assert {name: meter["billed"] for name, meter in tools.items()} == {
        "web_search": 1,
        "web_fetch": 1,
        "code_execution": 1,
        _MCP_TOOL: 1,
    }


def _native_body(**extra: Any) -> dict[str, Any]:
    """Each built-in declared in Anthropic's own keyword, beside an MCP server."""
    return _messages_body(
        [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"type": "code_execution_20250825", "name": "code_execution"},
        ],
        True,
        **extra,
    )


def _native_headers(api_key_header: dict[str, str]) -> dict[str, str]:
    return {**api_key_header, WEB_SEARCH_HEADER: "otari", WEB_FETCH_HEADER: "otari", CODE_EXECUTION_HEADER: "otari"}


def _latest_row(db_session_factory: Callable[[], Session], user_id: str) -> UsageLog:
    db = db_session_factory()
    try:
        row = db.query(UsageLog).filter(UsageLog.user_id == user_id).order_by(UsageLog.timestamp.desc()).first()
        assert row is not None, "expected a usage row for the request"
        db.expunge(row)
        return row
    finally:
        db.close()


async def _events(*events: Any) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _message_delta(stop_reason: str) -> MessageDeltaEvent:
    return MessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 2},
        }
    )


def _streamed_tool_round(names: list[str]) -> AsyncIterator[Any]:
    events: list[Any] = [MessageStartEvent(type="message_start", message=cast(Any, _message("end_turn", [])))]
    for index, block in enumerate(_tool_uses(names)):
        events += [
            ContentBlockStartEvent(
                type="content_block_start",
                index=index,
                content_block=cast(Any, block.model_copy(update={"input": {}})),
            ),
            ContentBlockDeltaEvent(
                type="content_block_delta",
                index=index,
                delta=cast(Any, InputJSONDelta(type="input_json_delta", partial_json=json.dumps(block.input))),
            ),
            ContentBlockStopEvent(type="content_block_stop", index=index),
        ]
    return _events(*events, _message_delta("tool_use"), MessageStopEvent(type="message_stop"))


def _streamed_text_round() -> AsyncIterator[Any]:
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


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: {")]


@pytest.mark.parametrize("combination", list(_COMBINATIONS))
def test_messages_stream_runs_each_combination_in_one_loop(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends, combination: str
) -> None:
    tools, with_mcp, names = _COMBINATIONS[combination]
    with _scripted(
        "gateway.services.mcp_loop_messages.amessages", [_streamed_tool_round(names), _streamed_text_round()]
    ) as sent:
        response = client.post(
            f"{API_ROOT}/messages", json=_messages_body(tools, with_mcp, stream=True), headers=api_key_header
        )

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(names)
    assert len(sent) == 2
    payloads = _sse_payloads(response.text)
    started = [p["content_block"]["type"] for p in payloads if p.get("type") == "content_block_start"]
    assert started == ["text"], "a gateway-run tool_use leaked to the client"
    assert [payload["type"] for payload in payloads if payload][-1] == "message_stop"


def test_messages_stream_announces_mcp_activity_and_builtins_natively(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    names = ["web_search", "web_fetch", "code_execution", _MCP_TOOL]
    with _scripted(
        "gateway.services.mcp_loop_messages.amessages", [_streamed_tool_round(names), _streamed_text_round()]
    ):
        response = client.post(
            f"{API_ROOT}/messages",
            json=_native_body(stream=True),
            headers={**_native_headers(api_key_header), "anthropic-beta": "mcp-client-2025-11-20"},
        )

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(names)
    started = [p["content_block"] for p in _sse_payloads(response.text) if p.get("type") == "content_block_start"]
    # The MCP call is reported live around its execution, the built-ins once the round ran.
    assert [block["type"] for block in started] == [
        "mcp_tool_use",
        "mcp_tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_fetch_tool_result",
        "server_tool_use",
        "code_execution_tool_result",
        "text",
    ]


def test_an_echoed_combined_transcript_reaches_the_provider_without_the_gateways_blocks(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    """Both kinds of block the gateway mints are stripped, so the next turn of a combined run replays cleanly."""
    server_use_id = f"{SERVER_TOOL_USE_ID_PREFIX}abc"
    activity_id = f"{MCP_ACTIVITY_ID_PREFIX}def"
    echoed = [
        {"role": "user", "content": "research and compute"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "mcp_tool_use",
                    "id": activity_id,
                    "name": _MCP_TOOL,
                    "server_name": "docs",
                    "input": {"q": "docs"},
                },
                {"type": "mcp_tool_result", "tool_use_id": activity_id, "content": "mcp result", "is_error": False},
                {"type": "server_tool_use", "id": server_use_id, "name": "web_search", "input": {"query": "python"}},
                {"type": "web_search_tool_result", "tool_use_id": server_use_id, "content": []},
                {"type": "text", "text": "done"},
            ],
        },
        {"role": "user", "content": "and again"},
    ]
    with _scripted("gateway.services.mcp_loop_messages.amessages", [_done_message()]) as sent:
        response = client.post(
            f"{API_ROOT}/messages",
            json=_messages_body([_SEARCH, _CODE], True, messages=echoed),
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert sent[0]["messages"][1]["content"] == [{"type": "text", "text": "done"}]


# ---------- collisions, caps and policy ----------


def test_an_mcp_tool_named_like_a_declared_builtin_is_refused_before_a_sandbox_opens(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    backends.mcp_tools = ("web_search", _MCP_TOOL)
    with _scripted("gateway.services.mcp_loop_messages.amessages", []) as sent:
        response = client.post(
            f"{API_ROOT}/messages", json=_messages_body([_SEARCH, _CODE], True), headers=api_key_header
        )

    assert response.status_code == 400, response.text
    assert "'web_search'" in response.json()["detail"]["error"]["message"]
    assert backends.port.opened == 0
    assert sent == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_shadowing_mcp_tool_is_refused_on_responses_too(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends, stream: bool
) -> None:
    backends.mcp_tools = ("code_execution",)
    response = client.post(
        f"{API_ROOT}/responses",
        json={"model": MODEL_NAME, "input": "hi", "tools": [_CODE], "mcp_servers": [_MCP_SERVER], "stream": stream},
        headers=api_key_header,
    )

    assert response.status_code == 400, response.text
    assert "'code_execution'" in response.json()["detail"]
    assert backends.port.opened == 0


def test_an_mcp_tool_named_like_an_undeclared_builtin_still_runs(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    """Only a declared built-in reserves its name: a request without web search keeps the server's own."""
    backends.mcp_tools = ("web_search",)
    with _scripted(
        "gateway.services.mcp_loop_messages.amessages",
        [_message("tool_use", _tool_uses(["web_search"])), _done_message()],
    ):
        response = client.post(f"{API_ROOT}/messages", json=_messages_body([_CODE], True), headers=api_key_header)

    assert response.status_code == 200, response.text
    assert backends.mcp_calls == ["web_search"]
    assert backends.searches.await_count == 0


def _completion(names: list[str] | None) -> ChatCompletion:
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{index}", type="function", function=Function(name=name, arguments=json.dumps(_ARGUMENTS[name]))
        )
        for index, name in enumerate(names or [])
    ]
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None if names else "done", tool_calls=cast(Any, calls or None)
                ),
                finish_reason="tool_calls" if names else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_a_capped_search_beside_mcp_stops_at_its_cap_while_mcp_keeps_running(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    names = ["web_search", _MCP_TOOL]
    with _scripted(
        "gateway.services.mcp_loop.acompletion", [_completion(names), _completion(names), _completion(None)]
    ) as sent:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "research"}],
                "tools": [{"type": "otari_web_search", "max_uses": 1}],
                "mcp_servers": [_MCP_SERVER],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert backends.searches.await_count == 1
    assert backends.mcp_calls == [_MCP_TOOL, _MCP_TOOL]
    second_round = {m["tool_call_id"]: m["content"] for m in sent[2]["messages"] if m.get("role") == "tool"}
    assert second_round["call_0"] == MAX_USES_EXCEEDED_ERROR
    assert second_round["call_1"] == "mcp result"


def _chunk(delta: dict[str, Any], finish_reason: str | None = None, usage: dict[str, int] | None = None) -> Any:
    return ChatCompletionChunk.model_validate(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **({"usage": usage} if usage else {}),
        }
    )


_STREAM_USAGE = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def _streamed_chat_round(names: list[str]) -> AsyncIterator[Any]:
    calls = [
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(_ARGUMENTS[name])},
                    }
                ]
            }
        )
        for index, name in enumerate(names)
    ]
    return _events(*calls, _chunk({}, "tool_calls", _STREAM_USAGE))


def _streamed_chat_answer() -> AsyncIterator[Any]:
    return _events(_chunk({"content": "done"}), _chunk({}, "stop", _STREAM_USAGE))


def test_chat_stream_runs_all_four_in_one_loop(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    names = ["web_search", "web_fetch", "code_execution", _MCP_TOOL]
    with _scripted("gateway.services.mcp_loop.acompletion", [_streamed_chat_round(names), _streamed_chat_answer()]):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "research"}],
                "tools": [_SEARCH, _FETCH, _CODE],
                "mcp_servers": [_MCP_SERVER],
                "stream": True,
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(names)
    deltas = [p["choices"][0]["delta"] for p in _sse_payloads(response.text) if p.get("choices")]
    assert not any(delta.get("tool_calls") for delta in deltas), "a gateway-run call leaked to the client"
    assert "".join(delta.get("content") or "" for delta in deltas) == "done"


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> str:
    listed = client.get(f"{API_ROOT}/workspaces", headers=master_key_header)
    assert listed.status_code == 200
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


@pytest.mark.parametrize(("companion", "with_mcp"), [(_SEARCH, False), (_FETCH, False), (None, True)])
def test_code_execution_disabled_for_the_workspace_refuses_the_whole_request(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    backends: _Backends,
    companion: dict[str, Any] | None,
    with_mcp: bool,
) -> None:
    """A tool its policy refuses is not dropped from a combination: the request is refused, as it is alone."""
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": False},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text

    tools = [_CODE, *([companion] if companion else [])]
    with _scripted("gateway.services.mcp_loop_messages.amessages", []) as sent:
        response = client.post(f"{API_ROOT}/messages", json=_messages_body(tools, with_mcp), headers=api_key_header)

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"]["message"] == "code execution is not enabled for this workspace"
    assert sent == []
    assert _ran(backends) == set()


@pytest.mark.parametrize(("companion", "with_mcp"), [(_CODE, False), (None, True), (_CODE, True)])
def test_web_search_disabled_for_the_workspace_refuses_the_whole_request(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    backends: _Backends,
    companion: dict[str, Any] | None,
    with_mcp: bool,
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search", json={"enabled": False}, headers=master_key_header
    )
    assert stored.status_code == 200, stored.text

    tools = [_SEARCH, *([companion] if companion else [])]
    with _scripted("gateway.services.mcp_loop_messages.amessages", []) as sent:
        response = client.post(f"{API_ROOT}/messages", json=_messages_body(tools, with_mcp), headers=api_key_header)

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"]["message"] == "web search is not enabled for this workspace"
    assert sent == []
    assert backends.port.opened == 0
    assert _ran(backends) == set()


# ---------- Responses ----------


def _response(output: list[Any]) -> Response:
    return Response(
        id="resp_1",
        created_at=0.0,
        model=MODEL_NAME,
        object="response",
        status="completed",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=5,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=7,
        ),
    )


_RESPONSES_NAMES = ["web_search", "web_fetch", "code_execution", _MCP_TOOL]


def _function_calls() -> list[ResponseFunctionToolCall]:
    return [
        ResponseFunctionToolCall(
            type="function_call", call_id=f"c{index}", name=name, arguments=json.dumps(_ARGUMENTS[name])
        )
        for index, name in enumerate(_RESPONSES_NAMES)
    ]


def _responses_body(**extra: Any) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "input": "research and compute",
        "tools": [_SEARCH, _FETCH, {"type": "code_interpreter", "container": {"type": "auto"}}],
        "mcp_servers": [_MCP_SERVER],
        **extra,
    }


def test_responses_runs_all_four_and_announces_the_builtins(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    with _scripted(
        "gateway.services.mcp_loop_responses.aresponses", [_response(_function_calls()), _response([])]
    ) as sent:
        response = client.post(
            f"{API_ROOT}/responses", json=_responses_body(), headers={**api_key_header, CODE_EXECUTION_HEADER: "otari"}
        )

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(_RESPONSES_NAMES)
    assert [item["type"] for item in response.json()["output"]] == ["web_search_call", "code_interpreter_call"]
    outputs = [item for item in sent[1]["input_data"] if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["c0", "c1", "c2", "c3"]


def _streamed_function_calls() -> AsyncIterator[Any]:
    events: list[Any] = []
    for index, call in enumerate(_function_calls()):
        events += [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=call.model_copy(update={"arguments": ""}),
                output_index=index,
                sequence_number=0,
            ),
            ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                item_id=f"fc_{index}",
                output_index=index,
                name=call.name,
                arguments=call.arguments,
                sequence_number=0,
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done", item=call, output_index=index, sequence_number=0
            ),
        ]
    return _events(*events, _completed())


def _completed() -> ResponseCompletedEvent:
    return ResponseCompletedEvent(type="response.completed", response=_response([]), sequence_number=0)


def test_responses_stream_runs_all_four_and_announces_the_builtins(
    client: TestClient, api_key_header: dict[str, str], backends: _Backends
) -> None:
    with _scripted(
        "gateway.services.mcp_loop_responses.aresponses", [_streamed_function_calls(), _events(_completed())]
    ):
        response = client.post(
            f"{API_ROOT}/responses",
            json=_responses_body(stream=True),
            headers={**api_key_header, CODE_EXECUTION_HEADER: "otari"},
        )

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(_RESPONSES_NAMES)
    done = [p["item"]["type"] for p in _sse_payloads(response.text) if p.get("type") == "response.output_item.done"]
    assert done == ["web_search_call", "code_interpreter_call"], (
        "a gateway-run call leaked, or a built-in went unannounced"
    )


# ---------- hybrid ----------


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = app_for(
        GatewayConfig(
            mode="hybrid",
            web_fetch_enabled=True,
            provider_user_identifier_enabled=False,
            platform={"base_url": "http://platform.test/api/v1"},
        )
    )
    with TestClient(app) as client:
        yield client
    reset_config()
    reset_db()


@dataclass
class _ControlPlane:
    """What a hybrid gateway asked its control plane, by path."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def bodies(self, suffix: str) -> list[dict[str, Any]]:
        return [body for url, body in self.calls if url.endswith(suffix)]


@pytest.fixture
def control_plane(control_plane_transport: InstallControlPlane) -> _ControlPlane:
    seen = _ControlPlane()

    async def answer(url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> httpx.Response:
        seen.calls.append((url, body))
        if url.endswith("/gateway/provider-keys/resolve"):
            attempt = {
                "attempt_id": "att-combined",
                "position": 0,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-platform-key",
                "api_base": "https://api.openai.com/v1",
                "managed": True,
            }
            return httpx.Response(
                200, json={"request_id": "req-combined", "fallback_enabled": False, "attempts": [attempt]}
            )
        if url.endswith("/gateway/code-execution/resolve"):
            return httpx.Response(200, json={"enabled": True})
        if url.endswith("/gateway/web-search/resolve"):
            return httpx.Response(200, json={"enabled": True, "authorized_tools": ["web_search", "web_fetch"]})
        return httpx.Response(204)

    control_plane_transport(answer)
    return seen


@pytest.mark.parametrize("stream", [False, True])
def test_hybrid_runs_all_four_with_each_resolve_and_report_unchanged(
    platform_client: TestClient, control_plane: _ControlPlane, backends: _Backends, stream: bool
) -> None:
    names = ["web_search", "web_fetch", "code_execution", _MCP_TOOL]
    rounds: list[Any] = (
        [_streamed_chat_round(names), _streamed_chat_answer()] if stream else [_completion(names), _completion(None)]
    )
    with _scripted("gateway.services.mcp_loop.acompletion", rounds) as sent:
        response = platform_client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": "anything",
                "messages": [{"role": "user", "content": "research and compute"}],
                "tools": [_SEARCH, _FETCH, _CODE],
                "mcp_servers": [_MCP_SERVER],
                "stream": stream,
            },
            headers={"Authorization": "Bearer user_test_token"},
        )

    assert response.status_code == 200, response.text
    assert _ran(backends) == set(names)
    assert len(sent) == 2
    assert {tool["function"]["name"] for tool in sent[0]["tools"]} == set(names)
    # Each tool's policy is resolved once, in the shape it has when declared alone.
    assert control_plane.bodies("/gateway/code-execution/resolve") == [{}]
    assert control_plane.bodies("/gateway/web-search/resolve") == [{"requested_tools": ["web_search", "web_fetch"]}]
    reports = control_plane.bodies("/gateway/usage")
    assert [(report["correlation_id"], report["status"]) for report in reports] == [("att-combined", "success")]
