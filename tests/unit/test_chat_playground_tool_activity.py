"""The Playground's chat stream reports gateway-run tool calls; the public one does not."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChoiceDeltaToolCall, ChunkChoice
from any_llm.types.completion import ChoiceDeltaToolCallFunction as DeltaFn

from gateway.api.routes import chat as chat_routes
from gateway.services import mcp_loop as mcp_loop_module
from gateway.services.mcp_loop import TOOL_ACTIVITY_FIELD


class _Pool:
    openai_tools: list[dict[str, Any]] = []

    def owns_tool(self, name: str) -> bool:
        return name == "fetch_url"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return "ok"


def _chunk(delta: ChoiceDelta, finish: Any = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c",
        object="chat.completion.chunk",
        created=0,
        model="fake",
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish)],
    )


async def _chunks(*chunks: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


async def _stream(adapter: chat_routes._ChatAdapter, monkeypatch: pytest.MonkeyPatch) -> list[ChatCompletionChunk]:
    rounds = iter(
        [
            _chunks(
                _chunk(
                    ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_1",
                                type="function",
                                function=DeltaFn(name="fetch_url", arguments="{}"),
                            )
                        ]
                    )
                ),
                _chunk(ChoiceDelta(), "tool_calls"),
            ),
            _chunks(_chunk(ChoiceDelta(content="done"), "stop")),
        ]
    )

    async def fake_acompletion(**_kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return next(rounds)

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)
    return [chunk async for chunk in adapter.open_tool_loop_stream({"model": "fake", "messages": []}, _Pool(), 5)]


async def _run(adapter: chat_routes._ChatAdapter, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    chunks = await _stream(adapter, monkeypatch)
    return [found for chunk in chunks if (found := getattr(chunk, TOOL_ACTIVITY_FIELD, None)) is not None]


@pytest.mark.asyncio
async def test_the_playground_adapter_reports_gateway_tool_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    activity = await _run(chat_routes._PLAYGROUND_ADAPTER, monkeypatch)
    assert [record["type"] for record in activity] == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_the_public_adapter_keeps_gateway_tool_activity_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _run(chat_routes._ADAPTER, monkeypatch) == []


@pytest.mark.asyncio
async def test_the_playground_sse_carries_each_activity_record_once_beside_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = chat_routes._PLAYGROUND_ADAPTER
    events = [adapter.format_chunk(chunk) for chunk in await _stream(adapter, monkeypatch)]
    carrying = [event for event in events if TOOL_ACTIVITY_FIELD in event]
    assert len(carrying) == 2
    for event in carrying:
        assert event.count(TOOL_ACTIVITY_FIELD) == 1
        body = json.loads(event.removeprefix("data: "))
        assert body[TOOL_ACTIVITY_FIELD]["type"] in {"tool_call", "tool_result"}
        assert body["choices"][0]["delta"].get("provider_specific_fields") is None
