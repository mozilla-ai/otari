"""The Playground's chat stream reports gateway-run tool calls; the public one does not."""

from __future__ import annotations

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


async def _run(adapter: chat_routes._ChatAdapter, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
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
    activity: list[dict[str, Any]] = []
    async for chunk in adapter.open_tool_loop_stream({"model": "fake", "messages": []}, _Pool(), 5):
        found = getattr(chunk.choices[0].delta, TOOL_ACTIVITY_FIELD, None) if chunk.choices else None
        if found is not None:
            activity.append(found)
    return activity


@pytest.mark.asyncio
async def test_the_playground_adapter_reports_gateway_tool_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    activity = await _run(chat_routes._PLAYGROUND_ADAPTER, monkeypatch)
    assert [record["type"] for record in activity] == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_the_public_adapter_keeps_gateway_tool_activity_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _run(chat_routes._ADAPTER, monkeypatch) == []
