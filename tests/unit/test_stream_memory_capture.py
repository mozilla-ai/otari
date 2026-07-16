"""Unit tests for streaming memory capture: the wrapper accumulates assistant text and
fires a best-effort remember on normal completion (but not on mid-stream disconnect)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from gateway.api.routes import chat as chat_module
from gateway.api.routes._pipeline import _stream_with_memory_capture
from gateway.api.routes.chat import _chat_chunk_text, _streaming_memory_remember


def _identity(chunk: Any) -> Any:
    return chunk


async def _gen(pieces: list[Any]) -> AsyncIterator[Any]:
    for piece in pieces:
        yield piece


@pytest.mark.asyncio
async def test_capture_remembers_accumulated_text_on_complete() -> None:
    captured: list[str] = []

    async def on_settled(text: str) -> None:
        captured.append(text)

    out = [chunk async for chunk in _stream_with_memory_capture(_gen(["Hel", "", "lo"]), _identity, on_settled)]
    await asyncio.sleep(0)  # let the fire-and-forget task run

    assert out == ["Hel", "", "lo"]  # chunks pass through unchanged
    assert captured == ["Hello"]


@pytest.mark.asyncio
async def test_capture_skips_remember_on_disconnect() -> None:
    captured: list[str] = []

    async def on_settled(text: str) -> None:
        captured.append(text)

    gen = cast(
        "AsyncGenerator[Any, None]",
        _stream_with_memory_capture(_gen(["Hi", " there"]), _identity, on_settled),
    )
    assert await gen.__anext__() == "Hi"
    await gen.aclose()  # client disconnect before the stream completes
    await asyncio.sleep(0)

    assert captured == []


@pytest.mark.asyncio
async def test_capture_swallows_settled_callback_error() -> None:
    async def on_settled(text: str) -> None:
        raise RuntimeError("memory backend exploded")

    # A raised settled callback must not surface as an unretrieved-task-exception
    # or otherwise disrupt the (already drained) stream.
    out = [chunk async for chunk in _stream_with_memory_capture(_gen(["Hi"]), _identity, on_settled)]
    await asyncio.sleep(0)  # let the detached task run and its done-callback fire

    assert out == ["Hi"]


@pytest.mark.asyncio
async def test_capture_no_remember_when_no_text() -> None:
    captured: list[str] = []

    async def on_settled(text: str) -> None:
        captured.append(text)

    async for _ in _stream_with_memory_capture(_gen([None, None]), _identity, on_settled):
        pass
    await asyncio.sleep(0)

    assert captured == []


def _chunk(content: str | None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def test_chat_chunk_text_extracts_delta_content() -> None:
    assert _chat_chunk_text(_chunk("hi")) == "hi"
    assert _chat_chunk_text(_chunk(None)) is None
    assert _chat_chunk_text(SimpleNamespace(choices=[])) is None  # no choices


def test_streaming_remember_none_when_no_token() -> None:
    assert _streaming_memory_remember(MagicMock(), None, []) is None


@pytest.mark.asyncio
async def test_streaming_remember_closure_calls_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[dict[str, str]]]] = []

    async def fake_remember(config: Any, token: str, messages: list[dict[str, str]]) -> None:
        calls.append((token, messages))

    monkeypatch.setattr(chat_module, "_remember_platform_memory", fake_remember)

    callback = _streaming_memory_remember(MagicMock(), "tk", [{"role": "user", "content": "My name is X"}])
    assert callback is not None
    await callback("Hello X")

    assert calls == [
        (
            "tk",
            [
                {"role": "user", "content": "My name is X"},
                {"role": "assistant", "content": "Hello X"},
            ],
        )
    ]
