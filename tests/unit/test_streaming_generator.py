"""Unit tests for the shared streaming_generator utility."""

from collections.abc import AsyncIterator

import pytest
from any_llm.types.completion import CompletionUsage

from gateway.streaming import (
    ANTHROPIC_STREAM_FORMAT,
    OPENAI_STREAM_FORMAT,
    streaming_generator,
)

_PROVIDER_CRASHED = "provider crashed"
_LOGGING_FAILED = "logging failed too"


def _format_chunk(chunk: str) -> str:
    return f"data: {chunk}\n\n"


def _extract_usage(chunk: str) -> CompletionUsage | None:
    if chunk == "usage":
        return CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return None


async def _items(*values: str) -> AsyncIterator[str]:
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_streaming_generator_success_with_usage() -> None:
    completed_usage: list[CompletionUsage] = []

    async def on_complete(usage: CompletionUsage) -> None:
        completed_usage.append(usage)

    async def on_error(error: str) -> None:
        pytest.fail("on_error should not be called")

    events: list[str] = []
    async for event in streaming_generator(
        stream=_items("hello", "usage"),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert events == ["data: hello\n\n", "data: usage\n\n", "data: [DONE]\n\n"]
    assert len(completed_usage) == 1
    assert completed_usage[0].prompt_tokens == 10
    assert completed_usage[0].completion_tokens == 5


@pytest.mark.asyncio
async def test_streaming_generator_no_usage_skips_on_complete() -> None:
    completed = False

    async def on_complete(usage: CompletionUsage) -> None:
        nonlocal completed
        completed = True

    async def on_error(error: str) -> None:
        pytest.fail("on_error should not be called")

    events: list[str] = []
    async for event in streaming_generator(
        stream=_items("hello"),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert events == ["data: hello\n\n", "data: [DONE]\n\n"]
    assert not completed


@pytest.mark.asyncio
async def test_streaming_generator_no_usage_invokes_on_no_usage() -> None:
    """When a stream finishes without usage, on_no_usage fires (F4 billing policy hook)."""
    completed = False
    no_usage_called = False

    async def on_complete(usage: CompletionUsage) -> None:
        nonlocal completed
        completed = True

    async def on_error(error: str) -> None:
        pytest.fail("on_error should not be called")

    async def on_no_usage() -> None:
        nonlocal no_usage_called
        no_usage_called = True

    events: list[str] = []
    async for event in streaming_generator(
        stream=_items("hello"),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
        on_no_usage=on_no_usage,
    ):
        events.append(event)

    assert events == ["data: hello\n\n", "data: [DONE]\n\n"]
    assert not completed
    assert no_usage_called


@pytest.mark.asyncio
async def test_streaming_generator_on_incomplete_on_client_disconnect() -> None:
    """Closing the generator mid-stream (client disconnect) fires on_incomplete, not complete/error."""
    settled: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        settled.append("complete")

    async def on_error(error: str) -> None:
        settled.append("error")

    async def on_no_usage() -> None:
        settled.append("no_usage")

    async def on_incomplete() -> None:
        settled.append("incomplete")

    async def _infinite() -> AsyncIterator[str]:
        i = 0
        while True:
            yield f"chunk-{i}"
            i += 1

    gen = streaming_generator(
        stream=_infinite(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
        on_no_usage=on_no_usage,
        on_incomplete=on_incomplete,
    )

    first = await gen.__anext__()
    assert first == "data: chunk-0\n\n"
    await gen.aclose()  # simulate the client hanging up mid-stream

    assert settled == ["incomplete"]


@pytest.mark.asyncio
async def test_streaming_generator_error_openai_format() -> None:
    error_logged: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(error: str) -> None:
        error_logged.append(error)

    async def _failing_stream() -> AsyncIterator[str]:
        yield "hello"
        raise RuntimeError(_PROVIDER_CRASHED)

    events: list[str] = []
    async for event in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert events[0] == "data: hello\n\n"
    assert "server_error" in events[1]
    assert events[2] == "data: [DONE]\n\n"
    assert error_logged == [_PROVIDER_CRASHED]


@pytest.mark.asyncio
async def test_streaming_generator_error_anthropic_format() -> None:
    error_logged: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(error: str) -> None:
        error_logged.append(error)

    async def _failing_stream() -> AsyncIterator[str]:
        raise RuntimeError(_PROVIDER_CRASHED)
        yield  # pragma: no cover

    events: list[str] = []
    async for event in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=ANTHROPIC_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert len(events) == 1
    assert "api_error" in events[0]
    assert events[0].startswith("event: error\n")
    assert error_logged == [_PROVIDER_CRASHED]


@pytest.mark.asyncio
async def test_streaming_generator_error_logging_failure_is_swallowed() -> None:
    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(error: str) -> None:
        raise RuntimeError(_LOGGING_FAILED)

    async def _failing_stream() -> AsyncIterator[str]:
        raise RuntimeError(_PROVIDER_CRASHED)
        yield  # pragma: no cover

    events: list[str] = []
    async for event in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert "server_error" in events[0]
    assert events[1] == "data: [DONE]\n\n"
