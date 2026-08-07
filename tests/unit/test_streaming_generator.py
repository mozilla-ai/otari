"""Unit tests for the shared streaming_generator utility."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from any_llm.types.completion import CompletionUsage

from gateway.streaming import (
    ANTHROPIC_STREAM_FORMAT,
    OPENAI_STREAM_FORMAT,
    RESPONSES_STREAM_FORMAT,
    streaming_generator,
)

_PROVIDER_CRASHED = "provider crashed"
_LOGGING_FAILED = "logging failed too"
_KEEPALIVE_INTERVAL = 0.01


class _StatusCodeError(Exception):
    """Upstream failure carrying an HTTP status, as the provider SDKs raise."""

    def __init__(self, status_code: int) -> None:
        super().__init__(_PROVIDER_CRASHED)
        self.status_code = status_code


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

    async def on_error(exc: BaseException) -> None:
        pytest.fail("on_error should not be called")

    events = [
        event
        async for event in streaming_generator(
            stream=_items("hello", "usage"),
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
        )
    ]

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

    async def on_error(exc: BaseException) -> None:
        pytest.fail("on_error should not be called")

    events = [
        event
        async for event in streaming_generator(
            stream=_items("hello"),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
        )
    ]

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

    async def on_error(exc: BaseException) -> None:
        pytest.fail("on_error should not be called")

    async def on_no_usage() -> None:
        nonlocal no_usage_called
        no_usage_called = True

    events = [
        event
        async for event in streaming_generator(
            stream=_items("hello"),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
            on_no_usage=on_no_usage,
        )
    ]

    assert events == ["data: hello\n\n", "data: [DONE]\n\n"]
    assert not completed
    assert no_usage_called


@pytest.mark.asyncio
async def test_streaming_generator_on_incomplete_on_client_disconnect() -> None:
    """Closing the generator mid-stream (client disconnect) fires on_incomplete, not complete/error."""
    settled: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        settled.append("complete")

    async def on_error(exc: BaseException) -> None:
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

    async def on_error(exc: BaseException) -> None:
        error_logged.append(str(exc))

    async def _failing_stream() -> AsyncIterator[str]:
        yield "hello"
        raise RuntimeError(_PROVIDER_CRASHED)

    events = [
        event
        async for event in streaming_generator(
            stream=_failing_stream(),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
        )
    ]

    assert events[0] == "data: hello\n\n"
    assert "server_error" in events[1]
    assert events[2] == "data: [DONE]\n\n"
    assert error_logged == [_PROVIDER_CRASHED]


@pytest.mark.asyncio
async def test_streaming_generator_hands_the_exception_to_on_error() -> None:
    """on_error receives the raised exception itself, not a rendered message.

    The settlement callback classifies it (recording the upstream HTTP status on
    the usage log, see ``failure_status_code``), which a string cannot support:
    provider error prose carries no reliable status. Pinned separately from the
    format tests above, which only compare ``str(exc)`` and so would pass either
    way.
    """
    raised = _StatusCodeError(429)
    received: list[BaseException] = []

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(exc: BaseException) -> None:
        received.append(exc)

    async def _failing_stream() -> AsyncIterator[str]:
        raise raised
        yield  # pragma: no cover

    async for _ in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        pass

    assert received == [raised]
    assert getattr(received[0], "status_code", None) == 429


@pytest.mark.asyncio
async def test_streaming_generator_error_anthropic_format() -> None:
    error_logged: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(exc: BaseException) -> None:
        error_logged.append(str(exc))

    async def _failing_stream() -> AsyncIterator[str]:
        raise RuntimeError(_PROVIDER_CRASHED)
        yield  # pragma: no cover

    events = [
        event
        async for event in streaming_generator(
            stream=_failing_stream(),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=ANTHROPIC_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
        )
    ]

    assert len(events) == 1
    assert "api_error" in events[0]
    assert events[0].startswith("event: error\n")
    assert error_logged == [_PROVIDER_CRASHED]


@pytest.mark.asyncio
async def test_streaming_generator_error_logging_failure_is_swallowed() -> None:
    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(exc: BaseException) -> None:
        raise RuntimeError(_LOGGING_FAILED)

    async def _failing_stream() -> AsyncIterator[str]:
        raise RuntimeError(_PROVIDER_CRASHED)
        yield  # pragma: no cover

    events = [
        event
        async for event in streaming_generator(
            stream=_failing_stream(),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
        )
    ]

    assert "server_error" in events[0]
    assert events[1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Transport keepalives (issue #527)
# ---------------------------------------------------------------------------


async def _noop_complete(usage: CompletionUsage) -> None:
    return None


async def _fail_on_error(exc: BaseException) -> None:
    pytest.fail(f"on_error should not be called: {exc}")


@pytest.mark.parametrize(
    ("fmt", "expected_keepalive"),
    [
        pytest.param(OPENAI_STREAM_FORMAT, ": keepalive\n\n", id="chat"),
        pytest.param(RESPONSES_STREAM_FORMAT, ": keepalive\n\n", id="responses"),
        pytest.param(ANTHROPIC_STREAM_FORMAT, 'event: ping\ndata: {"type": "ping"}\n\n', id="messages"),
    ],
)
@pytest.mark.asyncio
async def test_keepalives_are_emitted_while_awaiting_the_first_chunk(fmt: object, expected_keepalive: str) -> None:
    """A slow time-to-first-token gets keepalive frames, then the real chunk.

    Also pins that the held ``__anext__`` survives the keepalives: the chunk that
    lands after two idle intervals is still delivered, which it would not be if
    the wait were cancelled and restarted on each timeout.
    """
    release = asyncio.Event()
    settled: list[str] = []

    async def _slow_first_chunk() -> AsyncIterator[str]:
        await release.wait()
        yield "usage"

    async def on_complete(usage: CompletionUsage) -> None:
        settled.append("complete")

    gen = streaming_generator(
        stream=_slow_first_chunk(),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=fmt,  # type: ignore[arg-type]
        on_complete=on_complete,
        on_error=_fail_on_error,
        label="test:model",
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
    )

    assert await gen.__anext__() == expected_keepalive
    assert await gen.__anext__() == expected_keepalive
    release.set()
    assert await gen.__anext__() == "data: usage\n\n"
    assert await gen.__anext__() == fmt.done_marker  # type: ignore[attr-defined]
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    assert settled == ["complete"]


@pytest.mark.asyncio
async def test_no_keepalives_when_the_interval_is_zero() -> None:
    """The disabled interval is the pre-keepalive behavior: chunks and nothing else."""

    async def _slow_first_chunk() -> AsyncIterator[str]:
        await asyncio.sleep(_KEEPALIVE_INTERVAL * 3)
        yield "hello"

    events = [
        event
        async for event in streaming_generator(
            stream=_slow_first_chunk(),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_noop_complete,
            on_error=_fail_on_error,
            label="test:model",
            keepalive_interval_seconds=0.0,
        )
    ]

    assert events == ["data: hello\n\n", "data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_keepalives_bypass_usage_extraction_and_formatting() -> None:
    """Keepalives are transport-level: they never reach usage accounting."""
    inspected: list[str] = []
    formatted: list[str] = []

    def _record_usage(chunk: str) -> CompletionUsage | None:
        inspected.append(chunk)
        return _extract_usage(chunk)

    def _record_format(chunk: str) -> str:
        formatted.append(chunk)
        return _format_chunk(chunk)

    release = asyncio.Event()

    async def _slow_first_chunk() -> AsyncIterator[str]:
        await release.wait()
        yield "usage"

    gen = streaming_generator(
        stream=_slow_first_chunk(),
        format_chunk=_record_format,
        extract_usage=_record_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
    )

    assert await gen.__anext__() == ": keepalive\n\n"
    release.set()
    async for _ in gen:
        pass

    assert inspected == ["usage"]
    assert formatted == ["usage"]


@pytest.mark.asyncio
async def test_keepalive_wait_is_cancelled_on_client_disconnect() -> None:
    """Closing the generator during a keepalive gap tears down the pending wait.

    Without the cleanup, the outstanding ``__anext__`` task outlives the request
    and keeps the upstream stream open until the event loop finalizes it.
    """
    settled: list[str] = []
    upstream: list[str] = []
    release = asyncio.Event()

    async def _slow_first_chunk() -> AsyncIterator[str]:
        try:
            await release.wait()
            yield "hello"  # pragma: no cover
        except asyncio.CancelledError:
            upstream.append("cancelled")
            raise

    async def on_incomplete() -> None:
        settled.append("incomplete")

    gen = streaming_generator(
        stream=_slow_first_chunk(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        on_incomplete=on_incomplete,
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
    )

    assert await gen.__anext__() == ": keepalive\n\n"
    await gen.aclose()  # simulate the client hanging up during the first-chunk wait

    assert settled == ["incomplete"]
    assert upstream == ["cancelled"]


@pytest.mark.asyncio
async def test_keepalives_fill_gaps_between_chunks() -> None:
    """Long provider gaps mid-stream (e.g. a tool call) keep the socket warm too."""
    release = asyncio.Event()

    async def _gapped() -> AsyncIterator[str]:
        yield "first"
        await release.wait()
        yield "second"

    gen = streaming_generator(
        stream=_gapped(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
    )

    assert await gen.__anext__() == "data: first\n\n"
    assert await gen.__anext__() == ": keepalive\n\n"
    release.set()
    assert await gen.__anext__() == "data: second\n\n"
    assert await gen.__anext__() == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_keepalives_do_not_mask_an_upstream_error() -> None:
    """An upstream that raises after an idle gap still lands in the SSE channel."""
    errors: list[str] = []

    async def _failing_after_gap() -> AsyncIterator[str]:
        await asyncio.sleep(_KEEPALIVE_INTERVAL * 2)
        raise RuntimeError(_PROVIDER_CRASHED)
        yield  # pragma: no cover

    async def on_error(exc: BaseException) -> None:
        errors.append(str(exc))

    events = [
        event
        async for event in streaming_generator(
            stream=_failing_after_gap(),
            format_chunk=_format_chunk,
            extract_usage=lambda _: None,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_noop_complete,
            on_error=on_error,
            label="test:model",
            keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
        )
    ]

    assert events[0] == ": keepalive\n\n"
    assert "server_error" in events[-2]
    assert events[-1] == "data: [DONE]\n\n"
    assert errors == [_PROVIDER_CRASHED]
