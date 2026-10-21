"""Unit tests for the shared streaming_generator utility."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

import anyio
import anyio.lowlevel
import pytest
from any_llm.types.completion import CompletionUsage
from sqlalchemy.exc import OperationalError

from gateway.core.database import describe_database_error
from gateway.log_config import logger as gateway_logger
from gateway.streaming import (
    _MAX_TERMINAL_BUFFER_CHUNKS,
    ANTHROPIC_STREAM_FORMAT,
    OPENAI_STREAM_FORMAT,
    RESPONSES_STREAM_FORMAT,
    IncompleteStreamError,
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
async def test_streaming_generator_renders_the_error_for_the_exception_that_ended_it() -> None:
    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(exc: BaseException) -> None:
        return None

    crash = RuntimeError(_PROVIDER_CRASHED)
    rendered_for: list[BaseException] = []

    def error_payload(exc: BaseException) -> str:
        rendered_for.append(exc)
        return "event: error\ndata: {}\n\n"

    async def _failing_stream() -> AsyncIterator[str]:
        raise crash
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
            error_payload=error_payload,
        )
    ]

    assert events == ["event: error\ndata: {}\n\n"]
    assert rendered_for == [crash]


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


async def _noop_no_usage() -> None:
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


@pytest.mark.asyncio
async def test_on_first_chunk_fires_once_on_the_first_real_chunk() -> None:
    """``on_first_chunk`` fires exactly once, for a multi-chunk stream."""
    calls = 0

    def on_first_chunk() -> None:
        nonlocal calls
        calls += 1

    events = [
        event
        async for event in streaming_generator(
            stream=_items("hello", "world", "usage"),
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_noop_complete,
            on_error=_fail_on_error,
            label="test:model",
            on_first_chunk=on_first_chunk,
        )
    ]

    assert events == ["data: hello\n\n", "data: world\n\n", "data: usage\n\n", "data: [DONE]\n\n"]
    assert calls == 1


@pytest.mark.asyncio
async def test_on_first_chunk_is_not_triggered_by_keepalives() -> None:
    """A keepalive is transport filler, not a chunk: it must not count as TTFT."""
    release = asyncio.Event()
    calls = 0

    def on_first_chunk() -> None:
        nonlocal calls
        calls += 1

    async def _slow_first_chunk() -> AsyncIterator[str]:
        await release.wait()
        yield "hello"

    gen = streaming_generator(
        stream=_slow_first_chunk(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
        on_first_chunk=on_first_chunk,
    )

    assert await gen.__anext__() == ": keepalive\n\n"
    assert await gen.__anext__() == ": keepalive\n\n"
    assert calls == 0
    release.set()
    assert await gen.__anext__() == "data: hello\n\n"
    assert calls == 1


@pytest.mark.asyncio
async def test_on_first_chunk_does_not_fire_when_extract_usage_raises_on_first_chunk() -> None:
    """If the first chunk is never actually formatted and sent, TTFT must not
    be recorded for it: on_first_chunk previously fired before extract_usage
    ran, so a crash there still counted a chunk the client never received."""
    calls = 0
    errors: list[str] = []

    def on_first_chunk() -> None:
        nonlocal calls
        calls += 1

    def _raise_on_first(chunk: str) -> CompletionUsage | None:
        raise ValueError(_PROVIDER_CRASHED)

    async def on_error(exc: BaseException) -> None:
        errors.append(str(exc))

    events = [
        event
        async for event in streaming_generator(
            stream=_items("hello"),
            format_chunk=_format_chunk,
            extract_usage=_raise_on_first,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_noop_complete,
            on_error=on_error,
            label="test:model",
            on_first_chunk=on_first_chunk,
        )
    ]

    assert errors == [_PROVIDER_CRASHED]
    assert calls == 0
    assert not any(event.startswith("data: hello") for event in events)


@pytest.mark.asyncio
async def test_on_first_chunk_fires_on_flush_not_on_buffering() -> None:
    """A chunk held in the terminal buffer (a cost carrier) has not reached the
    client yet: on_first_chunk must fire when it is actually flushed and
    yielded, not when it is merely pulled off the upstream and buffered."""
    release = asyncio.Event()
    calls = 0

    def on_first_chunk() -> None:
        nonlocal calls
        calls += 1

    def _is_carrier(chunk: str) -> bool:
        return chunk == "carrier"

    async def _carrier_then_trailing() -> AsyncIterator[str]:
        yield "carrier"
        await release.wait()
        yield "trailing"

    gen = streaming_generator(
        stream=_carrier_then_trailing(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        on_no_usage=_noop_no_usage,
        label="test:model",
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
        is_cost_carrier=_is_carrier,
        on_first_chunk=on_first_chunk,
    )

    # "carrier" is pulled off the upstream immediately and buffered (it is a
    # cost carrier), so nothing has actually reached
    # the client yet: on_first_chunk must not have fired even though a real
    # chunk was already read.
    assert await gen.__anext__() == ": keepalive\n\n"
    assert calls == 0

    release.set()
    # "trailing" also gets buffered (buffering_terminal stays true), the
    # stream then ends and the terminal buffer flushes both chunks: this is
    # the first point anything is actually sent, and where TTFT must land.
    assert await gen.__anext__() == "data: carrier\n\n"
    assert calls == 1
    assert await gen.__anext__() == "data: trailing\n\n"
    assert calls == 1
    assert await gen.__anext__() == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_usage_settles_before_the_last_cost_carrier_is_sent_and_is_attached_to_it() -> None:
    """An earlier carrier is flushed when a later one arrives; the last one waits for the settlement."""
    log: list[str] = []

    def format_chunk(chunk: str) -> str:
        log.append(f"sent {chunk}")
        return _format_chunk(chunk)

    def extract_usage(chunk: str) -> CompletionUsage | None:
        return _extract_usage("usage") if chunk == "carrier-2" else None

    async def on_complete(usage: CompletionUsage) -> str:
        log.append("settled")
        return "cost"

    def attach_settlement(carrier: str, settlement: str) -> bool:
        log.append(f"attached {settlement} to {carrier}")
        return True

    events = [
        event
        async for event in streaming_generator(
            stream=_items("a", "carrier-1", "b", "carrier-2", "tail"),
            format_chunk=format_chunk,
            extract_usage=extract_usage,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=_fail_on_error,
            label="test:model",
            is_cost_carrier=lambda chunk: chunk.startswith("carrier"),
            attach_settlement=attach_settlement,
        )
    ]

    assert events == [_format_chunk(chunk) for chunk in ("a", "carrier-1", "b", "carrier-2", "tail")] + [
        OPENAI_STREAM_FORMAT.done_marker
    ]
    assert log == [
        "sent a",
        "sent carrier-1",
        "sent b",
        "settled",
        "attached cost to carrier-2",
        "sent carrier-2",
        "sent tail",
    ]


@pytest.mark.asyncio
async def test_a_carrier_followed_by_more_than_a_terminal_suffix_is_not_held() -> None:
    """Past the buffer's bound the carrier was not terminal, so the stream resumes before it ends."""
    release = asyncio.Event()
    fillers = [f"filler-{i}" for i in range(_MAX_TERMINAL_BUFFER_CHUNKS + 1)]
    attached: list[object] = []

    async def _carrier_then_more() -> AsyncIterator[str]:
        yield "carrier"
        for filler in fillers:
            yield filler
        await release.wait()
        yield "usage"

    async def on_complete(usage: CompletionUsage) -> str:
        return "cost"

    def attach_settlement(carrier: object, settlement: str) -> bool:
        attached.append(carrier)
        return False

    gen = streaming_generator(
        stream=_carrier_then_more(),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=_fail_on_error,
        label="test:model",
        is_cost_carrier=lambda chunk: chunk == "carrier",
        attach_settlement=attach_settlement,
    )

    held = [await gen.__anext__() for _ in range(len(fillers) + 1)]
    assert held == [_format_chunk(chunk) for chunk in ("carrier", *fillers)]
    release.set()
    assert [event async for event in gen] == [_format_chunk("usage"), OPENAI_STREAM_FORMAT.done_marker]
    assert attached == [None], "a released carrier is not the one the settlement is attached to"


def _is_stop(chunk: str) -> bool:
    return chunk == "stop"


async def _run_with_completion_check(
    *chunks: str, carries_cost: bool = False
) -> tuple[list[str], list[BaseException], list[str]]:
    """Run ``chunks`` through the generator with ``"stop"`` as the completing chunk.

    ``carries_cost`` makes ``"usage"`` a cost carrier, which the generator holds until the stream settles.
    """
    errors: list[BaseException] = []
    settled: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        settled.append("complete")

    async def on_error(exc: BaseException) -> None:
        errors.append(exc)

    async def on_no_usage() -> None:
        settled.append("no_usage")

    async def on_incomplete() -> None:
        settled.append("incomplete")

    events = [
        event
        async for event in streaming_generator(
            stream=_items(*chunks),
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=ANTHROPIC_STREAM_FORMAT,
            on_complete=on_complete,
            on_error=on_error,
            label="test:model",
            on_no_usage=on_no_usage,
            on_incomplete=on_incomplete,
            is_cost_carrier=(lambda chunk: chunk == "usage") if carries_cost else None,
            completes_stream=_is_stop,
        )
    ]
    return events, errors, settled


@pytest.mark.asyncio
async def test_an_upstream_that_ends_without_any_chunk_fails_the_stream() -> None:
    """An empty upstream used to close as a success: a 200 carrying only the done marker."""
    events, errors, settled = await _run_with_completion_check()

    assert events == [ANTHROPIC_STREAM_FORMAT.error_payload]
    [error] = errors
    assert isinstance(error, IncompleteStreamError)
    assert error.chunks_received == 0
    assert settled == [], "the error path settles; success and no-usage settlement must not also run"


@pytest.mark.asyncio
@pytest.mark.parametrize("carries_cost", [False, True])
async def test_an_upstream_that_stops_before_its_completing_chunk_fails_after_what_it_sent(
    carries_cost: bool,
) -> None:
    events, errors, settled = await _run_with_completion_check("hello", "usage", carries_cost=carries_cost)

    assert events == ["data: hello\n\n", "data: usage\n\n", ANTHROPIC_STREAM_FORMAT.error_payload]
    [error] = errors
    assert isinstance(error, IncompleteStreamError)
    assert error.chunks_received == 2
    assert settled == []


@pytest.mark.asyncio
async def test_a_stream_that_reaches_its_completing_chunk_is_unaffected_by_the_check() -> None:
    events, errors, settled = await _run_with_completion_check("hello", "usage", "stop")

    assert events == ["data: hello\n\n", "data: usage\n\n", "data: stop\n\n", ANTHROPIC_STREAM_FORMAT.done_marker]
    assert errors == []
    assert settled == ["complete"]


class _ClosableStream:
    """An endless stream recording whether it was closed.

    Reading a chunk and closing both suspend, as a provider's network calls do, so a cancellation can land in either.
    """

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> "_ClosableStream":
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0)
        return "chunk"

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


@pytest.mark.asyncio
async def test_a_stream_the_client_abandons_closes_its_source() -> None:
    """The source is closed on teardown, so the provider's connection does not wait for garbage collection."""
    source = _ClosableStream()
    gen = streaming_generator(
        stream=source,
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
    )

    assert await gen.__anext__() == "data: chunk\n\n"
    await gen.aclose()

    assert source.closed


def _settles_after_one_await(log: list[str], name: str) -> Callable[..., Awaitable[None]]:
    """A settlement callback that suspends before it finishes, as a database write does."""

    async def settle(*_: object) -> None:
        log.append(f"{name} started")
        await asyncio.sleep(0)
        log.append(f"{name} finished")

    return settle


async def _read_until_the_client_leaves(
    gen: AsyncIterator[str], leaves: Callable[[list[str]], bool]
) -> list[type[BaseException]]:
    """Read ``gen`` as Starlette does, canceling its task group once ``leaves`` accepts the events read so far.

    Starlette answers a disconnect that way, and the cancellation is redelivered at every await until the task leaves
    the group. ``aclose()`` throws one ``GeneratorExit`` instead, so it cannot show whether settling survives that.
    Returns what the reading task raised.
    """
    observed: list[type[BaseException]] = []

    async def stream_response(scope: anyio.CancelScope) -> None:
        received: list[str] = []
        try:
            async for event in gen:
                received.append(event)
                if leaves(received):
                    scope.cancel()
            # Starlette's closing send, where a cancellation the generator did not raise surfaces.
            await anyio.lowlevel.checkpoint()
        except BaseException as exc:
            observed.append(type(exc))
            raise

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(stream_response, task_group.cancel_scope)
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("keepalive_interval", [0.0, _KEEPALIVE_INTERVAL], ids=["direct", "keepalive"])
async def test_a_stream_canceled_by_a_client_disconnect_settles_once_and_closes_its_source(
    keepalive_interval: float,
) -> None:
    source = _ClosableStream()
    settlement: list[str] = []
    gen = streaming_generator(
        stream=source,
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        on_incomplete=_settles_after_one_await(settlement, "incomplete"),
        keepalive_interval_seconds=keepalive_interval,
    )

    observed = await _read_until_the_client_leaves(gen, lambda received: len(received) == 5)

    assert settlement == ["incomplete started", "incomplete finished"]
    assert source.closed
    assert observed == [asyncio.CancelledError]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "last_event_read", "callback"),
    [
        (("hello", "usage"), "data: usage\n\n", "complete"),
        (("hello",), "data: hello\n\n", "no_usage"),
    ],
    ids=["with-usage", "without-usage"],
)
async def test_a_completed_stream_whose_client_leaves_while_it_settles_still_settles(
    chunks: tuple[str, ...], last_event_read: str, callback: str
) -> None:
    settlement: list[str] = []
    gen = streaming_generator(
        stream=_items(*chunks),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_settles_after_one_await(settlement, "complete"),
        on_error=_fail_on_error,
        label="test:model",
        on_no_usage=_settles_after_one_await(settlement, "no_usage"),
        on_incomplete=_settles_after_one_await(settlement, "incomplete"),
    )

    observed = await _read_until_the_client_leaves(gen, lambda received: received[-1] == last_event_read)

    assert settlement == [f"{callback} started", f"{callback} finished"]
    assert observed == [asyncio.CancelledError]


@pytest.mark.asyncio
async def test_a_failed_stream_whose_client_leaves_while_it_settles_still_settles() -> None:
    async def fails_after_one_chunk() -> AsyncIterator[str]:
        yield "hello"
        raise _StatusCodeError(503)

    settlement: list[str] = []
    gen = streaming_generator(
        stream=fails_after_one_chunk(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=ANTHROPIC_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_settles_after_one_await(settlement, "error"),
        label="test:model",
        on_incomplete=_settles_after_one_await(settlement, "incomplete"),
    )

    observed = await _read_until_the_client_leaves(
        gen, lambda received: received[-1] == ANTHROPIC_STREAM_FORMAT.error_payload
    )

    assert settlement == ["error started", "error finished"]
    assert observed == [asyncio.CancelledError]


@pytest.mark.asyncio
async def test_an_inline_settlement_the_client_leaves_during_still_finishes() -> None:
    """The client leaves on a keepalive sent while the settlement is pending."""
    settlement: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        settlement.append("complete started")
        await asyncio.sleep(_KEEPALIVE_INTERVAL * 5)
        settlement.append("complete finished")

    gen = streaming_generator(
        stream=_items("hello", "usage"),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=_fail_on_error,
        label="test:model",
        on_incomplete=_settles_after_one_await(settlement, "incomplete"),
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
    )

    observed = await _read_until_the_client_leaves(gen, lambda received: received[-1] == OPENAI_STREAM_FORMAT.keepalive)

    assert settlement == ["complete started", "complete finished"]
    assert observed == [asyncio.CancelledError]


# ---------------------------------------------------------------------------
# Settlement failures in the log
# ---------------------------------------------------------------------------

_ROW_DATA = "sk-live-do-not-log"
_DATABASE_FAILURE = OperationalError("UPDATE usage_logs", {}, Exception(f"DETAIL: Failing row contains ({_ROW_DATA})"))


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _fails_with(exc: Exception, after_seconds: float = 0.0) -> Callable[..., Awaitable[None]]:
    async def fail(*_: object) -> None:
        await asyncio.sleep(after_seconds)
        raise exc

    return fail


async def _settle_inline(fail: Callable[..., Awaitable[None]]) -> None:
    async for _event in streaming_generator(
        stream=_items("hello", "usage"),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=fail,
        on_error=_fail_on_error,
        label="test:model",
    ):
        pass


async def _settle_inline_after_the_client_left(fail: Callable[..., Awaitable[None]]) -> None:
    gen = streaming_generator(
        stream=_items("hello", "usage"),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=fail,
        on_error=_fail_on_error,
        label="test:model",
        keepalive_interval_seconds=_KEEPALIVE_INTERVAL,
    )
    await _read_until_the_client_leaves(gen, lambda received: received[-1] == OPENAI_STREAM_FORMAT.keepalive)


async def _settle_an_upstream_error(fail: Callable[..., Awaitable[None]]) -> None:
    async def fails_after_one_chunk() -> AsyncIterator[str]:
        yield "hello"
        raise _StatusCodeError(503)

    async for _event in streaming_generator(
        stream=fails_after_one_chunk(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=fail,
        label="test:model",
    ):
        pass


async def _settle_an_abandoned_stream(fail: Callable[..., Awaitable[None]]) -> None:
    gen = streaming_generator(
        stream=_ClosableStream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        on_incomplete=fail,
    )
    await gen.__anext__()
    await gen.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settle", "settles_after_seconds"),
    [
        (_settle_inline, 0.0),
        (_settle_inline_after_the_client_left, _KEEPALIVE_INTERVAL * 5),
        (_settle_an_upstream_error, 0.0),
        (_settle_an_abandoned_stream, 0.0),
    ],
    ids=["inline", "inline-after-the-client-left", "upstream-error", "abandoned"],
)
@pytest.mark.parametrize(
    ("failure", "logged_as"),
    [
        (_DATABASE_FAILURE, describe_database_error(_DATABASE_FAILURE)),
        (RuntimeError(f"could not write the row for {_ROW_DATA}"), "RuntimeError"),
    ],
    ids=["database-error", "other-error"],
)
async def test_a_settlement_failure_is_logged_without_its_message(
    gateway_logs: pytest.LogCaptureFixture,
    settle: Callable[[Callable[..., Awaitable[None]]], Awaitable[None]],
    settles_after_seconds: float,
    failure: Exception,
    logged_as: str,
) -> None:
    """A database error's message carries its bound parameters and the row it touched."""
    await settle(_fails_with(failure, after_seconds=settles_after_seconds))

    [record] = [r for r in gateway_logs.records if r.getMessage().startswith("Failed to")]
    assert record.getMessage().endswith(f": {logged_as}")
    assert _ROW_DATA not in gateway_logs.text


async def _consume_hanging_up_after(gen: AsyncIterator[str], hang_up: str) -> None:
    """Read ``gen`` as Starlette does, ending the task scope once ``hang_up`` is read.

    A disconnect cancels the scope the response is streamed in, and anyio then cancels
    every await left inside it, including whatever the generator still has to do.
    """

    async def _send() -> None:
        async for event in gen:
            if event == hang_up:
                tg.cancel_scope.cancel()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_send)


async def _database_write(settled: list[str], name: str) -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    settled.append(name)


async def _then_raises(*values: str) -> AsyncIterator[str]:
    for value in values:
        yield value
    raise RuntimeError(_PROVIDER_CRASHED)


@pytest.mark.asyncio
async def test_a_client_that_hangs_up_mid_stream_is_still_billed() -> None:
    settled: list[str] = []

    async def _waits_on_the_provider() -> AsyncIterator[str]:
        yield "chunk"
        await asyncio.Event().wait()
        yield "never"

    gen = streaming_generator(
        stream=_waits_on_the_provider(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=_fail_on_error,
        label="test:model",
        on_incomplete=lambda: _database_write(settled, "incomplete"),
    )
    await _consume_hanging_up_after(gen, "data: chunk\n\n")

    assert settled == ["incomplete"]


@pytest.mark.asyncio
async def test_a_client_that_hangs_up_on_the_error_event_is_still_billed() -> None:
    """Claude Code's ``WebSearch`` hangs up on the error that ends a tool loop at its cap."""
    settled: list[str] = []

    gen = streaming_generator(
        stream=_then_raises("chunk"),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=ANTHROPIC_STREAM_FORMAT,
        on_complete=_noop_complete,
        on_error=lambda _exc: _database_write(settled, "error"),
        label="test:model",
    )
    await _consume_hanging_up_after(gen, ANTHROPIC_STREAM_FORMAT.error_payload)

    assert settled == ["error"]
