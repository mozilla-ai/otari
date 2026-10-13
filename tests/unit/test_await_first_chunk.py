"""Unit tests for ``await_first_chunk``, the wait that lets a stream fail before its status line."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from gateway.streaming import await_first_chunk


async def _stream(
    *, delay: float = 0.0, chunks: tuple[str, ...] = ("a", "b"), fail: Exception | None = None
) -> AsyncIterator[str]:
    """Yield ``chunks`` after ``delay``, then raise ``fail`` if given."""
    if delay:
        await asyncio.sleep(delay)
    for chunk in chunks:
        yield chunk
    if fail is not None:
        raise fail


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_a_stream_that_fails_before_its_first_chunk_raises_here() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'n'"):
        await await_first_chunk(_stream(chunks=(), fail=TypeError("unexpected keyword argument 'n'")), 1.0)


@pytest.mark.asyncio
async def test_the_first_chunk_is_re_emitted_before_the_rest() -> None:
    assert await _drain(await await_first_chunk(_stream(), 1.0)) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_failure_after_the_first_chunk_reaches_the_iterator() -> None:
    stream = await await_first_chunk(_stream(chunks=("a",), fail=RuntimeError("mid-stream")), 1.0)

    assert await anext(stream) == "a"
    with pytest.raises(RuntimeError, match="mid-stream"):
        await anext(stream)


@pytest.mark.asyncio
async def test_an_empty_stream_is_returned_empty() -> None:
    assert await _drain(await await_first_chunk(_stream(chunks=()), 1.0)) == []


@pytest.mark.asyncio
async def test_a_first_chunk_slower_than_the_window_is_not_lost() -> None:
    stream = await await_first_chunk(_stream(delay=0.1), 0.01)

    assert await _drain(stream) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_failure_after_the_window_reaches_the_iterator() -> None:
    stream = await await_first_chunk(_stream(delay=0.1, chunks=(), fail=RuntimeError("late")), 0.01)

    with pytest.raises(RuntimeError, match="late"):
        await _drain(stream)


@pytest.mark.asyncio
async def test_a_non_positive_window_waits_for_the_first_chunk() -> None:
    with pytest.raises(RuntimeError, match="slow failure"):
        await await_first_chunk(_stream(delay=0.05, chunks=(), fail=RuntimeError("slow failure")), 0)


class _SlowStream:
    """A stream whose reads never finish, recording whether it was closed."""

    def __init__(self) -> None:
        self.closed = False
        self.read_cancelled = asyncio.Event()

    def __aiter__(self) -> "_SlowStream":
        return self

    async def __anext__(self) -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.read_cancelled.set()
            raise
        return "never"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_cancelling_the_wait_cancels_the_read_and_closes_the_stream() -> None:
    stream = _SlowStream()
    waiting = asyncio.ensure_future(await_first_chunk(stream, 3600))
    await asyncio.sleep(0.01)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert stream.read_cancelled.is_set()
    assert stream.closed


@pytest.mark.asyncio
async def test_closing_the_iterator_after_the_window_cancels_the_pending_read() -> None:
    stream = _SlowStream()
    resumed = await await_first_chunk(stream, 0.01)
    assert not stream.read_cancelled.is_set()

    await resumed.aclose()  # type: ignore[attr-defined]

    assert stream.read_cancelled.is_set()
    assert stream.closed


class _FastStream:
    """A stream whose first read finishes at once, recording whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> "_FastStream":
        return self

    async def __anext__(self) -> str:
        return "chunk"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_closing_a_prefetched_stream_before_reading_it_closes_the_source() -> None:
    stream = _FastStream()
    prefetched = await await_first_chunk(stream, 1.0)

    await prefetched.aclose()  # type: ignore[attr-defined]

    assert stream.closed
