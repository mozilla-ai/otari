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
