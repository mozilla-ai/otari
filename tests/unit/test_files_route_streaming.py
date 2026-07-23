"""Unit tests for the ``/v1/files`` route's streaming helpers.

Covers ``_prime`` in isolation (no FastAPI app/DB needed): it has no
dependency on request/db/config, so it's tested directly rather than through
the full HTTP stack (see tests/integration/test_files_endpoint.py for that).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from gateway.api.routes.files import _prime


async def _iter(chunks: list[bytes]) -> AsyncGenerator[bytes, None]:
    for chunk in chunks:
        yield chunk


async def _collect(chunks: AsyncGenerator[bytes, None]) -> bytes:
    collected = bytearray()
    async for chunk in chunks:
        collected.extend(chunk)
    return bytes(collected)


@pytest.mark.asyncio
async def test_prime_passes_through_all_chunks() -> None:
    primed = await _prime(_iter([b"a", b"b", b"c"]))
    assert await _collect(primed) == b"abc"


@pytest.mark.asyncio
async def test_prime_handles_empty_source() -> None:
    primed = await _prime(_iter([]))
    assert await _collect(primed) == b""


@pytest.mark.asyncio
async def test_prime_raises_before_returning_on_immediate_failure() -> None:
    """The whole point of _prime: a failure on the first chunk raises here,
    in the route, before StreamingResponse ever gets a body iterator, so it
    can still become a clean error response instead of a truncated 200.
    """

    async def _broken() -> AsyncGenerator[bytes, None]:
        msg = "blob missing or unreadable"
        raise OSError(msg)
        yield b""  # pragma: no cover - unreachable, keeps this an async generator

    with pytest.raises(OSError, match="blob missing or unreadable"):
        await _prime(_broken())


@pytest.mark.asyncio
async def test_prime_does_not_hide_failures_after_the_first_chunk() -> None:
    """A failure past the first chunk still isn't caught by _prime (it only
    primes the first item) - it must still propagate once StreamingResponse
    iterates the rest, same as before this helper existed.
    """

    async def _fails_on_second_chunk() -> AsyncGenerator[bytes, None]:
        yield b"ok first chunk"
        msg = "disk error on second read"
        raise OSError(msg)

    primed = await _prime(_fails_on_second_chunk())
    with pytest.raises(OSError, match="disk error on second read"):
        await _collect(primed)


@pytest.mark.asyncio
async def test_prime_closes_inner_source_on_early_close() -> None:
    """Simulates a client disconnect mid-download.

    StreamingResponse closes the outer (primed) generator when the client
    goes away before the body finishes. That must propagate to closing the
    inner source too, or LocalDirFileStore's open file handle (held inside
    get_stream's _open_handle context manager) would stay open until GC
    eventually gets around to the abandoned generator.
    """
    closed = {"value": False}

    async def _source() -> AsyncGenerator[bytes, None]:
        try:
            yield b"first"
            yield b"second"
        finally:
            closed["value"] = True

    primed = await _prime(_source())
    first = await primed.__anext__()
    assert first == b"first"
    assert closed["value"] is False  # not yet, only the first chunk was read

    await primed.aclose()
    assert closed["value"] is True
