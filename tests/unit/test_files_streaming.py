"""Unit tests for streaming a stored file back.

Covers the route's declared binary response, and ``_primed`` in isolation: it
depends on nothing but its source iterator, so it is tested directly rather
than through the full HTTP stack (see tests/integration/test_files_endpoint.py
for that).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI

from gateway.api.routes.files import router
from gateway.core.config import API_ROOT
from gateway.services.files._service import _primed


def test_download_openapi_describes_binary_content() -> None:
    app = FastAPI()
    app.include_router(router, prefix=API_ROOT)
    responses = app.openapi()["paths"][f"{API_ROOT}/files/{{file_id}}/content"]["get"]["responses"]

    assert responses["200"]["content"] == {
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        "*/*": {"schema": {"type": "string", "format": "binary"}},
    }
    assert responses["200"]["headers"]["Content-Disposition"]["schema"]["type"] == "string"
    assert "application/json" in responses["422"]["content"]


async def _iter(chunks: list[bytes]) -> AsyncGenerator[bytes, None]:
    for chunk in chunks:
        yield chunk


async def _collect(chunks: AsyncGenerator[bytes, None]) -> bytes:
    collected = bytearray()
    async for chunk in chunks:
        collected.extend(chunk)
    return bytes(collected)


@pytest.mark.asyncio
async def test_primed_passes_through_all_chunks() -> None:
    primed = await _primed(_iter([b"a", b"b", b"c"]))
    assert await _collect(primed) == b"abc"


@pytest.mark.asyncio
async def test_primed_handles_empty_source() -> None:
    primed = await _primed(_iter([]))
    assert await _collect(primed) == b""


@pytest.mark.asyncio
async def test_primed_raises_before_returning_on_immediate_failure() -> None:
    """The whole point of ``_primed``: a failure on the first chunk raises here,
    while the caller can still be told the read failed, rather than after it has
    been told the read succeeded.
    """

    async def _broken() -> AsyncGenerator[bytes, None]:
        msg = "blob missing or unreadable"
        raise OSError(msg)
        yield b""  # pragma: no cover - unreachable, keeps this an async generator

    with pytest.raises(OSError, match="blob missing or unreadable"):
        await _primed(_broken())


@pytest.mark.asyncio
async def test_primed_does_not_hide_failures_after_the_first_chunk() -> None:
    """A failure past the first chunk is not caught: only the first item is primed.

    It must still reach whoever reads the rest.
    """

    async def _fails_on_second_chunk() -> AsyncGenerator[bytes, None]:
        yield b"ok first chunk"
        msg = "disk error on second read"
        raise OSError(msg)

    primed = await _primed(_fails_on_second_chunk())
    with pytest.raises(OSError, match="disk error on second read"):
        await _collect(primed)


@pytest.mark.asyncio
async def test_primed_closes_inner_source_on_early_close() -> None:
    """Simulates a client disconnect mid-download.

    A reader that stops early closes the outer (primed) generator. That must
    close the inner source too, or the store's open file handle stays open
    until the abandoned generator is collected.
    """
    closed = {"value": False}

    async def _source() -> AsyncGenerator[bytes, None]:
        try:
            yield b"first"
            yield b"second"
        finally:
            closed["value"] = True

    primed = await _primed(_source())
    first = await primed.__anext__()
    assert first == b"first"
    assert closed["value"] is False  # not yet, only the first chunk was read

    await primed.aclose()
    assert closed["value"] is True
