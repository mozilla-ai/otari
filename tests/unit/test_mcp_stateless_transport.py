"""The stateless transport's own bounds (R-TRANSPORT-1, R-TRANSPORT-2, R-ADM-1).

Everything here is defense against the remote MCP server, which is untrusted:
its response may be arbitrarily large, and it may answer a credentialed request
with a redirect to somewhere else entirely.
"""

from __future__ import annotations

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from gateway.services.mcp_stateless import (
    RESULT_MAX_BYTES,
    TRANSPORT_MAX_BYTES,
    ConcurrencyGate,
    McpCapacityUnavailable,
    TransportResponseTooLarge,
    build_http_client_factory,
    enforce_content_length,
    result_exceeds_bound,
)


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in self._chunks:
            yield chunk


class _UnreadableStream(httpx.AsyncByteStream):
    """A body that fails the test if anything reads it."""

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("the oversized body was read")
        yield b""


def test_the_transport_client_never_follows_a_redirect() -> None:
    factory = build_http_client_factory()

    client = factory({"Authorization": "Bearer server-secret"}, None, None)

    assert client.follow_redirects is False
    assert client.headers["Accept-Encoding"] == "identity"


@pytest.mark.asyncio
async def test_a_redirect_forwards_neither_credentials_nor_call_data() -> None:
    """The redirect is returned as-is, so nothing is re-sent to its destination."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"location": "https://attacker.example.com/collect"})

    factory = build_http_client_factory()
    client = factory({"Authorization": "Bearer server-secret"}, None, None)
    client._transport = httpx.MockTransport(handler)  # noqa: SLF001

    async with client:
        response = await client.post("https://mcp.example.com/mcp", json={"method": "tools/call"})

    assert response.status_code == 307
    assert len(seen) == 1
    assert seen[0].url.host == "mcp.example.com"


@pytest.mark.asyncio
async def test_an_oversized_content_length_is_refused_before_the_body_is_read() -> None:
    response = httpx.Response(
        200,
        headers={"content-length": str(RESULT_MAX_BYTES + 1)},
        stream=_UnreadableStream(),
    )

    with pytest.raises(TransportResponseTooLarge):
        await enforce_content_length(response)


@pytest.mark.asyncio
async def test_a_response_within_the_ceiling_is_allowed_through() -> None:
    response = httpx.Response(200, headers={"content-length": "2"}, stream=_ChunkedStream([b"ok"]))

    # Returning at all is the assertion: the hook refuses by raising.
    await enforce_content_length(response)
    assert await response.aread() == b"ok"


@pytest.mark.asyncio
async def test_a_chunked_response_is_refused_while_streaming_past_the_ceiling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"transfer-encoding": "chunked"},
            stream=_ChunkedStream([b"x" * TRANSPORT_MAX_BYTES, b"x"]),
        )

    client = build_http_client_factory()()
    client._transport = httpx.MockTransport(handler)  # noqa: SLF001
    async with client:
        with pytest.raises(TransportResponseTooLarge):
            await client.get("https://mcp.example.com/mcp")


@pytest.mark.asyncio
async def test_a_compressed_response_is_refused_before_the_body_is_read() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-length": "10"},
            stream=_UnreadableStream(),
        )

    client = build_http_client_factory()()
    client._transport = httpx.MockTransport(handler)  # noqa: SLF001
    async with client:
        with pytest.raises(TransportResponseTooLarge):
            await client.get("https://mcp.example.com/mcp")


def test_a_result_within_the_ceiling_passes_the_post_decode_measurement() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="Created issue #42")])

    assert result_exceeds_bound(result) is False


def test_an_oversized_result_fails_the_post_decode_measurement() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="x" * (RESULT_MAX_BYTES + 1))])

    assert result_exceeds_bound(result) is True


@pytest.mark.asyncio
async def test_a_gate_admits_up_to_its_ceiling() -> None:
    gate = ConcurrencyGate(limit=2, admission_timeout_s=0.01)

    async with gate.slot():
        async with gate.slot():
            pass


@pytest.mark.asyncio
async def test_a_gate_refuses_once_its_admission_deadline_passes() -> None:
    gate = ConcurrencyGate(limit=1, admission_timeout_s=0.01)

    async with gate.slot():
        with pytest.raises(McpCapacityUnavailable):
            async with gate.slot():
                pass


@pytest.mark.asyncio
async def test_a_refused_slot_is_released_for_the_next_caller() -> None:
    gate = ConcurrencyGate(limit=1, admission_timeout_s=0.01)

    async with gate.slot():
        with pytest.raises(McpCapacityUnavailable):
            async with gate.slot():
                pass

    async with gate.slot():
        pass


@pytest.mark.asyncio
async def test_discovery_load_cannot_exhaust_execution_capacity() -> None:
    """Separate gates (R-ADM-1), so hostile discovery cannot starve an approved call."""
    from gateway.services.mcp_stateless import DISCOVERY_GATE, EXECUTION_GATE

    assert DISCOVERY_GATE is not EXECUTION_GATE
    held = [await DISCOVERY_GATE._semaphore.acquire() for _ in range(DISCOVERY_GATE.limit)]  # noqa: SLF001
    try:
        async with EXECUTION_GATE.slot():
            pass
    finally:
        for _ in held:
            DISCOVERY_GATE._semaphore.release()  # noqa: SLF001
