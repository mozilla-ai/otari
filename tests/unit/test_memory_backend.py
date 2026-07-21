"""Unit tests for `MemoryBackend`.

Mocks the HTTP layer so the suite talks to no platform: a tiny in-process httpx transport
routes each POST to a handler and captures the request so we can assert on the dual-auth
headers and the body the gateway sends to /gateway/memory/{search,store,forget}.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from gateway.services.memory_backend import (
    MEMORY_FORGET_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_STORE_TOOL_NAME,
    MemoryBackend,
)


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handlers: dict[str, httpx.Response | Exception]) -> None:
        self._handlers = handlers
        self.captured: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured.append(request)
        handler = self._handlers.get(request.url.path)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {request.url.path}"})
        if isinstance(handler, Exception):
            raise handler
        return handler


def _patched_async_client(handlers: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> _MockTransport:
    transport = _MockTransport(handlers)
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return transport


def _backend() -> MemoryBackend:
    return MemoryBackend(
        base_url="http://backend:8000/api/v1/gateway/memory",
        gateway_token="gw-secret",
        user_token="tk-user",
    )


def test_advertises_three_tools_and_ownership() -> None:
    backend = _backend()
    names = [t["function"]["name"] for t in backend.openai_tools]
    assert names == [MEMORY_SEARCH_TOOL_NAME, MEMORY_STORE_TOOL_NAME, MEMORY_FORGET_TOOL_NAME]
    assert backend.owns_tool(MEMORY_SEARCH_TOOL_NAME)
    assert backend.owns_tool(MEMORY_FORGET_TOOL_NAME)
    assert not backend.owns_tool("web_search")
    assert backend.purpose_hints()  # non-empty


@pytest.mark.asyncio
async def test_call_tool_requires_entered_context() -> None:
    with pytest.raises(RuntimeError):
        await _backend().call_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "x"})


@pytest.mark.asyncio
async def test_search_formats_facts_and_sends_dual_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {
            "/api/v1/gateway/memory/search": httpx.Response(
                200, json={"facts": [{"id": "m1", "memory": "Prefers metric"}]}
            )
        },
        monkeypatch,
    )
    async with _backend() as backend:
        out = await backend.call_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "units?"})
    assert "Prefers metric" in out
    assert "id: m1" in out
    req = transport.captured[0]
    assert req.headers["X-Gateway-Token"] == "gw-secret"
    assert req.headers["X-User-Token"] == "tk-user"


@pytest.mark.asyncio
async def test_search_no_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client({"/api/v1/gateway/memory/search": httpx.Response(200, json={"facts": []})}, monkeypatch)
    async with _backend() as backend:
        out = await backend.call_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "q"})
    assert out == "No relevant memories found."


@pytest.mark.asyncio
async def test_search_empty_query_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client({}, monkeypatch)
    async with _backend() as backend:
        out = await backend.call_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "  "})
    assert out.startswith("[tool error]")


@pytest.mark.asyncio
async def test_store_returns_id(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _patched_async_client(
        {"/api/v1/gateway/memory/store": httpx.Response(200, json={"id": "new1", "memory": "User likes tea"})},
        monkeypatch,
    )
    async with _backend() as backend:
        out = await backend.call_tool(MEMORY_STORE_TOOL_NAME, {"content": "I like tea"})
    assert "new1" in out
    assert transport.captured[0].url.path == "/api/v1/gateway/memory/store"


@pytest.mark.asyncio
async def test_forget_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client({"/api/v1/gateway/memory/forget": httpx.Response(204)}, monkeypatch)
    async with _backend() as backend:
        out = await backend.call_tool(MEMORY_FORGET_TOOL_NAME, {"memory_id": "m1"})
    assert "Deleted memory m1" in out


@pytest.mark.asyncio
async def test_forget_not_found_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client({"/api/v1/gateway/memory/forget": httpx.Response(404)}, monkeypatch)
    async with _backend() as backend:
        out = await backend.call_tool(MEMORY_FORGET_TOOL_NAME, {"memory_id": "ghost"})
    assert out.startswith("[tool error]")


@pytest.mark.asyncio
async def test_unknown_tool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client({}, monkeypatch)
    async with _backend() as backend:
        with pytest.raises(KeyError):
            await backend.call_tool("not_a_memory_tool", {})
