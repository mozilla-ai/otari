"""Unit tests for `CompositeToolBackend` — the wrapper that lets memory run alongside another
gateway tool in one tool loop. Uses fake in-memory backends (no network) that record their
enter/exit lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.services.composite_backend import CompositeToolBackend


class _FakeBackend:
    def __init__(self, name: str, tool_names: list[str], *, fail_on_enter: bool = False) -> None:
        self.name = name
        self._tool_names = tool_names
        self._fail_on_enter = fail_on_enter
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeBackend:
        if self._fail_on_enter:
            raise RuntimeError(f"{self.name} unreachable")
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": n}} for n in self._tool_names]

    def owns_tool(self, name: str) -> bool:
        return name in self._tool_names

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(self.name, f"use {self.name}")]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return f"{self.name}:{name}"


def test_requires_at_least_one_backend() -> None:
    with pytest.raises(ValueError):
        CompositeToolBackend([])


def test_merges_tools_and_hints_and_ownership() -> None:
    a = _FakeBackend("web", ["web_search"])
    b = _FakeBackend("memory", ["memory_search", "memory_store"])
    comp = CompositeToolBackend([a, b])
    assert [t["function"]["name"] for t in comp.openai_tools] == ["web_search", "memory_search", "memory_store"]
    assert comp.owns_tool("web_search")
    assert comp.owns_tool("memory_store")
    assert not comp.owns_tool("nope")
    assert comp.purpose_hints() == [("web", "use web"), ("memory", "use memory")]


@pytest.mark.asyncio
async def test_routes_call_to_owning_backend() -> None:
    a = _FakeBackend("web", ["web_search"])
    b = _FakeBackend("memory", ["memory_search"])
    async with CompositeToolBackend([a, b]) as comp:
        assert await comp.call_tool("web_search", {}) == "web:web_search"
        assert await comp.call_tool("memory_search", {}) == "memory:memory_search"
        with pytest.raises(KeyError):
            await comp.call_tool("unknown", {})


@pytest.mark.asyncio
async def test_enters_and_exits_all_members() -> None:
    a = _FakeBackend("web", ["web_search"])
    b = _FakeBackend("memory", ["memory_search"])
    async with CompositeToolBackend([a, b]):
        assert a.entered and b.entered
    assert a.exited and b.exited


@pytest.mark.asyncio
async def test_partial_open_failure_closes_entered_members() -> None:
    good = _FakeBackend("web", ["web_search"])
    bad = _FakeBackend("memory", ["memory_search"], fail_on_enter=True)
    with pytest.raises(RuntimeError):
        async with CompositeToolBackend([good, bad]):
            pass
    # The first backend opened, then the second failed: the composite must still close the first.
    assert good.entered and good.exited
