"""Unit tests for the composite dispatch hook (gated serve path)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.responses import StreamingResponse

from gateway.services import composite_hook


class _StubBackend:
    def __init__(self, composites: list[dict[str, Any]]) -> None:
        self._composites = composites

    async def fetch(self, *, user_token: str | None = None, automation_key: str | None = None) -> list[dict[str, Any]]:
        return self._composites


class _Boom:
    async def fetch(self, *, user_token: str | None = None, automation_key: str | None = None) -> list[dict[str, Any]]:
        raise RuntimeError("boom")


class _BackgroundTasks:
    def add_task(self, *args: Any, **kwargs: Any) -> None:
        pass


def _composite(status: str = "approved") -> dict[str, Any]:
    return {
        "automation_key": "automation:x",
        "status": status,
        "tier": "t0_deterministic",
        "composite_program_id": "11111111-1111-1111-1111-111111111111",
        "composite_program_version_id": "22222222-2222-2222-2222-222222222222",
        "plan": {"nodes": [{"type": "emit_tool_use", "tool": "resolve_time", "args": {}}]},
        "verifier_spec": {},
    }


def _ctx() -> Any:
    return SimpleNamespace(config=SimpleNamespace(), user_token=None, db=None, reservation=None)


def _request(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "session_label": "automation:x",
        "messages": [{"role": "user", "content": "go"}],
        "model": "claude-haiku-4-5",
        "stream": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def _serve(request: Any, ctx: Any = None) -> Any:
    return await composite_hook.try_serve_composite(
        request=request, ctx=ctx or _ctx(), background_tasks=_BackgroundTasks()
    )


@pytest.fixture(autouse=True)
def _reset_backend() -> Any:
    composite_hook.reset_backend()
    yield
    composite_hook.reset_backend()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: Any) -> None:
    monkeypatch.delenv("OTARI_COMPOSITES_ENABLED", raising=False)
    monkeypatch.delenv("GATEWAY_COMPOSITES_ENABLED", raising=False)


@pytest.mark.asyncio
async def test_disabled_by_default_returns_none(monkeypatch: Any) -> None:
    monkeypatch.setattr(composite_hook, "_get_backend", lambda cfg: _StubBackend([_composite()]))
    result = await _serve(_request())
    assert result is None


@pytest.mark.asyncio
async def test_enabled_serves_approved_nonstream(monkeypatch: Any) -> None:
    monkeypatch.setenv("OTARI_COMPOSITES_ENABLED", "true")
    monkeypatch.setattr(composite_hook, "_get_backend", lambda cfg: _StubBackend([_composite()]))
    result = await _serve(_request())
    assert isinstance(result, dict)
    assert result["stop_reason"] == "tool_use"
    assert result["content"][0]["name"] == "resolve_time"


@pytest.mark.asyncio
async def test_enabled_serves_approved_stream(monkeypatch: Any) -> None:
    monkeypatch.setenv("OTARI_COMPOSITES_ENABLED", "true")
    monkeypatch.setattr(composite_hook, "_get_backend", lambda cfg: _StubBackend([_composite()]))
    result = await _serve(_request(stream=True))
    assert isinstance(result, StreamingResponse)


@pytest.mark.asyncio
async def test_shadow_composite_falls_through(monkeypatch: Any) -> None:
    monkeypatch.setenv("OTARI_COMPOSITES_ENABLED", "true")
    monkeypatch.setattr(composite_hook, "_get_backend", lambda cfg: _StubBackend([_composite("shadow")]))
    result = await _serve(_request())
    assert result is None


@pytest.mark.asyncio
async def test_no_session_label_returns_none(monkeypatch: Any) -> None:
    monkeypatch.setenv("OTARI_COMPOSITES_ENABLED", "true")
    monkeypatch.setattr(composite_hook, "_get_backend", lambda cfg: _StubBackend([_composite()]))
    result = await _serve(_request(session_label=None))
    assert result is None


@pytest.mark.asyncio
async def test_backend_error_fails_open(monkeypatch: Any) -> None:
    monkeypatch.setenv("OTARI_COMPOSITES_ENABLED", "true")
    monkeypatch.setattr(composite_hook, "_get_backend", lambda cfg: _Boom())
    result = await _serve(_request())
    assert result is None
