"""Unit tests for the pluggable composite backend and its cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gateway.services.composite_backend import (
    CachingCompositeBackend,
    LocalCompositeBackend,
    PlatformCompositeBackend,
)

_DEF_A = {"automation_key": "automation:a", "name": "a", "plan": {"nodes": []}}
_DEF_B = {"automation_key": "automation:b", "name": "b", "plan": {"nodes": []}}


@pytest.mark.asyncio
async def test_local_backend_loads_file_and_filters(tmp_path: Path) -> None:
    file = tmp_path / "composites.json"
    file.write_text(json.dumps([_DEF_A, _DEF_B]))
    backend = LocalCompositeBackend(str(file))

    assert len(await backend.fetch()) == 2
    only_a = await backend.fetch(automation_key="automation:a")
    assert [d["automation_key"] for d in only_a] == ["automation:a"]


@pytest.mark.asyncio
async def test_local_backend_loads_directory(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_DEF_A))
    (tmp_path / "b.json").write_text(json.dumps(_DEF_B))
    backend = LocalCompositeBackend(str(tmp_path))
    assert len(await backend.fetch()) == 2


@pytest.mark.asyncio
async def test_local_backend_missing_path_fails_open() -> None:
    assert await LocalCompositeBackend(None).fetch() == []
    assert await LocalCompositeBackend("/nonexistent/path.json").fetch() == []


@pytest.mark.asyncio
async def test_local_backend_malformed_fails_open(tmp_path: Path) -> None:
    file = tmp_path / "bad.json"
    file.write_text("{ not json")
    assert await LocalCompositeBackend(str(file)).fetch() == []


@pytest.mark.asyncio
async def test_platform_backend_empty_token_returns_empty() -> None:
    backend = PlatformCompositeBackend(config=object())  # type: ignore[arg-type]
    assert await backend.fetch(user_token=None) == []
    assert await backend.fetch(user_token="") == []


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(
        self, *, user_token: str | None = None, automation_key: str | None = None
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return [_DEF_A]


@pytest.mark.asyncio
async def test_cache_serves_within_ttl_and_refetches_after() -> None:
    now = {"t": 1000.0}
    inner = _CountingBackend()
    backend = CachingCompositeBackend(inner, ttl_seconds=30.0, clock=lambda: now["t"])

    await backend.fetch(user_token="tk", automation_key="automation:a")
    await backend.fetch(user_token="tk", automation_key="automation:a")
    assert inner.calls == 1  # second call served from cache

    now["t"] += 31.0
    await backend.fetch(user_token="tk", automation_key="automation:a")
    assert inner.calls == 2  # TTL expired -> refetch


@pytest.mark.asyncio
async def test_cache_keys_by_token_and_automation() -> None:
    inner = _CountingBackend()
    backend = CachingCompositeBackend(inner, ttl_seconds=30.0, clock=lambda: 0.0)
    await backend.fetch(user_token="tk1", automation_key="automation:a")
    await backend.fetch(user_token="tk2", automation_key="automation:a")
    await backend.fetch(user_token="tk1", automation_key="automation:b")
    assert inner.calls == 3  # distinct keys, each a miss


@pytest.mark.asyncio
async def test_cache_is_bounded_under_key_rotation() -> None:
    # A caller rotating tokens must not grow the cache without bound.
    inner = _CountingBackend()
    backend = CachingCompositeBackend(inner, ttl_seconds=30.0, max_entries=64, clock=lambda: 0.0)
    for i in range(1000):
        await backend.fetch(user_token=f"tk{i}", automation_key="automation:a")
    assert len(backend._cache) <= 64
