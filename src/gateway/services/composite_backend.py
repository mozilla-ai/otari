"""Pluggable composite backend: where the interpreter's plans come from.

Mirrors ``SandboxBackend``/``OTARI_SANDBOX_URL`` one-to-one
(docs/tool-compositor-layer-plan.md sec 2.5, 5.2). Two implementations, selected
by mode:

- ``LocalCompositeBackend`` (standalone/OSS): loads hand-authored or exported
  plans from a local file or directory, keyed by automation key. Zero platform
  dependency.
- ``PlatformCompositeBackend`` (hybrid): fetches approved/shadow plans from the
  platform via ``GET /gateway/composites``.

Both are wrapped in a short-TTL read-through cache (the only gateway-side state,
restart-anytime) and fail open: any error yields no composites, so the request
proceeds to the provider exactly as today.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env

logger = logging.getLogger(__name__)


class CompositeBackend(Protocol):
    async def fetch(
        self, *, user_token: str | None, automation_key: str | None
    ) -> list[dict[str, Any]]: ...


def _filter_by_key(defs: list[dict[str, Any]], automation_key: str | None) -> list[dict[str, Any]]:
    if automation_key is None:
        return defs
    return [d for d in defs if d.get("automation_key") == automation_key]


class LocalCompositeBackend:
    """Load composite definitions from a local JSON file or a directory of them."""

    def __init__(self, path: str | None) -> None:
        self._path = Path(path) if path else None

    async def fetch(
        self, *, user_token: str | None = None, automation_key: str | None = None
    ) -> list[dict[str, Any]]:
        return _filter_by_key(self._load(), automation_key)

    def _load(self) -> list[dict[str, Any]]:
        if self._path is None or not self._path.exists():
            return []
        try:
            files = sorted(self._path.glob("*.json")) if self._path.is_dir() else [self._path]
            out: list[dict[str, Any]] = []
            for file in files:
                data = json.loads(file.read_text())
                if isinstance(data, list):
                    out.extend(d for d in data if isinstance(d, dict))
                elif isinstance(data, dict):
                    out.append(data)
            return out
        except (OSError, ValueError):
            logger.warning("Failed to load local composites from %s; failing open", self._path, exc_info=True)
            return []


class PlatformCompositeBackend:
    """Fetch composite definitions from the platform (hybrid mode)."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    async def fetch(
        self, *, user_token: str | None = None, automation_key: str | None = None
    ) -> list[dict[str, Any]]:
        if not user_token:
            return []
        # Imported lazily to avoid a route-module import cycle.
        from gateway.api.routes._platform import _resolve_platform_composites

        return await _resolve_platform_composites(self._config, user_token, automation_key)


class CachingCompositeBackend:
    """Short-TTL read-through cache over another backend, keyed per token+key."""

    def __init__(
        self,
        inner: CompositeBackend,
        *,
        ttl_seconds: float = 30.0,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._cache: dict[tuple[str | None, str | None], tuple[float, list[dict[str, Any]]]] = {}

    async def fetch(
        self, *, user_token: str | None = None, automation_key: str | None = None
    ) -> list[dict[str, Any]]:
        key = (user_token, automation_key)
        now = self._clock()
        entry = self._cache.get(key)
        if entry is not None and now - entry[0] < self._ttl:
            return entry[1]
        value = await self._inner.fetch(user_token=user_token, automation_key=automation_key)
        self._store(key, now, value)
        return value

    def _store(self, key: tuple[str | None, str | None], now: float, value: list[dict[str, Any]]) -> None:
        # Bound the cache so a caller rotating tokens/automation keys cannot grow
        # it without limit (the only gateway-side state).
        if key not in self._cache and len(self._cache) >= self._max_entries:
            self._evict(now)
        self._cache[key] = (now, value)

    def _evict(self, now: float) -> None:
        for expired_key in [k for k, (t, _) in self._cache.items() if now - t >= self._ttl]:
            del self._cache[expired_key]
        # Still at cap after dropping expired: evict oldest by insertion order.
        while len(self._cache) >= self._max_entries:
            del self._cache[next(iter(self._cache))]


def build_composite_backend(config: GatewayConfig) -> CompositeBackend:
    """Select the backend: a local path wins (standalone-useful), else the
    platform in hybrid mode, else an empty local backend."""
    local_path = otari_env("COMPOSITES_PATH", "").strip()
    if local_path:
        return CachingCompositeBackend(LocalCompositeBackend(local_path))
    if config.is_hybrid_mode:
        return CachingCompositeBackend(PlatformCompositeBackend(config))
    return CachingCompositeBackend(LocalCompositeBackend(None))
