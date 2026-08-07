"""Model metadata from the public models.dev catalog, fetched and cached.

models.dev (https://models.dev, MIT licensed) publishes per-model metadata the
bundled genai-prices data does not carry: input/output modalities, reasoning and
tool-calling flags, structured-output and attachment support, knowledge cutoff,
release date, and open-weights status. This module fetches its ``api.json`` once,
caches it in-memory with a long TTL, and degrades to "no metadata" on any failure
so an offline or unreachable models.dev never breaks the dashboard.

The outbound call is gated by ``GatewayConfig.models_dev_metadata``; when that is
false, or when the fetch fails, callers simply get an empty map and the UI falls
back to what genai-prices already provides.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.core.config import GatewayConfig
from gateway.log_config import logger

MODELS_DEV_URL = "https://models.dev/api.json"

# The catalog is a few MB; a short read timeout keeps a slow models.dev from
# stalling a dashboard load, and a failure just means "no enrichment this time".
_FETCH_TIMEOUT_SECONDS = 15.0

# Read the body incrementally and stop if it exceeds this, so a misbehaving or
# compromised models.dev cannot force the gateway to buffer an unbounded response
# into memory. Generous relative to the few-MB real catalog; overflow degrades to
# "no enrichment", the same as any other fetch failure.
_MAX_CATALOG_BYTES = 64 * 1024 * 1024

# A failed fetch is cached briefly so an offline gateway does not re-attempt (and
# re-wait the timeout) on every request, without pinning the failure for the full
# success TTL.
_NEGATIVE_TTL_SECONDS = 60


@dataclass
class ModelCatalogEntry:
    """Metadata models.dev carries for one model."""

    name: str | None = None
    description: str | None = None
    family: str | None = None
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    reasoning: bool = False
    tool_call: bool = False
    structured_output: bool = False
    attachment: bool = False
    temperature: bool = False
    context_window: int | None = None
    max_output_tokens: int | None = None
    knowledge_cutoff: str | None = None
    release_date: str | None = None
    last_updated: str | None = None
    open_weights: bool = False
    deprecated: bool = False
    cost_input: float | None = None
    cost_output: float | None = None


@dataclass
class _CatalogCache:
    """Single-blob TTL cache for the whole models.dev catalog."""

    data: dict[str, Any] | None = None
    ok: bool = False
    at: float = 0.0  # time.monotonic() of the last attempt, 0.0 when never tried


_cache = _CatalogCache()
_lock = asyncio.Lock()
_MISS = object()


def clear_catalog_cache() -> None:
    """Reset the cache (tests, and after a config change)."""
    _cache.data = None
    _cache.ok = False
    _cache.at = 0.0


def _read_cache(ttl: int, *, serve_stale: bool = False, force: bool = False) -> object:
    """Return the cached data if still fresh, else the ``_MISS`` sentinel.

    ``serve_stale`` accepts a cached *successful* blob at any age, so a read
    never pays the 15s fetch timeout; the background refresher is what keeps it
    current. A failed fetch is deliberately excluded: it falls through to
    ``_NEGATIVE_TTL_SECONDS`` below, because the refresh interval is the success
    cadence (a day by default) and pinning a transient models.dev outage for that
    long, with no read able to retry it and no ``refresh`` flag on
    ``/v1/models/metadata``, would lose enrichment far longer than the failure.
    ``force`` is the refresher's own read: it treats every entry as expired so
    the fetch actually happens.
    """
    if _cache.at == 0.0:
        return _MISS
    if force:
        return _MISS
    if serve_stale and _cache.ok:
        return _cache.data
    # A successful fetch honors the configured TTL (0 disables caching, so it is
    # never fresh); a failed one is held only briefly before retrying.
    if _cache.ok and ttl == 0:
        return _MISS
    limit = ttl if _cache.ok else _NEGATIVE_TTL_SECONDS
    if (time.monotonic() - _cache.at) < limit:
        return _cache.data
    return _MISS


async def _fetch() -> dict[str, Any] | None:
    try:
        async with (
            httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client,
            client.stream("GET", MODELS_DEV_URL, headers={"User-Agent": "otari-gateway"}) as resp,
        ):
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_CATALOG_BYTES:
                    logger.warning("models.dev catalog exceeded %d bytes; ignoring", _MAX_CATALOG_BYTES)
                    return None
                chunks.append(chunk)
        data = json.loads(b"".join(chunks))
    except Exception as exc:
        logger.warning("models.dev catalog fetch failed: %s", exc)
        return None
    if not isinstance(data, dict):
        logger.warning("models.dev catalog was not a JSON object; ignoring")
        return None
    return data


async def load_models_dev_catalog(
    config: GatewayConfig,
    *,
    serve_stale: bool = False,
    force: bool = False,
) -> dict[str, Any] | None:
    """Return the cached models.dev catalog, fetching it if stale.

    Returns ``None`` when metadata enrichment is disabled or the fetch failed.
    ``serve_stale`` answers from the cache at any age so a dashboard read never
    waits on models.dev; ``force`` is the background refresher's fetch.
    """
    if not config.models_dev_metadata:
        return None

    ttl = config.models_dev_cache_ttl_seconds
    cached = _read_cache(ttl, serve_stale=serve_stale, force=force)
    if cached is not _MISS:
        return cached  # type: ignore[return-value]

    async with _lock:
        # Double-check under the lock so a burst of dashboard loads triggers one
        # fetch, not one per request.
        cached = _read_cache(ttl, serve_stale=serve_stale, force=force)
        if cached is not _MISS:
            return cached  # type: ignore[return-value]

        data = await _fetch()
        _cache.data = data
        _cache.ok = data is not None
        _cache.at = time.monotonic()
        return data


# Floor on the refresh cadence, for the same reason as the discovery refresher's:
# a tiny configured TTL must not turn into a fetch storm against models.dev.
_MIN_REFRESH_INTERVAL_SECONDS = 300.0


def background_catalog_enabled(config: GatewayConfig) -> bool:
    """Whether reads may answer from the cache because a refresher fills it.

    ``models_dev_cache_ttl_seconds = 0`` is the documented way to disable
    caching, so it keeps meaning "fetch on every read", and ``models_dev_metadata``
    already disables enrichment outright.
    """
    return config.models_dev_metadata and config.models_dev_cache_ttl_seconds > 0


def _catalog_refresh_interval(config: GatewayConfig) -> float:
    return max(_MIN_REFRESH_INTERVAL_SECONDS, float(config.models_dev_cache_ttl_seconds))


async def run_catalog_refresher(config: GatewayConfig, interval: float | None = None) -> None:
    """Keep the models.dev catalog warm so no read waits on the 15s fetch.

    Primes once immediately, then refetches on the interval. Errors are already
    swallowed inside ``_fetch`` (a failure degrades to "no enrichment"), so the
    guard here only covers an unexpected one taking the refresher down.

    Re-checks ``background_catalog_enabled`` every tick rather than being started
    only when it holds, for the same reason as the discovery refresher:
    ``models_dev_metadata`` and ``models_dev_cache_ttl_seconds`` are both
    runtime-settable, so the loop has to be able to start and stop refetching
    without a restart.
    """
    while True:
        try:
            if background_catalog_enabled(config):
                await load_models_dev_catalog(config, force=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("models.dev catalog refresh failed; retrying on the next tick", exc_info=True)
        # Recomputed per tick, not captured once, so a TTL changed at runtime
        # takes effect on the next round.
        delay = interval if interval is not None else _catalog_refresh_interval(config)
        await asyncio.sleep(delay)


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def parse_entry(model: dict[str, Any]) -> ModelCatalogEntry:
    """Build a :class:`ModelCatalogEntry` from a models.dev model object."""
    modalities = _as_dict(model.get("modalities"))
    limit = _as_dict(model.get("limit"))
    cost = _as_dict(model.get("cost"))
    return ModelCatalogEntry(
        name=_as_str(model.get("name")),
        description=_as_str(model.get("description")),
        family=_as_str(model.get("family")),
        input_modalities=_str_list(modalities.get("input")),
        output_modalities=_str_list(modalities.get("output")),
        reasoning=bool(model.get("reasoning")),
        tool_call=bool(model.get("tool_call")),
        structured_output=bool(model.get("structured_output")),
        attachment=bool(model.get("attachment")),
        temperature=bool(model.get("temperature")),
        context_window=_as_int(limit.get("context")),
        max_output_tokens=_as_int(limit.get("output")),
        knowledge_cutoff=_as_str(model.get("knowledge")),
        release_date=_as_str(model.get("release_date")),
        last_updated=_as_str(model.get("last_updated")),
        open_weights=bool(model.get("open_weights")),
        deprecated=model.get("status") == "deprecated",
        cost_input=_as_float(cost.get("input")),
        cost_output=_as_float(cost.get("output")),
    )


def build_metadata_map(config: GatewayConfig, catalog: dict[str, Any] | None) -> dict[str, ModelCatalogEntry]:
    """Metadata for every model under a configured provider, keyed ``instance:model``.

    Keys use the configured instance name (not the provider type), so a named
    instance backed by, say, openai still joins to the models the dashboard shows
    under that instance. models.dev is looked up by the instance's provider type.
    """
    out: dict[str, ModelCatalogEntry] = {}
    if not catalog:
        return out
    for instance in config.providers:
        provider_type = config.provider_instance_type(instance)
        provider = catalog.get(provider_type)
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            if isinstance(model_id, str) and isinstance(model, dict):
                out[f"{instance}:{model_id}"] = parse_entry(model)
    return out
