"""Model metadata from the public models.dev catalog, fetched and cached.

models.dev (https://models.dev, MIT licensed) publishes per-model metadata beyond
what pricing needs: input/output modalities, reasoning and
tool-calling flags, structured-output and attachment support, knowledge cutoff,
release date, and open-weights status. This module fetches its ``api.json`` (the
v2 document, or v1 when v2 fails), reads it into the typed catalog of
``gateway.types.models_dev``, and serves that from memory, keeping the last good
copy through a failure, so an offline or unreachable models.dev never breaks the
dashboard. The catalog refresher (``services/pricing_refresh_service.py``)
stores each download in the database and serves the stored copy on every worker.

Until a worker has a downloaded or stored catalog, it serves the one bundled
with the gateway. ``GatewayConfig.models_dev_metadata`` gates the outbound call
only. Prices come from the bundled or accepted snapshot (``services/pricing``),
not from this fetch.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.pricing import bundled_catalog
from gateway.types.model_endpoint import ModelEndpoint, models_dev_endpoints
from gateway.types.model_type import ModelType, models_dev_model_type
from gateway.types.models_dev import MODELS_DEV_SOURCES, ModelsDevCatalog, ModelsDevModel, ModelsDevSource

# The catalog is a few MB; a short read timeout keeps a slow models.dev from
# stalling a dashboard load, and a failure just means "the last good copy this time".
_FETCH_TIMEOUT_SECONDS = 15.0

# Read the body incrementally and stop if it exceeds this, so a misbehaving or
# compromised models.dev cannot force the gateway to buffer an unbounded response
# into memory. Generous relative to the few-MB real catalog; overflow is a failed
# fetch like any other.
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
    canonical_model_id: str | None = None
    """models.dev's provider-independent id, ``vendor/model``: who made the model."""
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
    type: ModelType | None = None
    """The kind models.dev gives the model; ``None`` from a v1 document, which gives none."""
    endpoints: list[ModelEndpoint] = field(default_factory=list)
    """The Otari routes that serve the model, from its ``api`` block; none from a v1 document, which has none."""

    @classmethod
    def from_model(cls, model: ModelsDevModel) -> "ModelCatalogEntry":
        """The metadata view of one catalog model."""
        rates = model.cost.rates if model.cost is not None else None
        return cls(
            name=model.name,
            description=model.description,
            family=model.family,
            canonical_model_id=model.canonical_model_id,
            input_modalities=list(model.input_modalities),
            output_modalities=list(model.output_modalities),
            reasoning=model.reasoning,
            tool_call=model.tool_call,
            structured_output=model.structured_output,
            attachment=model.attachment,
            temperature=model.temperature,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            knowledge_cutoff=model.knowledge,
            release_date=model.release_date,
            last_updated=model.last_updated,
            open_weights=model.open_weights,
            deprecated=model.status == "deprecated",
            cost_input=None if rates is None or rates.input is None else float(rates.input),
            cost_output=None if rates is None or rates.output is None else float(rates.output),
            type=models_dev_model_type(model.type),
            endpoints=list(models_dev_endpoints(api.protocol for api in model.api)),
        )


@dataclass
class _CatalogCache:
    """Single-blob TTL cache for the whole models.dev catalog.

    ``data`` is the catalog this worker serves, kept through a failed fetch so
    an outage does not take the facts away, and ``fetched_at`` when it was
    downloaded, by this worker or by the one that stored it. ``ok`` and ``at``
    describe this worker's last fetch attempt, and so which TTL governs the next
    one.
    """

    data: ModelsDevCatalog | None = None
    fetched_at: datetime | None = None
    ok: bool = False
    at: float = 0.0  # time.monotonic() of the last attempt, 0.0 when never tried


_cache = _CatalogCache()
_lock = asyncio.Lock()
_MISS = object()


def _fallback() -> ModelsDevCatalog | None:
    """The catalog a worker serves before it has downloaded one: the bundled one."""
    return bundled_catalog()


def _served() -> ModelsDevCatalog | None:
    return _cache.data if _cache.data is not None else _fallback()


def cached_models_dev_catalog() -> ModelsDevCatalog | None:
    """The catalog as last fetched, at any age, else the bundled one, without fetching or locking.

    For a reader that runs off the request path and must not bind the fetch
    lock to its own loop: it answers from whatever a page load or the catalog
    refresher last put here.
    """
    return _served()


def serve_catalog(catalog: ModelsDevCatalog, fetched_at: datetime) -> None:
    """Serve ``catalog``, downloaded at ``fetched_at``: a stored copy, or one this worker fetched."""
    _cache.data = catalog
    _cache.fetched_at = fetched_at


def clear_catalog_cache() -> None:
    """Reset the cache (tests, and after a config change)."""
    _cache.data = None
    _cache.fetched_at = None
    _cache.ok = False
    _cache.at = 0.0


def _read_cache(ttl: int) -> object:
    """Return the cached data if still fresh, else the ``_MISS`` sentinel.

    A successful fetch honors the configured TTL (0 disables caching, so it is
    never fresh); a failed one is held only for ``_NEGATIVE_TTL_SECONDS``.
    """
    if _cache.at == 0.0 or (_cache.ok and ttl == 0):
        return _MISS
    limit = ttl if _cache.ok else _NEGATIVE_TTL_SECONDS
    if (time.monotonic() - _cache.at) < limit:
        return _cache.data
    return _MISS


class ModelsDevFetchError(Exception):
    """The models.dev catalog could not be downloaded or read."""


class ModelsDevRefusedError(ModelsDevFetchError):
    """The download has a shape the gateway cannot read, so it is not served."""


async def _download() -> ModelsDevCatalog:
    """The first models.dev document that downloads and reads, v2 then v1; raises the last one's failure.

    Each document that fails is logged once, with the one tried next.
    """
    failure: ModelsDevFetchError | None = None
    for source in MODELS_DEV_SOURCES:
        if failure is not None:
            logger.warning("%s; reading the models.dev %s catalog instead", failure, source.name)
        try:
            return await _download_from(source)
        except ModelsDevFetchError as exc:
            failure = exc
    assert failure is not None
    raise failure


async def _download_from(source: ModelsDevSource) -> ModelsDevCatalog:
    """One bounded download of one document, read into a catalog; every failure is a ``ModelsDevFetchError``.

    A document whose shape the gateway cannot read is refused. One it reads
    with a few damaged models is kept, with one warning naming them.
    """
    name = f"models.dev {source.name} catalog"
    try:
        async with (
            httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client,
            client.stream("GET", source.url, headers={"User-Agent": "otari-gateway"}) as resp,
        ):
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_CATALOG_BYTES:
                    raise ModelsDevFetchError(f"{name} exceeded {_MAX_CATALOG_BYTES} bytes")
                chunks.append(chunk)
        data = json.loads(b"".join(chunks))
    except ModelsDevFetchError:
        raise
    except Exception as exc:
        raise ModelsDevFetchError(f"{name} fetch failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelsDevRefusedError(f"{name} refused: it is not a JSON object")
    catalog = source.parse(data)
    if (reason := catalog.refusal()) is not None:
        raise ModelsDevRefusedError(f"{name} refused: {reason}")
    if catalog.issues.count or catalog.issues.unknown_units:
        # The line carries fetched text, so the console handler must not read it as markup.
        logger.warning("%s read with %s", name, catalog.issues.describe(), extra={"markup": False})
    return catalog


async def _fetch() -> ModelsDevCatalog | None:
    try:
        return await _download()
    except ModelsDevRefusedError as exc:
        logger.error("%s; keeping the last good copy", exc)
        return None
    except ModelsDevFetchError as exc:
        logger.warning("%s; keeping the last good copy", exc)
        return None


def _fresh_cached(reuse_within: float) -> ModelsDevCatalog | None:
    fetched_at = _cache.fetched_at
    if reuse_within > 0 and fetched_at is not None and (datetime.now(UTC) - fetched_at).total_seconds() < reuse_within:
        return _cache.data
    return None


async def fetch_models_dev_document(*, reuse_within: float = 0.0, fill_cache: bool = False) -> ModelsDevCatalog:
    """Download ``api.json`` for a caller that needs it now, raising on failure.

    The price refresh and the catalog refresher use this. A catalog served here
    that was downloaded within ``reuse_within`` seconds, by this worker or by the
    one that stored it, is returned instead of downloading again. A download
    made here is served from then on when ``fill_cache`` is set.
    """
    async with _lock:
        if (reused := _fresh_cached(reuse_within)) is not None:
            return reused
        data = await _download()
        if fill_cache:
            serve_catalog(data, datetime.now(UTC))
        return data


async def load_models_dev_catalog(config: GatewayConfig, *, serve_stale: bool = False) -> ModelsDevCatalog | None:
    """Return the models.dev catalog this worker serves, fetching it first only where a read may.

    A read fetches only with ``models_dev_metadata`` on, ``serve_stale`` off
    and the cache past ``models_dev_cache_ttl_seconds``. ``serve_stale`` is for
    a read the catalog refresher keeps warm (:func:`background_catalog_enabled`),
    so it never waits on models.dev. Until a catalog is fetched or stored, the
    bundled one is returned; a failed fetch returns the last good catalog.
    """
    if not config.models_dev_metadata or serve_stale:
        return _served()

    ttl = config.models_dev_cache_ttl_seconds
    if _read_cache(ttl) is not _MISS:
        return _served()

    async with _lock:
        # Double-check under the lock so a burst of dashboard loads triggers one
        # fetch, not one per request.
        if _read_cache(ttl) is not _MISS:
            return _served()
        data = await _fetch()
        if data is not None:
            serve_catalog(data, datetime.now(UTC))
        _cache.ok = data is not None
        _cache.at = time.monotonic()
        return _served()


# Floor on the refresh cadence, for the same reason as the discovery refresher's:
# a tiny configured TTL must not turn into a fetch storm against models.dev.
_MIN_REFRESH_INTERVAL_SECONDS = 300.0


def background_catalog_enabled(config: GatewayConfig) -> bool:
    """Whether the catalog refresher downloads, so reads answer from what it serves.

    ``models_dev_cache_ttl_seconds = 0`` is the documented way to disable
    caching, so it keeps meaning "fetch on every read", and without
    ``models_dev_metadata`` nothing is downloaded for the facts.
    """
    return config.models_dev_metadata and config.models_dev_cache_ttl_seconds > 0


def catalog_refresh_interval(config: GatewayConfig) -> float:
    """How often the catalog refresher downloads models.dev, in seconds."""
    return max(_MIN_REFRESH_INTERVAL_SECONDS, float(config.models_dev_cache_ttl_seconds))


def build_metadata_map(config: GatewayConfig, catalog: ModelsDevCatalog | None) -> dict[str, ModelCatalogEntry]:
    """Metadata for every model under a configured provider, keyed ``instance:model``.

    Keys use the configured instance name (not the provider type), so a named
    instance backed by, say, openai still joins to the models the dashboard shows
    under that instance. models.dev is looked up by the instance's provider type.
    """
    out: dict[str, ModelCatalogEntry] = {}
    if catalog is None:
        return out
    for instance in config.providers:
        provider = catalog.provider_for(config.provider_instance_type(instance))
        if provider is None:
            continue
        for model_id, model in provider.models.items():
            out[f"{instance}:{model_id}"] = ModelCatalogEntry.from_model(model)
    return out
