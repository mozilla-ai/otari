"""Runtime search tools: dashboard-configured ``/v1/search`` tools, merged over config.

The search counterpart of :mod:`gateway.services.provider_store_service`, and
deliberately the same shape. A search tool can come from two places:
``config.yml`` ``search_tools:`` entries, immutable at runtime and validated at
startup, and ``search_tool_credentials`` rows written through the dashboard.
Both mean the same thing to a request, so the dispatch path must see them merged.

Resolution has to stay synchronous: ``resolve_search_tool`` reads
``config.search_tools`` on the request path with no database session of its own.
So stored tools are overlaid onto ``config.search_tools`` in memory: loaded at
startup, refreshed on a TTL, and re-applied immediately on the worker that served
a write. A stored row wins over a config-file entry of the same name, and that
shadowing is logged at startup so it is never silent.

The API key is held encrypted and is optional (a ``searxng`` backend is normally
keyless); it is decrypted here only to build the in-memory overlay. A row whose
key cannot be decrypted (no or wrong ``OTARI_SECRET_KEY``) is skipped with a
warning rather than crashing the gateway. Standalone mode only: the caller must
not load or refresh this in the hybrid platform path.
"""

import asyncio
import time
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import SearchToolCredential
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)

# How long a worker may serve a stale search-tool overlay before refreshing. The
# same TTL the provider overlay uses, for the same reason: a newly added or
# edited tool reaches every replica within it.
SEARCH_TOOL_CACHE_TTL_SECONDS = 30.0


class _Unset:
    """Sentinel type: 'this field was not provided', distinct from an explicit None."""


# A field left at UNSET keeps its stored value; passing None clears it. This lets
# a PATCH drop an api_base or rotate a key without disturbing the rest.
UNSET: Final = _Unset()

# name -> decrypted overlay entry (the same shape as a config.search_tools value)
_cache: dict[str, dict[str, Any]] = {}
_cached_at: float | None = None


def _last4(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return api_key[-4:]


def _row_to_entry(row: SearchToolCredential) -> dict[str, Any]:
    """Build a config.search_tools-shaped overlay entry from a stored row.

    Raises ``SecretBoxUnavailableError`` / ``SecretDecryptionError`` when the row
    has a key that cannot be decrypted; the caller decides whether to skip it.
    """
    entry: dict[str, Any] = {"provider": row.provider}
    if row.api_base:
        entry["api_base"] = row.api_base
    if row.timeout_seconds:
        entry["timeout"] = row.timeout_seconds
    if row.options:
        entry["options"] = dict(row.options)
    if row.encrypted_api_key:
        entry["api_key"] = decrypt_secret(row.encrypted_api_key)
    return entry


def cached_search_tools() -> dict[str, dict[str, Any]]:
    """The stored search-tool overlay this worker last loaded (decrypted)."""
    return {name: dict(entry) for name, entry in _cache.items()}


def cache_is_stale(ttl: float = SEARCH_TOOL_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


def reset_search_tool_cache() -> None:
    """Drop the overlay cache so the next load starts clean (startup, tests)."""
    global _cached_at  # noqa: PLW0603

    _cache.clear()
    _cached_at = None


def config_file_search_tools(config: GatewayConfig) -> dict[str, dict[str, Any]]:
    """The config-file search tools, with no stored overlay applied.

    Before the first :func:`apply_to_config` there is no overlay to strip, so
    ``config.search_tools`` is itself the baseline.
    """
    baseline = config._search_tool_baseline
    return baseline if baseline is not None else config.search_tools


def apply_to_config(config: GatewayConfig) -> set[str]:
    """Rebuild ``config.search_tools`` as config-file tools overlaid by the cache.

    Captures the config-file tools as the per-config baseline on first call
    (before any overlay), so repeated applies stay idempotent and a removed
    stored row restores the config entry even after a cache reset. Returns the
    set of names where a stored row shadows a config one.
    """
    if config._search_tool_baseline is None:
        config._search_tool_baseline = {name: dict(entry) for name, entry in config.search_tools.items()}
    baseline = config._search_tool_baseline
    config.search_tools = {**baseline, **_cache}
    return set(baseline) & set(_cache)


async def refresh_search_tool_cache(db: AsyncSession, config: GatewayConfig) -> set[str]:
    """Reload the overlay from the database, apply it, and return shadowed names."""
    global _cached_at  # noqa: PLW0603

    rows = (await db.execute(select(SearchToolCredential))).scalars().all()
    overlay: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            overlay[row.name] = _row_to_entry(row)
        except (SecretBoxUnavailableError, SecretDecryptionError):
            logger.warning(
                "Skipping stored search tool '%s': its API key could not be decrypted (check OTARI_SECRET_KEY).",
                row.name,
            )
    _cache.clear()
    _cache.update(overlay)
    _cached_at = time.monotonic()
    return apply_to_config(config)


async def load_search_tools_at_startup(db: AsyncSession, config: GatewayConfig) -> None:
    """Prime the overlay so the first request does not race the first refresh.

    A failure here is logged rather than raised: stored tools are an addition to
    the config ones, and a gateway that serves every config-file search tool is
    better than one that refuses to start because a credential load failed.
    """
    reset_search_tool_cache()
    try:
        shadowed = await refresh_search_tool_cache(db, config)
    except Exception:
        logger.exception("Failed to load stored search tools; continuing with config search tools only")
        return
    if _cache:
        logger.info("Loaded %d stored search tool(s)", len(_cache))
    for name in sorted(shadowed):
        logger.warning(
            "Stored search tool '%s' shadows the config.yml search tool of the same name; "
            "the dashboard entry is in effect.",
            name,
        )


async def run_search_tool_refresher(config: GatewayConfig, interval: float = SEARCH_TOOL_CACHE_TTL_SECONDS) -> None:
    """Reload the search-tool overlay forever so other writers' changes arrive.

    A write refreshes the worker that served it; this covers sibling workers and
    other replicas, which converge within ``interval``. Every error is swallowed
    and retried on the next tick so a database blip cannot kill the refresher and
    freeze the overlay. Cancelled at shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_search_tool_cache(db, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Stored search tool refresh failed; retrying in %ss", interval, exc_info=True)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def list_search_tools(db: AsyncSession) -> list[SearchToolCredential]:
    """Every stored search tool, ordered by name."""
    rows = (await db.execute(select(SearchToolCredential).order_by(SearchToolCredential.name))).scalars().all()
    return list(rows)


async def get_search_tool(db: AsyncSession, name: str) -> SearchToolCredential | None:
    """The stored search tool called ``name``, or ``None``."""
    return await db.get(SearchToolCredential, name)


async def get_search_tool_for_update(db: AsyncSession, name: str) -> SearchToolCredential | None:
    """Like :func:`get_search_tool`, but locks the row ``FOR UPDATE``.

    Used by the PATCH path so a version check and the write it guards run under
    the same row lock, exactly as the provider-credential path does.
    """
    stmt = select(SearchToolCredential).where(SearchToolCredential.name == name).with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


async def save_search_tool(
    db: AsyncSession,
    *,
    name: str,
    provider: str | _Unset = UNSET,
    api_base: str | None | _Unset = UNSET,
    api_key: str | None | _Unset = UNSET,
    timeout: float | None | _Unset = UNSET,
    options: dict[str, Any] | None | _Unset = UNSET,
) -> SearchToolCredential:
    """Create or update a stored search tool (staged; caller commits).

    Each field is tri-state: left at ``UNSET`` it keeps the stored value; passed
    ``None`` it is cleared; passed a value it is set. ``api_key`` is encrypted
    before storage and requires ``OTARI_SECRET_KEY`` (raises
    ``SecretBoxUnavailableError``); passing it ``None`` clears the stored key,
    which is the normal state for a keyless SearXNG backend. ``options`` is
    normalised to ``{}`` when cleared, since the column is non-null. The
    plaintext key is never logged.
    """
    existing = await db.get(SearchToolCredential, name)
    if existing is None:
        # ``provider`` is non-null, so a create must supply it; the route
        # validates that before staging.
        row = SearchToolCredential(name=name, provider="", options={})
        db.add(row)
    else:
        row = existing

    if not isinstance(provider, _Unset):
        row.provider = provider
    if not isinstance(api_base, _Unset):
        row.api_base = api_base
    if not isinstance(timeout, _Unset):
        row.timeout_seconds = timeout
    if not isinstance(options, _Unset):
        row.options = options or {}
    if not isinstance(api_key, _Unset):
        if api_key:
            row.encrypted_api_key = encrypt_secret(api_key)
            row.last4 = _last4(api_key)
        else:
            row.encrypted_api_key = None
            row.last4 = None

    return row


async def reencrypt_search_tools(db: AsyncSession) -> tuple[int, int]:
    """Re-encrypt stored search-tool keys with the current primary OTARI_SECRET_KEY.

    Returns ``(reencrypted, unreadable)``. Rows without a stored key are ignored.
    A key that cannot be decrypted with the configured key set is left untouched
    and counted as unreadable, so the operator can recover it by replacing that
    tool's key.
    """
    rows = (
        (await db.execute(select(SearchToolCredential).where(SearchToolCredential.encrypted_api_key.is_not(None))))
        .scalars()
        .all()
    )
    reencrypted = 0
    unreadable = 0
    for row in rows:
        if row.encrypted_api_key is None:
            continue
        try:
            plaintext = decrypt_secret(row.encrypted_api_key)
        except SecretDecryptionError:
            unreadable += 1
            continue
        row.encrypted_api_key = encrypt_secret(plaintext)
        reencrypted += 1
    return reencrypted, unreadable


async def delete_search_tool(db: AsyncSession, name: str) -> bool:
    """Delete a stored search tool (staged; caller commits). Returns whether it existed."""
    row = await db.get(SearchToolCredential, name)
    if row is None:
        return False
    await db.delete(row)
    return True
