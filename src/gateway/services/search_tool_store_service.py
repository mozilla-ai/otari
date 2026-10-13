"""Runtime search and fetch instances: dashboard-configured rows, merged over config.

The search counterpart of :mod:`gateway.services.provider_store_service`, and
deliberately the same shape. A search instance can come from two places:
``config.yml`` ``search_tools:`` entries, immutable at runtime and validated at
startup, and ``search_tool_credentials`` rows written through the dashboard.
Both mean the same thing to a request, so the dispatch path must see them merged.
A fetch instance is the same, with ``fetch_tools:`` and the rows whose ``kind`` is
``fetch``.

Resolution has to stay synchronous: ``resolve_search_tool`` reads
``config.search_tools`` on the request path with no database session of its own.
So stored rows are overlaid onto ``config.search_tools`` and
``config.fetch_tools`` in memory: loaded at startup, refreshed on a TTL, and
re-applied immediately on the worker that served a write. A stored row wins over
a config-file entry of the same name and kind, and that shadowing is logged at
startup so it is never silent. The periodic refresh also re-reads the runtime
tool settings, which name these instances, so a default set through another
replica arrives with the rows it names.

The API key is held encrypted and is optional (a ``searxng`` backend is normally
keyless); it is decrypted here only to build the in-memory overlay. A row whose
key cannot be decrypted (no or wrong ``OTARI_SECRET_KEY``) is skipped with a
warning rather than crashing the gateway. Standalone mode only: the caller must
not load or refresh this in the hybrid platform path.
"""

import asyncio
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.core.settings.tools import ToolKind, dangling_reference_warnings, warn_about_instances
from gateway.log_config import logger
from gateway.models.secret_fields import restore_redacted_values
from gateway.models.tools import SearchToolCredential
from gateway.repositories.tools import SearchToolRepository
from gateway.services.runtime_settings_service import SettingValue
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tool_settings_service import apply_override, load_overrides

# How long a worker may serve a stale search-tool overlay before refreshing. The
# same TTL the provider overlay uses, for the same reason: a newly added or
# edited tool reaches every replica within it.
SEARCH_TOOL_CACHE_TTL_SECONDS = 30.0


class _Unset:
    """Sentinel type: 'this field was not provided', distinct from an explicit None."""


# A field left at UNSET keeps its stored value; passing None clears it. This lets
# a PATCH drop an api_base or rotate a key without disturbing the rest.
UNSET: Final = _Unset()

# name -> decrypted overlay entry, the same shape as a config.search_tools value
# for a search row and a config.fetch_tools value for a fetch row
_cache: dict[str, dict[str, Any]] = {}
_fetch_cache: dict[str, dict[str, Any]] = {}
_cached_at: float | None = None
# Each row's kind and version as the last overlay saw it, so what the instance
# rules say about a row is logged when it is new or has changed, not on every
# refresh.
_seen_rows: dict[str, tuple[str, datetime | None]] = {}
# The value each default setting or fetch_tool held when it was last reported as
# naming no instance, so a reference left dangling is reported once per value.
_reported_references: dict[str, str] = {}
# The runtime tool settings as this worker last read them from the database, so
# a refresh applies only a value that changed there since. Comparing with the
# config instead would undo a write this worker applied after the read.
_read_settings: dict[str, SettingValue] = {}


def _last4(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return api_key[-4:]


def _row_to_entry(row: SearchToolCredential) -> dict[str, Any]:
    """Build a config.search_tools- or config.fetch_tools-shaped overlay entry from a stored row.

    Raises ``SecretBoxUnavailableError`` / ``SecretDecryptionError`` when the row
    has a key that cannot be decrypted; the caller decides whether to skip it.
    """
    entry: dict[str, Any] = {"provider": row.provider}
    if row.fetch_tool and row.kind != "fetch":
        entry["fetch_tool"] = row.fetch_tool
    if row.api_base:
        entry["api_base"] = row.api_base
    if row.timeout_seconds:
        entry["timeout"] = row.timeout_seconds
    if row.options:
        entry["options"] = dict(row.options)
    if row.encrypted_api_key:
        entry["api_key"] = decrypt_secret(row.encrypted_api_key)
    return entry


def cache_is_stale(ttl: float = SEARCH_TOOL_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


def reset_search_tool_cache() -> None:
    """Drop the overlay cache so the next load starts clean (startup, tests)."""
    global _cached_at  # noqa: PLW0603

    _cache.clear()
    _fetch_cache.clear()
    _seen_rows.clear()
    _reported_references.clear()
    _read_settings.clear()
    _cached_at = None


def config_file_search_tools(config: GatewayConfig) -> dict[str, dict[str, Any]]:
    """The config-file search tools, with no stored overlay applied.

    Before the first :func:`apply_to_config` there is no overlay to strip, so
    ``config.search_tools`` is itself the baseline.
    """
    baseline = config._search_tool_baseline
    return baseline if baseline is not None else config.search_tools


def config_file_fetch_tools(config: GatewayConfig) -> dict[str, dict[str, Any]]:
    """The config-file fetch instances, with no stored overlay applied."""
    baseline = config._fetch_tool_baseline
    return baseline if baseline is not None else config.fetch_tools


def stored_tool_names() -> frozenset[str]:
    """The names of the stored rows the last overlay holds, of either kind."""
    return frozenset(_cache) | frozenset(_fetch_cache)


def config_file_tools(config: GatewayConfig, kind: ToolKind) -> dict[str, dict[str, Any]]:
    """The config-file instances of ``kind``, with no stored overlay applied."""
    return config_file_search_tools(config) if kind == "search" else config_file_fetch_tools(config)


def apply_to_config(config: GatewayConfig) -> set[str]:
    """Rebuild ``config.search_tools`` and ``config.fetch_tools`` as config-file entries overlaid by the cache.

    Captures each map's config-file entries as its per-config baseline on first
    call (before any overlay), so repeated applies stay idempotent and a removed
    stored row restores the config entry even after a cache reset. Returns the
    set of names where a stored row shadows a config one of its kind.
    """
    if config._search_tool_baseline is None:
        config._search_tool_baseline = {name: dict(entry) for name, entry in config.search_tools.items()}
    if config._fetch_tool_baseline is None:
        config._fetch_tool_baseline = {name: dict(entry) for name, entry in config.fetch_tools.items()}
    search_baseline = config._search_tool_baseline
    fetch_baseline = config._fetch_tool_baseline
    config.search_tools = {**search_baseline, **_cache}
    config.fetch_tools = {**fetch_baseline, **_fetch_cache}
    return (set(search_baseline) & set(_cache)) | (set(fetch_baseline) & set(_fetch_cache))


def _overlay_rows(config: GatewayConfig, rows: Sequence[SearchToolCredential], *, report_changes: bool) -> set[str]:
    """Overlay the rows on the config's maps, and return the shadowed names.

    A row new or changed since the last overlay has its undecryptable key
    reported, and, with ``report_changes``, what the instance rules say about
    it. Startup passes ``False``, because the lifespan's own check reports
    every instance once the runtime settings have loaded too.
    """
    global _cached_at  # noqa: PLW0603

    overlays: dict[str, dict[str, dict[str, Any]]] = {"search": {}, "fetch": {}}
    seen: dict[str, tuple[str, datetime | None]] = {}
    changed: dict[str, list[str]] = {"search": [], "fetch": []}
    for row in rows:
        kind: ToolKind = "fetch" if row.kind == "fetch" else "search"
        seen[row.name] = (kind, row.updated_at)
        is_new = _seen_rows.get(row.name) != seen[row.name]
        try:
            overlays[kind][row.name] = _row_to_entry(row)
        except (SecretBoxUnavailableError, SecretDecryptionError):
            if is_new:
                logger.warning(
                    "Skipping stored %s tool '%s': its API key could not be decrypted (check OTARI_SECRET_KEY).",
                    kind,
                    row.name,
                )
            continue
        if is_new:
            changed[kind].append(row.name)
    _seen_rows.clear()
    _seen_rows.update(seen)
    _cache.clear()
    _cache.update(overlays["search"])
    _fetch_cache.clear()
    _fetch_cache.update(overlays["fetch"])
    _cached_at = time.monotonic()
    shadowed = apply_to_config(config)
    if report_changes and (changed["search"] or changed["fetch"]):
        warn_about_instances(config, changed["search"], changed["fetch"])
    return shadowed


def _apply_settings(config: GatewayConfig, overrides: dict[str, SettingValue]) -> None:
    """Apply each runtime tool setting whose stored value changed since this worker last read it."""
    for key, value in overrides.items():
        if key in _read_settings and _read_settings[key] == value:
            continue
        _read_settings[key] = value
        if getattr(config, key) != value:
            apply_override(config, key, value)
            # Key only: a *_url value may embed credentials.
            logger.info("Applied tool setting %s, changed since this worker last read it", key)


def _report_dangling_references(config: GatewayConfig) -> None:
    """Warn about a default or fetch_tool that names no instance, once for each value it takes."""
    dangling = dangling_reference_warnings(config)
    for key, (value, message) in dangling.items():
        if _reported_references.get(key) != value:
            logger.warning(message, value)
            _reported_references[key] = value
    for key in set(_reported_references) - set(dangling):
        del _reported_references[key]


async def refresh_search_tool_cache(db: AsyncSession, config: GatewayConfig) -> set[str]:
    """Reload the stored rows, apply them, and return shadowed names.

    The rows only. A write to the tool settings reloads them to check a default
    against, and must not re-apply the stored settings it is about to replace.
    """
    shadowed = _overlay_rows(config, await SearchToolRepository(db).list_committed(), report_changes=True)
    _report_dangling_references(config)
    return shadowed


async def refresh_tool_instances(db: AsyncSession, config: GatewayConfig) -> set[str]:
    """Reload the stored rows and the runtime tool settings, apply both, and return shadowed names.

    Only startup reads the tool settings otherwise, so without this a runtime
    change reaches only the worker that served the write. Both are read in
    ``db``'s one transaction, the rows first: a create that adds a second search
    instance stores the default naming the first in the same commit, so a read
    between the two statements gets at worst that default with the rows before
    the create, never the second row without the default. Both are applied with
    no wait in between, so no request sees one without the other.
    """
    rows = await SearchToolRepository(db).list_committed()
    overrides = await load_overrides(db, report_invalid=False)
    shadowed = _overlay_rows(config, rows, report_changes=True)
    _apply_settings(config, overrides)
    _report_dangling_references(config)
    return shadowed


async def load_search_tools_at_startup(db: AsyncSession, config: GatewayConfig) -> None:
    """Prime the overlay so the first request does not race the first refresh.

    A failure here is logged rather than raised: stored tools are an addition to
    the config ones, and a gateway that serves every config-file search tool is
    better than one that refuses to start because a credential load failed.

    What the instance rules say is left to the lifespan's own check, which runs
    next, once every setting has loaded; the dangling references it reports are
    recorded here so the first refresh does not report them again.
    """
    reset_search_tool_cache()
    try:
        shadowed = _overlay_rows(config, await SearchToolRepository(db).list_committed(), report_changes=False)
        # The lifespan applied these just before; recorded so the first refresh
        # applies only what changes after.
        _read_settings.update(await load_overrides(db, report_invalid=False))
    except Exception:
        logger.exception("Failed to load stored search tools; continuing with config search tools only")
        shadowed = set()
    _reported_references.update({key: value for key, (value, _) in dangling_reference_warnings(config).items()})
    if _cache or _fetch_cache:
        logger.info("Loaded %d stored search and %d stored fetch tool(s)", len(_cache), len(_fetch_cache))
    for name in sorted(shadowed):
        logger.warning(
            "Stored tool '%s' shadows the config.yml entry of the same name and kind; "
            "the dashboard entry is in effect.",
            name,
        )


async def run_search_tool_refresher(config: GatewayConfig, interval: float = SEARCH_TOOL_CACHE_TTL_SECONDS) -> None:
    """Reload the stored rows and the runtime tool settings forever so other writers' changes arrive.

    A write refreshes the worker that served it; this covers sibling workers and
    other replicas, which converge within ``interval``. Every error is swallowed
    and retried on the next tick so a database blip cannot kill the refresher and
    freeze the overlay. Cancelled at shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_tool_instances(db, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Stored search tool refresh failed; retrying in %ss", interval, exc_info=True)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def list_search_tools(db: AsyncSession, kind: ToolKind | None = None) -> list[SearchToolCredential]:
    """Every stored row, or every one of ``kind``, ordered by name."""
    stmt = select(SearchToolCredential).order_by(SearchToolCredential.name)
    if kind is not None:
        stmt = stmt.where(SearchToolCredential.kind == kind)
    return list((await db.execute(stmt)).scalars().all())


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
    kind: ToolKind | _Unset = UNSET,
    provider: str | _Unset = UNSET,
    fetch_tool: str | None | _Unset = UNSET,
    api_base: str | None | _Unset = UNSET,
    api_key: str | None | _Unset = UNSET,
    timeout: float | None | _Unset = UNSET,
    options: dict[str, Any] | None | _Unset = UNSET,
) -> SearchToolCredential:
    """Create or update a stored search or fetch instance (staged; caller commits).

    Each field is tri-state: left at ``UNSET`` it keeps the stored value; passed
    ``None`` it is cleared; passed a value it is set. ``kind`` is set at create
    only, ``search`` when left at ``UNSET``; the route refuses a change. ``api_key``
    is encrypted before storage and requires ``OTARI_SECRET_KEY`` (raises
    ``SecretBoxUnavailableError``); passing it ``None`` clears the stored key,
    which is the normal state for a keyless SearXNG backend. ``options`` is
    normalised to ``{}`` when cleared, since the column is non-null. The
    plaintext key is never logged.
    """
    existing = await db.get(SearchToolCredential, name)
    if existing is None:
        # ``provider`` is non-null, so a create must supply it; the route
        # validates that before staging.
        row = SearchToolCredential(
            name=name, kind="search" if isinstance(kind, _Unset) else kind, provider="", options={}
        )
        db.add(row)
    else:
        row = existing

    if not isinstance(provider, _Unset):
        row.provider = provider
    if not isinstance(fetch_tool, _Unset):
        row.fetch_tool = fetch_tool
    if not isinstance(api_base, _Unset):
        row.api_base = api_base
    if not isinstance(timeout, _Unset):
        row.timeout_seconds = timeout
    if not isinstance(options, _Unset):
        # Same mask round-trip rule as ``provider_store_service.save_credential``.
        row.options = restore_redacted_values(options, existing.options if existing else None) or {}
    if not isinstance(api_key, _Unset):
        if api_key:
            row.encrypted_api_key = encrypt_secret(api_key)
            row.last4 = _last4(api_key)
        else:
            row.encrypted_api_key = None
            row.last4 = None

    return row


async def reencrypt_search_tools(db: AsyncSession) -> tuple[int, int, int]:
    """Re-encrypt stored search-tool keys with the current primary OTARI_SECRET_KEY.

    Returns ``(reencrypted, unreadable, skipped)``. Rows without a stored key are
    ignored. A key that cannot be decrypted with the configured key set is left
    untouched and counted as unreadable, so the operator can recover it by
    replacing that tool's key.

    Each row is written only if it still holds the ciphertext that was read, so
    an edit committed mid-rotation is not overwritten. Such a row is counted as
    skipped rather than retried: whoever wrote it already encrypted it with the
    primary key.
    """
    rows = (
        (await db.execute(select(SearchToolCredential).where(SearchToolCredential.encrypted_api_key.is_not(None))))
        .scalars()
        .all()
    )
    reencrypted = 0
    unreadable = 0
    skipped = 0
    for row in rows:
        original = row.encrypted_api_key
        if original is None:
            continue
        try:
            plaintext = decrypt_secret(original)
        except SecretDecryptionError:
            unreadable += 1
            continue
        # Core UPDATE rather than a mutation on the loaded row: the whole point
        # is the WHERE, and an ORM flush would carry no condition at all.
        result = await db.execute(
            update(SearchToolCredential)
            .where(SearchToolCredential.name == row.name, SearchToolCredential.encrypted_api_key == original)
            .values(encrypted_api_key=encrypt_secret(plaintext))
            .execution_options(synchronize_session=False)
        )
        # `execute` is typed as returning Result; an UPDATE always yields a
        # CursorResult, which is where rowcount lives.
        if cast(CursorResult[Any], result).rowcount == 1:
            reencrypted += 1
        else:
            skipped += 1
    return reencrypted, unreadable, skipped


async def delete_search_tool(db: AsyncSession, name: str) -> bool:
    """Delete a stored search or fetch instance (staged; caller commits). Returns whether it existed."""
    row = await db.get(SearchToolCredential, name)
    if row is None:
        return False
    await db.delete(row)
    return True
