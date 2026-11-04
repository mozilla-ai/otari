"""Runtime search and fetch instances: dashboard-configured rows, merged over config.

A search instance can come from two places:
``config.yml`` ``search_tools:`` entries, immutable at runtime and validated at
startup, and ``search_tool_credentials`` rows written through the dashboard.
Both mean the same thing to a request, so the dispatch path must see them merged.
A fetch instance is the same, with ``fetch_tools:`` and the rows whose ``kind`` is
``fetch``.

Resolution has to stay synchronous: the resolver reads ``config.search_tools``
on the request path with no database session of its own.
So stored rows are overlaid onto ``config.search_tools`` and
``config.fetch_tools`` in memory: loaded at startup, refreshed on a TTL, and
re-applied immediately on the worker that served a write. A stored row wins over
a config-file entry of the same name and kind, and that shadowing is logged at
startup so it is never silent. The periodic refresh also re-reads the runtime
tool settings, which name these instances, so a default set through another
replica arrives with the rows it names.

The overlay is this worker's copy, one per process: ``SearchToolService``
reads the rows and the settings, and hands them here to apply.

The API key is held encrypted and is optional (a ``searxng`` backend is normally
keyless); it is decrypted here only to build the in-memory overlay. A row whose
key cannot be decrypted (no or wrong ``OTARI_SECRET_KEY``) is skipped with a
warning rather than crashing the gateway.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.settings.tools import ToolKind, dangling_reference_warnings, warn_about_instances
from gateway.log_config import logger
from gateway.models.tools import SearchToolCredential
from gateway.services.runtime_settings_service import SettingValue
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
)
from gateway.services.tool_settings_service import apply_override

# How long a worker may serve a stale search-tool overlay before refreshing, so
# a newly added or edited tool reaches every replica within it.
SEARCH_TOOL_CACHE_TTL_SECONDS = 30.0

# name -> decrypted overlay entry, the same shape as a config.search_tools value
# for a search row and a config.fetch_tools value for a fetch row
_cache: dict[str, dict[str, Any]] = {}
_fetch_cache: dict[str, dict[str, Any]] = {}
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


def reset_search_tool_cache() -> None:
    """Drop the overlay cache so the next load starts clean (startup, tests)."""
    _cache.clear()
    _fetch_cache.clear()
    _seen_rows.clear()
    _reported_references.clear()
    _read_settings.clear()


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


def apply_stored_rows(config: GatewayConfig, rows: Sequence[SearchToolCredential]) -> set[str]:
    """Apply the stored rows, and return shadowed names.

    The rows only. A write to the tool settings reloads them to check a default
    against, and must not re-apply the stored settings it is about to replace.
    """
    shadowed = _overlay_rows(config, rows, report_changes=True)
    _report_dangling_references(config)
    return shadowed


def apply_stored_tools(
    config: GatewayConfig, rows: Sequence[SearchToolCredential], overrides: dict[str, SettingValue]
) -> set[str]:
    """Apply the stored rows and the runtime tool settings, and return shadowed names.

    Only startup reads the tool settings otherwise, so without this a runtime
    change reaches only the worker that served the write. Both are applied with
    no wait in between, so no request sees one without the other.
    """
    shadowed = _overlay_rows(config, rows, report_changes=True)
    _apply_settings(config, overrides)
    _report_dangling_references(config)
    return shadowed


def apply_at_startup(config: GatewayConfig, rows: Sequence[SearchToolCredential]) -> set[str]:
    """Prime the overlay so the first request does not race the first refresh, and return shadowed names.

    What the instance rules say is left to the lifespan's own check, which runs
    next, once every setting has loaded.
    """
    return _overlay_rows(config, rows, report_changes=False)


def record_read_settings(overrides: dict[str, SettingValue]) -> None:
    """Record the runtime tool settings the lifespan applied just before.

    So the first refresh applies only what changes after.
    """
    _read_settings.update(overrides)


def report_startup(config: GatewayConfig, shadowed: set[str]) -> None:
    """Log what the startup load holds, and record the dangling references the lifespan's check reports.

    Recorded so the first refresh does not report them again.
    """
    _reported_references.update({key: value for key, (value, _) in dangling_reference_warnings(config).items()})
    if _cache or _fetch_cache:
        logger.info("Loaded %d stored search and %d stored fetch tool(s)", len(_cache), len(_fetch_cache))
    for name in sorted(shadowed):
        logger.warning(
            "Stored tool '%s' shadows the config.yml entry of the same name and kind; "
            "the dashboard entry is in effect.",
            name,
        )


async def run_search_tool_refresher(
    refresh: Callable[[], Awaitable[object]], interval: float = SEARCH_TOOL_CACHE_TTL_SECONDS
) -> None:
    """Reload the stored rows and the runtime tool settings forever so other writers' changes arrive.

    A write refreshes the worker that served it; this covers sibling workers and
    other replicas, which converge within ``interval``. Every error is swallowed
    and retried on the next tick so a database blip cannot kill the refresher and
    freeze the overlay. Cancelled at shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await refresh()
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning(
                "Stored search tool refresh failed (%s); retrying in %ss", describe_database_error(exc), interval
            )
        except Exception:
            logger.warning("Stored search tool refresh failed; retrying in %ss", interval, exc_info=True)
