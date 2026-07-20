"""Runtime provider credentials: dashboard-configured providers, merged over config.

A provider instance can come from two places. ``config.yml`` providers are
immutable at runtime and validated at startup; ``provider_credentials`` rows are
written through the dashboard. Both mean the same thing to a request, so the
dispatch path must see them merged.

Resolution has to stay synchronous. ``get_provider_kwargs`` and
``resolve_provider_selector`` read ``config.providers`` directly and are called
from the streaming hot path, which has no database session. So stored providers
are overlaid onto ``config.providers`` in memory: loaded at startup, refreshed
on a TTL (like the alias cache), and re-applied immediately on the worker that
served a write. A stored row wins over a config-file entry of the same instance
name, and that shadowing is logged at startup so it is never silent.

The API key is held encrypted; it is decrypted here only to build the in-memory
overlay. A row whose key cannot be decrypted (no or wrong ``OTARI_SECRET_KEY``)
is skipped with a warning rather than crashing the gateway. Standalone mode
only: the caller must not load or refresh this in the hybrid platform path.
"""

import asyncio
import time
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import ProviderCredential
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)

# How long a worker may serve a stale provider overlay before refreshing. A newly
# added or edited provider takes at most this long to reach every replica;
# providers already serving traffic are unaffected.
PROVIDER_CACHE_TTL_SECONDS = 30.0


class _Unset:
    """Sentinel type: 'this field was not provided', distinct from an explicit None."""


# A field left at UNSET keeps its stored value; passing None clears it. This lets
# a PATCH distinguish "leave the api_base alone" from "remove the api_base".
UNSET: Final = _Unset()

# instance -> decrypted overlay entry (the same shape as a config.providers value)
_cache: dict[str, dict[str, Any]] = {}
_cached_at: float | None = None
# The config-file baseline lives on the config object (config._provider_baseline),
# captured once, so it is per-config and survives a cache reset without being
# re-derived from an already-merged providers map.


def _last4(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return api_key[-4:]


def _row_to_entry(row: ProviderCredential) -> dict[str, Any]:
    """Build a config.providers-shaped overlay entry from a stored row.

    Raises ``SecretBoxUnavailableError`` / ``SecretDecryptionError`` when the row
    has a key that cannot be decrypted; the caller decides whether to skip it.
    """
    entry: dict[str, Any] = {}
    if row.provider_type:
        entry["provider_type"] = row.provider_type
    if row.api_base:
        entry["api_base"] = row.api_base
    if row.client_args:
        entry["client_args"] = dict(row.client_args)
    if row.encrypted_api_key:
        entry["api_key"] = decrypt_secret(row.encrypted_api_key)
    return entry


def cached_providers() -> dict[str, dict[str, Any]]:
    """The stored provider overlay this worker last loaded (decrypted)."""
    return {name: dict(entry) for name, entry in _cache.items()}


def cache_is_stale(ttl: float = PROVIDER_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


def reset_provider_cache() -> None:
    """Drop the overlay cache so the next load starts clean (startup, tests).

    The config-file baseline lives on the config object, so it is not touched
    here; a fresh config starts with an empty baseline of its own.
    """
    global _cached_at  # noqa: PLW0603

    _cache.clear()
    _cached_at = None


def apply_to_config(config: GatewayConfig) -> set[str]:
    """Rebuild ``config.providers`` as config-file providers overlaid by the cache.

    Captures the config-file providers as the per-config baseline on first call
    (before any overlay), so repeated applies stay idempotent and a removed
    stored row restores the config entry even after a cache reset. Returns the
    set of instance names where a stored row shadows a config one.
    """
    if config._provider_baseline is None:
        config._provider_baseline = {name: dict(entry) for name, entry in config.providers.items()}
    baseline = config._provider_baseline
    config.providers = {**baseline, **_cache}
    return set(baseline) & set(_cache)


async def refresh_provider_cache(db: AsyncSession, config: GatewayConfig) -> set[str]:
    """Reload the overlay from the database, apply it, and return shadowed names."""
    global _cached_at  # noqa: PLW0603

    rows = (await db.execute(select(ProviderCredential))).scalars().all()
    overlay: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            overlay[row.instance] = _row_to_entry(row)
        except (SecretBoxUnavailableError, SecretDecryptionError):
            logger.warning(
                "Skipping stored provider '%s': its API key could not be decrypted "
                "(check OTARI_SECRET_KEY).",
                row.instance,
            )
    _cache.clear()
    _cache.update(overlay)
    _cached_at = time.monotonic()
    return apply_to_config(config)


async def load_providers_at_startup(db: AsyncSession, config: GatewayConfig) -> None:
    """Prime the overlay so the first request does not race the first refresh.

    A failure here is logged rather than raised: stored providers are an addition
    to the config ones, and a gateway that serves every config-file provider is
    better than one that refuses to start because a credential load failed.
    """
    reset_provider_cache()
    try:
        shadowed = await refresh_provider_cache(db, config)
    except Exception:
        logger.exception("Failed to load stored providers; continuing with config providers only")
        return
    if _cache:
        logger.info("Loaded %d stored provider(s)", len(_cache))
    for instance in sorted(shadowed):
        logger.warning(
            "Stored provider '%s' shadows the config.yml provider of the same name; "
            "the dashboard credential is in effect.",
            instance,
        )


async def run_provider_refresher(config: GatewayConfig, interval: float = PROVIDER_CACHE_TTL_SECONDS) -> None:
    """Reload the provider overlay forever so other writers' changes arrive.

    A write refreshes the worker that served it; this covers sibling workers and
    other replicas, which converge within ``interval``. Every error is swallowed
    and retried on the next tick so a database blip cannot kill the refresher and
    freeze the overlay. Cancelled at shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_provider_cache(db, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Stored provider refresh failed; retrying in %ss", interval, exc_info=True)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def list_credentials(db: AsyncSession) -> list[ProviderCredential]:
    """Every stored provider credential, ordered by instance name."""
    rows = (await db.execute(select(ProviderCredential).order_by(ProviderCredential.instance))).scalars().all()
    return list(rows)


async def get_credential(db: AsyncSession, instance: str) -> ProviderCredential | None:
    """The stored credential for ``instance``, or ``None``."""
    return await db.get(ProviderCredential, instance)


async def get_credential_for_update(db: AsyncSession, instance: str) -> ProviderCredential | None:
    """Like :func:`get_credential`, but locks the row ``FOR UPDATE``.

    Used by the PATCH path so a version check and the write it guards run under
    the same row lock: on PostgreSQL a concurrent PATCH waits and then sees the
    new ``updated_at`` (so it 412s instead of clobbering). SQLite serialises
    writers anyway, so the lock is a no-op there without changing correctness.
    """
    stmt = select(ProviderCredential).where(ProviderCredential.instance == instance).with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


async def save_credential(
    db: AsyncSession,
    *,
    instance: str,
    provider_type: str | None | _Unset = UNSET,
    api_base: str | None | _Unset = UNSET,
    api_key: str | None | _Unset = UNSET,
    client_args: dict[str, Any] | None | _Unset = UNSET,
) -> ProviderCredential:
    """Create or update a stored credential (staged; caller commits).

    Each field is tri-state: left at ``UNSET`` it keeps the stored value; passed
    ``None`` it is cleared; passed a value it is set. That lets a PATCH remove an
    ``api_base`` or rotate a key without disturbing the rest. ``api_key`` is
    encrypted before storage and requires ``OTARI_SECRET_KEY`` (raises
    ``SecretBoxUnavailableError``); passing it ``None`` clears the stored key
    (keyless local backends). ``client_args`` is normalised to ``{}`` when
    cleared, since the column is non-null. The plaintext key is never logged.
    """
    existing = await db.get(ProviderCredential, instance)
    if existing is None:
        row = ProviderCredential(instance=instance, client_args={})
        db.add(row)
    else:
        row = existing

    if not isinstance(provider_type, _Unset):
        row.provider_type = provider_type
    if not isinstance(api_base, _Unset):
        row.api_base = api_base
    if not isinstance(client_args, _Unset):
        row.client_args = client_args or {}
    if not isinstance(api_key, _Unset):
        if api_key:
            row.encrypted_api_key = encrypt_secret(api_key)
            row.last4 = _last4(api_key)
        else:
            row.encrypted_api_key = None
            row.last4 = None

    return row


async def delete_credential(db: AsyncSession, instance: str) -> bool:
    """Delete a stored credential (staged; caller commits). Returns whether it existed."""
    row = await db.get(ProviderCredential, instance)
    if row is None:
        return False
    await db.delete(row)
    return True
