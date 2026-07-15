"""Runtime model aliases: storage-backed display names for real selectors.

An alias can come from two places. ``config.yml`` aliases are immutable at
runtime and validated at startup; ``model_aliases`` rows are writable through
``/v1/aliases``. Both mean the same thing to a request, so everything that
resolves or lists aliases reads them merged, via :func:`effective_aliases`.

Resolution has to stay synchronous. ``resolve_provider_selector`` is called from
eleven places, including ``services/vision.py``, which has no database session
and no way to get one; making alias lookup async would mean threading a session
through the whole dispatch path. So stored aliases are held in a process-wide
cache, refreshed from the database rather than read per request. A write
refreshes its own worker immediately; other workers and replicas converge within
``ALIAS_CACHE_TTL_SECONDS``, which is the staleness window for a newly created
alias, not for anything already serving traffic.
"""

import asyncio
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import ModelAlias

# How long a worker may serve a stale alias map before refreshing. A new alias
# takes at most this long to work on every replica; existing ones are unaffected.
ALIAS_CACHE_TTL_SECONDS = 30.0

_cache: dict[str, str] = {}
_cached_at: float | None = None


def cached_aliases() -> dict[str, str]:
    """The stored aliases this worker last loaded. Empty before the first load."""
    return dict(_cache)


def cache_is_stale(ttl: float = ALIAS_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


async def refresh_alias_cache(db: AsyncSession) -> dict[str, str]:
    """Reload the alias cache from the database and return it."""
    global _cached_at  # noqa: PLW0603

    rows = (await db.execute(select(ModelAlias))).scalars().all()
    _cache.clear()
    _cache.update({row.name: row.target for row in rows})
    _cached_at = time.monotonic()
    return dict(_cache)


def reset_alias_cache() -> None:
    """Drop the cache so the next load starts clean (startup and tests).

    Without this a test's aliases would leak into the next one through the
    process-wide cache, and a worker restarting against a different database
    would answer from the old one until its first refresh.
    """
    global _cached_at  # noqa: PLW0603

    _cache.clear()
    _cached_at = None


def effective_aliases(config: GatewayConfig) -> dict[str, str]:
    """Every alias in force: stored ones plus the configured ones.

    ``config.yml`` wins on a name collision, but the API refuses to create a
    stored alias that shadows a configured one, so this is a safety net rather
    than a rule anyone should have to reason about.
    """
    return {**_cache, **config.aliases}


def resolve_effective_alias(config: GatewayConfig, name: str) -> str | None:
    """The target ``name`` resolves to, or None when it is not an alias."""
    target = effective_aliases(config).get(name)
    return target if isinstance(target, str) and target else None


async def run_alias_refresher(interval: float = ALIAS_CACHE_TTL_SECONDS) -> None:
    """Reload the alias cache forever, so other writers' aliases arrive.

    A write refreshes the worker that served it, which covers a single-process
    gateway. This covers the rest: sibling workers and other replicas learn about
    an alias within ``interval``. Cancelled at shutdown.

    Every error is swallowed and retried on the next tick. A database blip must
    not kill the refresher, because nothing would restart it and the worker would
    then serve a frozen alias map for as long as it stayed up.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_alias_cache(db)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Model alias refresh failed; retrying in %ss", interval, exc_info=True)


async def load_aliases_at_startup(db: AsyncSession) -> None:
    """Prime the cache so the first request does not race the first refresh.

    A failure here is logged rather than raised: stored aliases are an addition
    to config ones, and a gateway that serves every other model is better than
    one that refuses to start because an alias lookup failed.
    """
    reset_alias_cache()
    try:
        aliases = await refresh_alias_cache(db)
    except Exception:
        logger.exception("Failed to load model aliases; continuing with config aliases only")
        return
    if aliases:
        logger.info("Loaded %d stored model alias(es)", len(aliases))
