"""Runtime model aliases: storage-backed display names for real selectors.

An alias can come from two places. ``config.yml`` aliases are immutable at
runtime and validated at startup; ``model_aliases`` rows are writable through
``/v1/aliases``. Both mean the same thing to a request, so everything that
resolves or lists aliases reads them merged, via :func:`effective_aliases`.

A stored alias also has a scope. A row with no ``user_id`` is global, which is
what a ``config.yml`` alias always is. A row with a ``user_id`` belongs to that
user alone: nobody else resolves it, and it shadows a global alias of the same
name for that user. Precedence is most-specific-first, so
``user-scoped > config.yml > global stored``. The middle pair is the pre-existing
rule (the API refuses to store a global alias shadowing a configured one, so it
is a safety net); the user-scoped layer sits on top of both, because overriding
a shared name for one user is the whole point of scoping it.

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

# Global stored aliases: name -> target.
_cache: dict[str, str] = {}
# User-scoped stored aliases: user_id -> {name -> target}. Kept separate from
# _cache rather than keyed on (user_id, name) so a resolution for one user never
# has to scan another user's aliases.
_user_cache: dict[str, dict[str, str]] = {}
_cached_at: float | None = None


def cached_aliases(user_id: str | None = None) -> dict[str, str]:
    """The stored aliases this worker last loaded. Empty before the first load.

    Returns the global ones, or ``user_id``'s own layer alone when given: the
    caller-facing merge is :func:`effective_aliases`.
    """
    if user_id is None:
        return dict(_cache)
    return dict(_user_cache.get(user_id, {}))


def cache_is_stale(ttl: float = ALIAS_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


async def refresh_alias_cache(db: AsyncSession) -> dict[str, str]:
    """Reload the alias cache from the database and return the global layer."""
    global _cached_at  # noqa: PLW0603

    rows = (await db.execute(select(ModelAlias))).scalars().all()
    _cache.clear()
    _user_cache.clear()
    for row in rows:
        if row.user_id is None:
            _cache[row.name] = row.target
        else:
            _user_cache.setdefault(row.user_id, {})[row.name] = row.target
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
    _user_cache.clear()
    _cached_at = None


def effective_aliases(config: GatewayConfig, user_id: str | None = None) -> dict[str, str]:
    """Every alias in force for ``user_id``: stored ones plus the configured ones.

    Layered most-specific-last, so ``user_id``'s own aliases win over a
    ``config.yml`` one, which in turn wins over a global stored one. A caller with
    no user (the master key) sees the global and configured layers only, never
    another user's aliases.
    """
    merged = {**_cache, **config.aliases}
    if user_id is not None:
        merged.update(_user_cache.get(user_id, {}))
    return merged


def resolve_effective_alias(config: GatewayConfig, name: str, user_id: str | None = None) -> str | None:
    """The target ``name`` resolves to for ``user_id``, or None when not an alias."""
    target = effective_aliases(config, user_id).get(name)
    return target if isinstance(target, str) and target else None


def all_alias_names(config: GatewayConfig) -> set[str]:
    """Every name that is an alias to somebody, in any scope.

    For writes that must refuse to treat an alias name as a real model key
    (pricing rows, model allow-list entries). Those checks are scope-blind on
    purpose: the name means "alias" to at least one caller, so storing it as a
    model key would be dead data no matter whose request came in.
    """
    names = set(_cache) | set(config.aliases)
    for scoped in _user_cache.values():
        names |= set(scoped)
    return names


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
    scoped = sum(len(names) for names in _user_cache.values())
    if aliases or scoped:
        logger.info("Loaded %d global and %d user-scoped model alias(es)", len(aliases), scoped)
