"""Stored routing policies, held in a process-wide cache.

Why a cache rather than a query per request: policy lookup happens on the hot
path, and it also has to answer from synchronous call sites that have no database
session (the same constraint that shaped ``alias_service``, whose design this
follows). So rows are loaded into the process and refreshed on a timer.

Consequences worth stating plainly, because one of them is a security property:

* A newly written policy takes up to :data:`POLICY_CACHE_TTL_SECONDS` to reach
  other workers and replicas. The worker that served the write refreshes
  immediately, so a single-process gateway is consistent at once.
* That window also applies to *removing* a candidate and to *attaching a
  mandatory guardrail*. For up to the TTL, other replicas keep serving the old
  plan. An operator tightening a policy for security reasons needs to know the
  change is eventually consistent, not immediate.

Layering: this sits alongside ``alias_service`` rather than inside
``services/routing/`` on purpose. The routing package depends on
``provider_kwargs`` (via ``model_access``) to resolve selectors, and
``provider_kwargs`` needs policy lookup to resolve a static policy name, so a
store inside that package would close an import cycle. Being a leaf peer keeps the
graph one-directional.

A stored spec is validated on write and again on load. A row written by
a newer version whose schema this build does not understand is skipped with a
warning rather than crashing the loader, so one bad row cannot take routing down
for every other policy.
"""

import asyncio
import time

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import RoutingPolicy
from gateway.models.routing import PolicySpec

__all__ = [
    "POLICY_CACHE_TTL_SECONDS",
    "cached_policies",
    "effective_policies",
    "load_policies_at_startup",
    "policy_cache_is_stale",
    "refresh_policy_cache",
    "reset_policy_cache",
    "resolve_effective_policy",
    "run_policy_refresher",
]

# Matches the alias cache's TTL: the two are the same kind of indirection, and
# having them converge at different rates would be a surprise nobody benefits from.
POLICY_CACHE_TTL_SECONDS = 30.0

# Global stored policies: name -> spec.
_cache: dict[str, PolicySpec] = {}
# User-scoped stored policies: user_id -> {name -> spec}. Kept separate rather than
# keyed on (user_id, name) so resolving for one user never scans another's.
_user_cache: dict[str, dict[str, PolicySpec]] = {}
_cached_at: float | None = None


def cached_policies(user_id: str | None = None) -> dict[str, PolicySpec]:
    """The stored policies this worker last loaded. Empty before the first load."""
    if user_id is None:
        return dict(_cache)
    return dict(_user_cache.get(user_id, {}))


def policy_cache_is_stale(ttl: float = POLICY_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


def _parse(row: RoutingPolicy) -> PolicySpec | None:
    try:
        return PolicySpec.model_validate(row.spec)
    except ValidationError:
        logger.warning(
            "Stored routing policy %r (user=%s) does not validate against this build's schema; skipping it. "
            "Other policies are unaffected.",
            row.name,
            row.user_id,
            exc_info=True,
        )
        return None


async def refresh_policy_cache(db: AsyncSession) -> dict[str, PolicySpec]:
    """Reload the policy cache from the database and return the global layer.

    Builds new dicts and rebinds, rather than clearing and refilling in place: the
    swap is then atomic from a concurrent reader's point of view even if this
    function later grows an ``await`` in the middle. Clear-then-refill would open a
    window where a request resolves a policy name against an empty map and 400s on
    a model that exists.
    """
    global _cache, _user_cache, _cached_at  # noqa: PLW0603

    rows = (await db.execute(select(RoutingPolicy))).scalars().all()
    fresh_global: dict[str, PolicySpec] = {}
    fresh_scoped: dict[str, dict[str, PolicySpec]] = {}
    for row in rows:
        spec = _parse(row)
        if spec is None:
            continue
        if row.user_id is None:
            fresh_global[row.name] = spec
        else:
            fresh_scoped.setdefault(row.user_id, {})[row.name] = spec

    _cache, _user_cache = fresh_global, fresh_scoped
    _cached_at = time.monotonic()
    return dict(_cache)


def reset_policy_cache() -> None:
    """Drop the cache so the next load starts clean (startup and tests)."""
    global _cache, _user_cache, _cached_at  # noqa: PLW0603

    _cache, _user_cache = {}, {}
    _cached_at = None


def effective_policies(config: GatewayConfig, user_id: str | None = None) -> dict[str, PolicySpec]:
    """Every policy in force for ``user_id``: stored ones plus configured ones.

    Precedence is most-specific-last, matching aliases exactly:
    ``user-scoped stored > config.yml > global stored``. The middle pair is the
    pre-existing rule for aliases (and the write path refuses to store a global
    policy that shadows a configured one, so this ordering is a safety net rather
    than the enforcement); the user-scoped layer sits on top because overriding a
    shared name for one caller is the entire point of scoping it.

    Returns nothing when routing is disabled, which is what makes
    ``routing.enabled: false`` a true off-switch for stored policies too, not only
    for the ones in the config file.
    """
    if not config.routing.enabled:
        return {}
    merged: dict[str, PolicySpec] = {**_cache, **config.routing.policies}
    if user_id is not None:
        merged.update(_user_cache.get(user_id, {}))
    return merged


def resolve_effective_policy(
    config: GatewayConfig, name: str, user_id: str | None = None
) -> PolicySpec | None:
    """The policy ``name`` resolves to for ``user_id``, or ``None``."""
    return effective_policies(config, user_id).get(name)


def all_policy_names(config: GatewayConfig) -> set[str]:
    """Every name that is a policy to somebody, in any scope.

    Scope-blind on purpose, like ``all_alias_names``: for writes that must refuse
    to treat a policy name as a real model key, the name means "policy" to at
    least one caller, so storing it as a model key would be dead data regardless
    of whose request arrives.
    """
    if not config.routing.enabled:
        return set()
    names = set(_cache) | set(config.routing.policies)
    for scoped in _user_cache.values():
        names |= set(scoped)
    return names


async def run_policy_refresher(interval: float = POLICY_CACHE_TTL_SECONDS) -> None:
    """Reload the policy cache forever, so other writers' policies arrive.

    Every error is swallowed and retried on the next tick. A database blip must not
    kill the refresher: nothing would restart it, and the worker would then serve a
    frozen policy map for as long as it stayed up.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_policy_cache(db)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Routing policy refresh failed; retrying in %ss", interval, exc_info=True)


async def load_policies_at_startup(db: AsyncSession) -> None:
    """Prime the cache so the first request does not race the first refresh.

    A failure here is logged rather than raised, matching the alias loader: stored
    policies are an addition to the configured ones, and a gateway that serves
    every other model is better than one that refuses to start because a policy
    lookup failed.
    """
    reset_policy_cache()
    try:
        policies = await refresh_policy_cache(db)
    except Exception:
        logger.exception("Failed to load routing policies; continuing with config policies only")
        return
    scoped = sum(len(names) for names in _user_cache.values())
    if policies or scoped:
        logger.info("Loaded %d global and %d user-scoped routing policy/policies", len(policies), scoped)
