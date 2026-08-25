"""Runtime model aliases: storage-backed display names for real selectors.

An alias can come from two places. ``config.yml`` aliases are immutable at
runtime and validated at startup; ``model_aliases`` rows are writable through
``/v1/aliases``. Both mean the same thing to a request, so everything that
resolves or lists aliases reads them merged, via :func:`effective_aliases`.

A stored alias has two independent scopes. Its **workspace** says which tenant
owns it: an alias resolves only for requests in that workspace, which is what
lets two workspaces each define ``fast`` and get their own target. Within a
workspace, ``user_id`` scopes it further: ``NULL`` means every caller in that
workspace sees it, and a non-null ``user_id`` belongs to that user alone,
shadowing the workspace-wide row of the same name for them.

A ``config.yml`` alias has no workspace. It comes from a file the deployment
owns, so it is in force in every workspace, and it is never listed under one.

Precedence is most-specific-first, so ``user-scoped > config.yml > stored
workspace-wide``. The middle pair is the pre-existing rule (the API refuses to
store a workspace-wide alias shadowing a configured one, so it is a safety
net); the user-scoped layer sits on top of both, because overriding a shared name
for one user is the whole point of scoping it.

Resolution has to stay synchronous. ``resolve_provider_selector`` is called from
eleven places, including ``services/vision.py``, which has no database session
and no way to get one; making alias lookup async would mean threading a session
through the whole dispatch path. So stored aliases are held in a process-wide
cache, refreshed from the database rather than read per request. A write
refreshes its own worker immediately; other workers and replicas converge within
``ALIAS_CACHE_TTL_SECONDS``, which is the staleness window for a newly created
alias, not for anything already serving traffic.

The cache is keyed by workspace first, which is what made widening the table's
uniqueness constraint safe: while it was keyed on name alone, a second
workspace's ``fast`` silently shadowed the first at request time, so the
constraint could not allow one. A resolution that names no workspace falls back
to the deployment's default, matching ``services/workspace_scope``: a caller with
no workspace of its own (the master key, an operator-configured selector) is
acting deployment-wide, and the default workspace is where its writes land too.
"""

import asyncio
import time
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import ModelAlias
from gateway.services.workspace_scope import lookup_default_workspace_id

# How long a worker may serve a stale alias map before refreshing. A new alias
# takes at most this long to work on every replica; existing ones are unaffected.
ALIAS_CACHE_TTL_SECONDS = 30.0

# Workspace-wide stored aliases: workspace_id -> {name -> target}.
_cache: dict[uuid.UUID, dict[str, str]] = {}
# User-scoped stored aliases: workspace_id -> user_id -> {name -> target}. Kept
# separate from _cache rather than keyed on (user_id, name) so a resolution for
# one user never has to scan another user's aliases.
_user_cache: dict[uuid.UUID, dict[str, dict[str, str]]] = {}
# Where a resolution that names no workspace looks. Loaded with the rows, so it
# cannot disagree with them, and ``None`` on a deployment with no workspace at
# all, where there are no stored aliases to resolve anyway.
_default_workspace: uuid.UUID | None = None
_cached_at: float | None = None


def _scope(workspace_id: uuid.UUID | None) -> uuid.UUID | None:
    """The workspace a lookup reads, resolving "none given" to the default."""
    return workspace_id if workspace_id is not None else _default_workspace


def cached_aliases(user_id: str | None = None, *, workspace_id: uuid.UUID | None = None) -> dict[str, str]:
    """The stored aliases this worker last loaded. Empty before the first load.

    Returns one workspace's workspace-wide layer, or ``user_id``'s own layer
    within it alone when given: the caller-facing merge is :func:`effective_aliases`.
    """
    scope = _scope(workspace_id)
    if scope is None:
        return {}
    if user_id is None:
        return dict(_cache.get(scope, {}))
    return dict(_user_cache.get(scope, {}).get(user_id, {}))


def cache_is_stale(ttl: float = ALIAS_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _cached_at is None or (time.monotonic() - _cached_at) >= ttl


async def refresh_alias_cache(db: AsyncSession) -> dict[uuid.UUID, dict[str, str]]:
    """Reload the alias cache from the database and return the workspace-wide layers.

    Builds fresh dicts and rebinds them, so the swap is atomic from a concurrent
    reader's point of view: the default workspace is resolved with an ``await``
    in the middle, and clearing in place would leave a window where a request
    resolves against an empty map and 400s on a model that exists.
    """
    global _cache, _user_cache, _default_workspace, _cached_at  # noqa: PLW0603

    rows = (await db.execute(select(ModelAlias))).scalars().all()
    fresh_global: dict[uuid.UUID, dict[str, str]] = {}
    fresh_scoped: dict[uuid.UUID, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.user_id is None:
            fresh_global.setdefault(row.workspace_id, {})[row.name] = row.target
        else:
            fresh_scoped.setdefault(row.workspace_id, {}).setdefault(row.user_id, {})[row.name] = row.target

    default_workspace = await lookup_default_workspace_id(db)

    _cache, _user_cache, _default_workspace = fresh_global, fresh_scoped, default_workspace
    _cached_at = time.monotonic()
    return {workspace: dict(names) for workspace, names in _cache.items()}


def reset_alias_cache() -> None:
    """Drop the cache so the next load starts clean (startup and tests).

    Without this a test's aliases would leak into the next one through the
    process-wide cache, and a worker restarting against a different database
    would answer from the old one until its first refresh.
    """
    global _cache, _user_cache, _default_workspace, _cached_at  # noqa: PLW0603

    _cache, _user_cache = {}, {}
    _default_workspace = None
    _cached_at = None


def effective_aliases(
    config: GatewayConfig, user_id: str | None = None, *, workspace_id: uuid.UUID | None = None
) -> dict[str, str]:
    """Every alias in force for ``user_id`` in ``workspace_id``.

    Layered most-specific-last, so the user's own aliases win over a
    ``config.yml`` one, which in turn wins over the workspace's own stored
    layer. A caller with no user (the master key) sees the workspace-wide and
    configured layers only, never another user's aliases. An omitted
    ``workspace_id`` reads the deployment's default workspace, which is where a
    deployment-wide write lands.
    """
    scope = _scope(workspace_id)
    stored = _cache.get(scope, {}) if scope is not None else {}
    merged = {**stored, **config.aliases}
    if user_id is not None and scope is not None:
        merged.update(_user_cache.get(scope, {}).get(user_id, {}))
    return merged


def resolve_effective_alias(
    config: GatewayConfig, name: str, user_id: str | None = None, *, workspace_id: uuid.UUID | None = None
) -> str | None:
    """The target ``name`` resolves to for this caller, or None when not an alias."""
    target = effective_aliases(config, user_id, workspace_id=workspace_id).get(name)
    return target if isinstance(target, str) and target else None


def all_alias_names(config: GatewayConfig) -> set[str]:
    """Every name that is an alias to somebody, in any scope.

    For writes that must refuse to treat an alias name as a real model key
    (pricing rows, model allow-list entries). Those checks are scope-blind on
    purpose, across workspaces as well as users: the name means "alias" to at
    least one caller, so storing it as a model key would be dead data no matter
    whose request came in.
    """
    names = set(config.aliases)
    for workspace_names in _cache.values():
        names |= set(workspace_names)
    for per_user in _user_cache.values():
        for scoped in per_user.values():
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
    workspace_global = sum(len(names) for names in aliases.values())
    scoped = sum(len(names) for per_user in _user_cache.values() for names in per_user.values())
    if workspace_global or scoped:
        logger.info(
            "Loaded %d workspace-wide and %d user-scoped model alias(es) across %d workspace(s)",
            workspace_global,
            scoped,
            len(set(_cache) | set(_user_cache)),
        )
