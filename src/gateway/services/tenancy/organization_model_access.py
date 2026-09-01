"""What a dashboard session may be shown of the model catalog.

``services/model_access`` answers this for an API key, off a stored column. A
session has no such column: its reach is decided by membership. The allow-list
here is produced in that module's own wire shape (canonical ``instance:model``
entries plus the ``instance:*`` wildcard) so ``is_model_allowed`` decides both. A
second matcher would let the catalog and the inference gate disagree about the
same model.

Two disjoint addressing schemes decide it, and the split is
``services/provider_kwargs``'s rather than this module's:

* A selector whose prefix names a ``config.providers`` instance draws its
  credentials from ``config.yml`` and never touches an organization's keys, so
  every workspace of every tenant may use it. Each configured instance
  contributes ``instance:*``.
* A bare ``provider:model`` selector resolves through the organization's own BYO
  keys for the request's workspace, so it contributes only where the caller has
  one.

An organization holding no BYO key still gets every configured instance, which on
a standalone deployment is the whole catalog. Opening this filter therefore
changes nothing for a single-tenant deployment and narrows only where a tenant's
reach is actually narrower.

The scope also answers one thing the allow-list cannot. Aliases and stored
policies are workspace-scoped rows, and the catalog reads them for a workspace
rather than filtering them by target, so a name in a workspace the caller may not
see would be listed even though every entry it resolves to is permitted. See
:attr:`SessionCatalogScope.reads_default_workspace`.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.models.tenancy import User, Workspace
from gateway.repositories.tenancy.org_provider_key_repository import (
    Candidate,
    OrgProviderKeyRepository,
    WorkspaceProviderKeyOverrideRepository,
    WorkspaceProviderModelRestrictionRepository,
    resolve_active_key,
)
from gateway.services.provider_kwargs import provider_key
from gateway.services.tenancy.authorization import VisibleWorkspaceScope, resolve_visible_workspace_scope
from gateway.services.tenancy.errors import TenancyForbiddenError, TenancyNotFoundError
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.workspace_scope import lookup_default_workspace_id


@dataclass(frozen=True)
class SessionCatalogScope:
    """How much of the catalog one dashboard session may be shown."""

    allowlist: list[str]
    """``model_access`` entries. Empty means nothing is reachable, which is a real
    answer: no configured instance and no usable key."""

    reads_default_workspace: bool
    """Whether the workspace-scoped alias and policy layers may be read.

    ``services/alias_service`` and ``services/policy_store`` read the deployment's
    default workspace when no workspace is named, which is what a master-key
    caller wants and what a session used to get. On a multi-tenant deployment that
    workspace belongs to one organization, so serving its stored aliases to
    everybody publishes another tenant's names: the allow-list does not catch
    them, because an alias pointing at a configured instance is a target every
    tenant may reach. True only when this caller may actually see that workspace,
    which on a single-tenant deployment is everyone in it.
    """


async def _byo_entries_for_workspaces(
    db: AsyncSession,
    *,
    organization_id: uuid.UUID,
    workspace_ids: list[uuid.UUID],
) -> set[str]:
    """BYO entries as the caller's own workspaces resolve them.

    Two queries whatever the number of workspaces: one for the candidates, one
    for the restrictions of the keys that survived. One key per (workspace,
    provider) is what a request would actually use, so ``resolve_active_key``
    picks it here too: a provider whose every key this workspace disabled
    contributes nothing, and a model restriction narrows the wildcard to the
    models it names. The union across the workspaces, not the intersection: a
    model one of them can call is a model this caller can call.
    """
    if not workspace_ids:
        return set()
    candidates = await WorkspaceProviderKeyOverrideRepository(db).candidates_for_workspaces(
        organization_id=organization_id, workspace_ids=workspace_ids
    )
    active: dict[tuple[uuid.UUID, uuid.UUID], str] = {}
    for workspace_id, rows in candidates.items():
        by_provider: dict[str, list[Candidate]] = defaultdict(list)
        for key, override in rows:
            by_provider[key.provider].append((key, override))
        for provider, group in by_provider.items():
            resolved = resolve_active_key(group)
            if resolved is not None:
                active[(workspace_id, resolved.id)] = provider

    restrictions = await WorkspaceProviderModelRestrictionRepository(db).list_for_workspace_keys(active)
    entries: set[str] = set()
    for (workspace_id, key_id), provider in active.items():
        prefix = provider_key(provider)
        allowed = restrictions.get((workspace_id, key_id))
        # An absent narrowing is not an empty allow-list: no restriction row means
        # every model of that provider (see ``WorkspaceProviderModelRestriction``).
        entries.update({f"{prefix}:{model}" for model in allowed} if allowed else {f"{prefix}:*"})
    return entries


async def _sees_default_workspace(db: AsyncSession, scope: VisibleWorkspaceScope) -> bool:
    """Whether the deployment's default workspace is one this caller may see.

    Looked up rather than created: ``default_workspace_id`` provisions one when
    none exists, which a catalog read must not do as a side effect.
    """
    default_workspace_id = await lookup_default_workspace_id(db)
    if default_workspace_id is None:
        return False
    if not scope.sees_every_workspace:
        return default_workspace_id in (scope.workspace_ids or [])
    owner = (
        await db.execute(
            select(col(Workspace.organization_id)).where(col(Workspace.id) == default_workspace_id)
        )
    ).scalar_one_or_none()
    return owner == scope.organization.id


async def resolve_session_catalog_scope(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: User,
    organizations: OrganizationService | None = None,
) -> SessionCatalogScope:
    """What this session identity may be shown. Unrestricted is the caller's own call.

    An owner or admin is answered from the organization's providers directly
    rather than by walking its workspaces. That is both truer to what they may do
    (a workspace's disable or model restriction is theirs to lift) and cheaper by
    a query.

    A caller with no live organization membership is answered with the configured
    instances rather than refused. That is the same rule applied to an empty
    tenant, not a fallback around one: they reach no BYO key because there is no
    organization holding any, and the configured instances are deployment-wide.
    The routers that exist to answer "which organization" still refuse such a
    caller; a catalog read is not one of them.
    """
    services = organizations if organizations is not None else OrganizationService(db)
    entries = {f"{instance}:*" for instance in config.providers}
    try:
        scope = await resolve_visible_workspace_scope(db, user=user, organizations=services)
    except (TenancyForbiddenError, TenancyNotFoundError):
        return SessionCatalogScope(allowlist=sorted(entries), reads_default_workspace=False)

    if scope.sees_every_workspace:
        providers = await OrgProviderKeyRepository(db).list_provider_names(scope.organization.id)
        entries.update(f"{provider_key(provider)}:*" for provider in providers)
    else:
        entries |= await _byo_entries_for_workspaces(
            db,
            organization_id=scope.organization.id,
            workspace_ids=scope.workspace_ids or [],
        )
    return SessionCatalogScope(
        allowlist=sorted(entries),
        reads_default_workspace=await _sees_default_workspace(db, scope),
    )


async def resolve_session_model_allowlist(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: User,
    organizations: OrganizationService | None = None,
) -> list[str]:
    """The allow-list half of :func:`resolve_session_catalog_scope`.

    For the callers that only decide whether one selector is reachable, which is
    every write guard: a target does not belong to a workspace until it is
    stored, so the alias and policy scoping the full result carries says nothing
    about it.
    """
    scope = await resolve_session_catalog_scope(db, config, user=user, organizations=organizations)
    return scope.allowlist


__all__ = ["SessionCatalogScope", "resolve_session_catalog_scope", "resolve_session_model_allowlist"]
