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
  one. Where a workspace has no such key, a provider the hosted port serves
  contributes the models it advertises there, or ``provider:*`` when it
  advertises no roster.

An organization holding no BYO key still gets every configured instance, which on
a standalone deployment is the whole catalog.

The scope also answers one thing the allow-list cannot. Aliases and stored
policies are workspace-scoped rows, and the catalog reads them for a workspace
rather than filtering them by target, so a name in a workspace the caller may not
see would be listed even though every entry it resolves to is permitted. See
:attr:`SessionCatalogScope.reads_default_workspace`.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.models.provider_keys import (
    OrgProviderKey,
)
from gateway.models.tenancy import User, Workspace
from gateway.ports.model_provider_port import HostedModels, ModelProviderPort
from gateway.repositories.providers import OrgProviderKeyModelRepository
from gateway.repositories.tenancy.org_provider_key_repository import (
    OrgProviderKeyRepository,
    WorkspaceProviderModelRestrictionRepository,
)
from gateway.services.provider_kwargs import provider_key
from gateway.services.tenancy.authorization import VisibleWorkspaceScope, resolve_visible_workspace_scope
from gateway.services.tenancy.errors import TenancyForbiddenError, TenancyNotFoundError
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService, has_credential, key_is_usable
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.workspace_scope import lookup_default_workspace_id, organization_for_workspace_id


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

    deployment_supplied_providers: frozenset[str]
    """Hosted providers the deployment pays for in at least one of the organization's workspaces."""

    offered_keys: frozenset[str]
    """Canonical ``provider:model`` selectors the organization offers and serves.

    The allow-list above says what this caller *may* reach; this says what the
    catalog should *list*. The two are different questions for these models
    alone: discovery dials ``config.providers`` instances only, so a model on an
    organization's own key is permitted by the allow-list and listed by nothing.
    Empty for a caller with no offered models, which is every caller on a
    deployment where nobody has adopted any.
    """


async def resolve_hosted_models(
    model_provider: ModelProviderPort | None, organization_id: uuid.UUID | None
) -> HostedModels:
    """What the hosted port advertises, keyed by wire provider name. Empty with no port."""
    if model_provider is None:
        return {}
    hosted = await model_provider.get_hosted_models(organization_id=organization_id)
    return {provider_key(provider): roster for provider, roster in hosted.items()}


def hosted_allowlist_entries(hosted_models: HostedModels, providers: frozenset[str]) -> set[str]:
    """The allow-list entries the hosted ``providers`` contribute.

    A provider contributes each model advertised on it, so a model the
    deployment switched off leaves the catalog as it left dispatch; one that
    advertises no particular models contributes ``provider:*``.
    """
    entries: set[str] = set()
    for provider in providers:
        advertised = hosted_models.get(provider)
        if advertised is None:
            entries.add(f"{provider}:*")
        else:
            entries.update(f"{provider}:{model}" for model in advertised)
    return entries


async def resolve_organization_byo_providers(db: AsyncSession, organization_id: uuid.UUID | None) -> frozenset[str]:
    """The providers one organization holds a usable key of its own for, in any workspace.

    For a caller answered from the whole organization, a deployment operator
    acting in it. Looser than ``OrgProviderKeyService.get_byo_providers``, which
    asks whether *every* workspace calls the provider on its own key: this asks
    whether any could, which is what decides that what the deployment advertises
    must not narrow the organization's view of the provider.
    """
    if organization_id is None:
        return frozenset()
    live = await OrgProviderKeyRepository(db).list_live_keys(organization_id)
    return frozenset(provider_key(key.provider) for key in live if key_is_usable(key))


async def resolve_workspace_byo_providers(db: AsyncSession, workspace_id: uuid.UUID | None) -> frozenset[str]:
    """The providers one workspace calls on a key of its own.

    For a caller that dispatches from one workspace: an API key, or the master
    key in the deployment's default workspace. Narrowed exactly as dispatch is:
    the key active in that workspace, holding a credential, which is the one
    condition under which dispatch never asks the hosted port. A key the
    organization holds in another workspace does not count, because this
    caller cannot reach the provider on it.
    """
    if workspace_id is None:
        return frozenset()
    organization_id = await organization_for_workspace_id(db, workspace_id)
    if organization_id is None:
        return frozenset()
    active = await OrgProviderKeyService(db).get_active_keys(
        organization_id=organization_id, workspace_ids=[workspace_id]
    )
    return frozenset(
        provider_key(provider) for provider, key in active.get(workspace_id, {}).items() if has_credential(key)
    )


def _narrowed(
    prefix: str,
    offered: list[str] | None,
    restricted: list[str] | None,
) -> set[str]:
    """One key's entries, narrowed by whichever of the two allow-lists exist.

    Two independent narrowings over one key, and **absence means different
    things from presence in both**. No offered rows means the key has never been
    refreshed, so it still reaches everything its provider serves; no restriction
    rows means this workspace has not narrowed it. An *empty* list is the
    opposite answer in both cases, and it is reachable: an organization can
    switch every model off.

    They compose by intersection, never union, because each may only narrow. A
    workspace's restriction cannot reach a model the organization withdrew, and
    the organization offering a model does not lift a workspace's own list.
    """
    if offered is None and restricted is None:
        return {f"{prefix}:*"}
    if offered is None:
        allowed = set(restricted or ())
    elif restricted is None:
        allowed = set(offered)
    else:
        allowed = set(offered) & set(restricted)
    return {f"{prefix}:{model}" for model in allowed}


async def _get_byo_allowlist(db: AsyncSession, active_keys: dict[uuid.UUID, dict[str, OrgProviderKey]]) -> set[str]:
    """Returns the BYO allow-list the caller's workspaces resolve to.

    Two narrowings apply per key: the models the organization offers and serves
    on it, and the workspace's own restriction. The result is the union across
    the workspaces: a model one of them can call is a model this caller can call.
    """
    usable = {
        (workspace_id, key.id): provider
        for workspace_id, keys in active_keys.items()
        for provider, key in keys.items()
        if key_is_usable(key)
    }
    restrictions = await WorkspaceProviderModelRestrictionRepository(db).list_for_workspace_keys(usable)
    offered = await OrgProviderKeyModelRepository(db).enabled_models_for_keys({key for _, key in usable})
    allowlist: set[str] = set()
    for (workspace_id, key_id), provider in usable.items():
        allowlist |= _narrowed(
            provider_key(provider),
            offered.get(key_id),
            restrictions.get((workspace_id, key_id)),
        )
    return allowlist


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
        await db.execute(select(col(Workspace.organization_id)).where(col(Workspace.id) == default_workspace_id))
    ).scalar_one_or_none()
    return owner == scope.organization.id


async def resolve_workspace_offered_keys(db: AsyncSession, workspace_id: uuid.UUID | None) -> frozenset[str]:
    """The ``provider:model`` selectors one workspace's organization offers and serves.

    The API key's counterpart to :attr:`SessionCatalogScope.offered_keys`, and the
    reason it exists separately: ``catalog_scope`` answers an API key from the
    key's own stored allow-list without ever resolving its organization, so
    nothing on that path would otherwise know that the organization had adopted
    any models. ``GET /api/v1/models`` with an API key *is* the data-plane
    catalog, so leaving it out would list the adopted models to the dashboard and
    not to the callers that dispatch them.

    Narrowed exactly as dispatch narrows: the key names one workspace, so the
    keys active *in that workspace* are what it reaches.
    """
    if workspace_id is None:
        return frozenset()
    organization_id = await organization_for_workspace_id(db, workspace_id)
    if organization_id is None:
        return frozenset()
    active_keys = await OrgProviderKeyService(db).get_active_keys(
        organization_id=organization_id, workspace_ids=[workspace_id]
    )
    entries = await _get_byo_allowlist(db, active_keys)
    # A ``provider:*`` entry is an unnarrowed key: it says the workspace may
    # reach everything that provider serves and nothing about what to list.
    return frozenset(entry for entry in entries if not entry.endswith(":*"))


async def resolve_organization_offered_keys(db: AsyncSession, organization_id: uuid.UUID) -> frozenset[str]:
    """Every ``provider:model`` one organization offers and serves, across its keys.

    The unnarrowed answer, for a caller who is not acting inside one workspace: a
    deployment operator, or a master key, both of which read the catalog whole.
    No workspace restriction applies, because a narrowing one workspace set is
    not a narrowing of what such a caller may see; what the organization itself
    switched off still is.
    """
    live = [key for key in await OrgProviderKeyRepository(db).list_live_keys(organization_id) if key_is_usable(key)]
    if not live:
        return frozenset()
    offered = await OrgProviderKeyModelRepository(db).enabled_models_for_keys({key.id for key in live})
    entries: set[str] = set()
    for key in live:
        entries |= _narrowed(provider_key(key.provider), offered.get(key.id), None)
    return frozenset(entry for entry in entries if not entry.endswith(":*"))


async def resolve_all_organizations_offered_keys(db: AsyncSession) -> dict[uuid.UUID, frozenset[str]]:
    """Every organization's served ``provider:model`` selectors, keyed by organization.

    The selector index's read: it builds one view per organization in a single
    pass, so this is two queries for the deployment rather than two per tenant.
    An organization offering nothing is absent.
    """
    live = [key for key in await OrgProviderKeyRepository(db).list_all_live() if key_is_usable(key)]
    if not live:
        return {}
    offered = await OrgProviderKeyModelRepository(db).enabled_models_for_keys({key.id for key in live})
    by_organization: dict[uuid.UUID, set[str]] = {}
    for key in live:
        entries = _narrowed(provider_key(key.provider), offered.get(key.id), None)
        by_organization.setdefault(key.organization_id, set()).update(
            entry for entry in entries if not entry.endswith(":*")
        )
    return {organization: frozenset(keys) for organization, keys in by_organization.items() if keys}


async def workspace_organizations(db: AsyncSession) -> dict[uuid.UUID, uuid.UUID]:
    """Which organization each workspace belongs to."""
    rows = await db.execute(select(col(Workspace.id), col(Workspace.organization_id)))
    return {workspace_id: organization_id for workspace_id, organization_id in rows.all()}


async def resolve_default_workspace_offered_keys(db: AsyncSession) -> frozenset[str]:
    """What the deployment's own organization offers, for a master-key caller.

    ``lookup_default_workspace_id`` rather than ``default_workspace_id``: the
    latter provisions a workspace when none exists, which a catalog read must not
    do as a side effect.
    """
    workspace_id = await lookup_default_workspace_id(db)
    if workspace_id is None:
        return frozenset()
    organization_id = await organization_for_workspace_id(db, workspace_id)
    if organization_id is None:
        return frozenset()
    return await resolve_organization_offered_keys(db, organization_id)


async def resolve_session_catalog_scope(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: User,
    organizations: OrganizationService | None = None,
    model_provider: ModelProviderPort | None,
) -> SessionCatalogScope:
    """Returns what this session identity may be shown.

    An owner or admin is answered from the whole organization,
    because a workspace's disable or model restriction is theirs to lift.
    A member is answered from their own workspaces.
    A caller with no live organization membership gets the configured instances rather than a refusal.
    Passing ``None`` as ``model_provider`` lists no hosted providers.

    NOTE: this never answers unrestricted, so a caller should decide that for a deployment operator first.
    """
    services = organizations if organizations is not None else OrganizationService(db, membership_listener=None)
    allowlist = {f"{instance}:*" for instance in config.providers}
    try:
        scope = await resolve_visible_workspace_scope(db, user=user, organizations=services)
    except (TenancyForbiddenError, TenancyNotFoundError):
        return SessionCatalogScope(
            allowlist=sorted(allowlist),
            reads_default_workspace=False,
            deployment_supplied_providers=frozenset(),
            offered_keys=frozenset(),
        )

    provider_keys = OrgProviderKeyService(db)
    hosted_models = await resolve_hosted_models(model_provider, scope.organization.id)
    hosted = frozenset(hosted_models)
    if scope.sees_every_workspace:
        live_keys = [
            live
            for live in await OrgProviderKeyRepository(db).list_live_keys(scope.organization.id)
            if key_is_usable(live)
        ]
        # No workspace restriction applies here: this caller sees every workspace,
        # so a narrowing one of them set is not a narrowing of what they may see.
        # What the organization itself withdrew still is.
        offered = await OrgProviderKeyModelRepository(db).enabled_models_for_keys({live.id for live in live_keys})
        byo_allowlist: set[str] = set()
        for live in live_keys:
            byo_allowlist |= _narrowed(provider_key(live.provider), offered.get(live.id), None)
        reachable_hosted = hosted
    else:
        active_keys = await provider_keys.get_active_keys(
            organization_id=scope.organization.id, workspace_ids=scope.workspace_ids or []
        )
        byo_allowlist = await _get_byo_allowlist(db, active_keys)
        # Dispatch asks the port only in a workspace with no active key that has a credential.
        reachable_hosted = frozenset(
            provider
            for provider in hosted
            if any((key := keys.get(provider)) is None or not has_credential(key) for keys in active_keys.values())
        )
    byo_providers = (
        await provider_keys.get_byo_providers(organization_id=scope.organization.id) if hosted else frozenset()
    )
    allowlist |= byo_allowlist
    allowlist |= hosted_allowlist_entries(hosted_models, reachable_hosted)
    return SessionCatalogScope(
        allowlist=sorted(allowlist),
        reads_default_workspace=await _sees_default_workspace(db, scope),
        deployment_supplied_providers=hosted - byo_providers,
        # Only the entries naming one model. A ``provider:*`` is an unnarrowed
        # key, which says the organization may reach everything that provider
        # serves and nothing about which of them to list.
        offered_keys=frozenset(entry for entry in byo_allowlist if not entry.endswith(":*")),
    )


async def resolve_session_model_allowlist(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: User,
    organizations: OrganizationService | None = None,
    model_provider: ModelProviderPort | None,
) -> list[str]:
    """The allow-list half of :func:`resolve_session_catalog_scope`.

    For the callers that only decide whether one selector is reachable, which is
    every write guard: a target does not belong to a workspace until it is
    stored, so the alias and policy scoping the full result carries says nothing
    about it.
    """
    scope = await resolve_session_catalog_scope(
        db, config, user=user, organizations=organizations, model_provider=model_provider
    )
    return scope.allowlist


__all__ = [
    "SessionCatalogScope",
    "hosted_allowlist_entries",
    "resolve_all_organizations_offered_keys",
    "resolve_hosted_models",
    "resolve_organization_byo_providers",
    "resolve_session_catalog_scope",
    "resolve_session_model_allowlist",
    "resolve_workspace_byo_providers",
    "workspace_organizations",
]
