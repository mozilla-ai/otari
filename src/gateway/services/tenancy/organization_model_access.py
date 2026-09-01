"""Which models a dashboard session may reach, as a ``model_access`` allow-list.

``services/model_access`` answers this for an API key, off a stored column. A
session has no such column: its reach is decided by membership. This module
produces the answer in that module's own wire shape (canonical ``instance:model``
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
"""

import uuid
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.tenancy import User
from gateway.repositories.tenancy.org_provider_key_repository import (
    Candidate,
    OrgProviderKeyRepository,
    WorkspaceProviderKeyOverrideRepository,
    WorkspaceProviderModelRestrictionRepository,
    resolve_active_key,
)
from gateway.services.provider_kwargs import provider_key
from gateway.services.tenancy.authorization import resolve_visible_workspace_scope
from gateway.services.tenancy.errors import TenancyForbiddenError, TenancyNotFoundError
from gateway.services.tenancy.organization_service import OrganizationService


async def _byo_entries_for_workspaces(
    db: AsyncSession,
    *,
    organization_id: uuid.UUID,
    workspace_ids: list[uuid.UUID],
) -> set[str]:
    """BYO entries as the caller's own workspaces resolve them.

    One key per (workspace, provider) is what a request would actually use, so
    ``resolve_active_key`` picks it here too: a provider whose every key this
    workspace disabled contributes nothing, and a model restriction narrows the
    wildcard to the models it names. The union across the workspaces, not the
    intersection: a model one of them can call is a model this caller can call.
    """
    overrides = WorkspaceProviderKeyOverrideRepository(db)
    restrictions = WorkspaceProviderModelRestrictionRepository(db)
    entries: set[str] = set()
    for workspace_id in workspace_ids:
        candidates = await overrides.all_candidates(organization_id=organization_id, workspace_id=workspace_id)
        by_provider: dict[str, list[Candidate]] = defaultdict(list)
        for key, override in candidates:
            by_provider[key.provider].append((key, override))
        for provider, group in by_provider.items():
            active = resolve_active_key(group)
            if active is None:
                continue
            allowed = await restrictions.list_for_workspace_key(
                workspace_id=workspace_id, org_provider_key_id=active.id
            )
            prefix = provider_key(provider)
            # No restriction row means every model of that provider: an absent
            # narrowing is not an empty allow-list (see
            # ``WorkspaceProviderModelRestriction``).
            entries.update({f"{prefix}:{model}" for model in allowed} if allowed else {f"{prefix}:*"})
    return entries


async def resolve_session_model_allowlist(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: User,
    organizations: OrganizationService | None = None,
) -> list[str]:
    """The models this session identity may reach.

    Never ``None``: unrestricted is the deployment operator's answer, and each
    call site decides that before asking. An empty list is a real answer, and
    means the deployment configures no provider instance and the caller's
    organization holds no usable key, which is an empty catalog either way.

    An owner or admin is answered from the organization's providers directly
    rather than by walking its workspaces. Both are truer to what they may do (a
    workspace's disable or model restriction is theirs to lift) and bounded: the
    per-workspace walk below costs a query per workspace, which is fine for the
    handful a member belongs to and not for a large tenant's whole list.

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
        return sorted(entries)
    if scope.sees_every_workspace:
        providers = await OrgProviderKeyRepository(db).list_provider_names(scope.organization.id)
        entries.update(f"{provider_key(provider)}:*" for provider in providers)
    else:
        entries |= await _byo_entries_for_workspaces(
            db,
            organization_id=scope.organization.id,
            workspace_ids=scope.workspace_ids or [],
        )
    return sorted(entries)


__all__ = ["resolve_session_model_allowlist"]
