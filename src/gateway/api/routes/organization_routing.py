"""The routing configuration of the caller's own organization, for a tenant.

``/v1/routing/policies`` and ``/v1/aliases`` are deployment-wide and
operator-only, which is right: they list every tenant's stored rows and take a
``workspace_id`` the client supplies, so nothing but the operator gate stands
between a signed-in member and another organization's routing (the escalation
otari-ai#1880 closed). These routers are a second, narrower reading and writing
of the same tables, shaped like ``organization_usage.py``
(mozilla-ai/otari#837), giving the roles matrix what it asks for: View for a
member, Edit for an admin (otari-ai#1942, otari-ai#1969).

* **Scope is derived, never accepted.** The organization comes from the caller's
  own ``active_organization_id`` by way of ``resolve_visible_workspace_scope``,
  which refuses a pointer with no live membership behind it. No request here
  names an organization.
* **How much of it a read covers** follows the rule the workspace list uses: an
  owner, an admin or a superuser reads every workspace in the organization, and
  a member or viewer reads the ones they actively belong to. A member who
  belongs to no workspace still gets the config-file entries, which are
  deployment-wide and in force in every workspace they could ever join.
* **Both reads are bounded.** ``limit`` caps the stored rows a request
  materializes, and the config-file entries are appended within the same cap, so
  neither response grows with the tenant. The default is the cap rather than a
  page, because the dashboard renders policies and aliases as one sorted table
  and a page of one beside every row of the other would be incoherent; the bound
  is there for the organization that outgrows the page, not to paginate the page.
* **Workspace-wide rows only, on the reads as much as the writes.** A stored row
  may be scoped to one ``users.user_id`` as well as to a workspace, and neither
  half of this surface handles that scope. The write refuses ``user_id`` because
  it is a deployment-wide identifier; the read has to agree, or an admin is shown
  a row they cannot address. While these were listed, Delete on one carried no
  user scope, so it matched ``user_id IS NULL`` and destroyed the workspace-wide
  row of that name while answering 204. A tenant cannot interpret the identifier
  either: it names an API-key user, where a dashboard member is a tenancy UUID.
* **A write needs a named workspace, resolved inside the organization.** There is
  no default-workspace fallback here, unlike the operator routers: the
  deployment's default workspace is not the tenant's to write into unless it
  happens to be one of theirs, and a silent fallback is how it would become one.

Aliases and policies did not need an owner column for any of this. Both tables
already carry a non-nullable ``workspace_id``, and a workspace belongs to exactly
one organization, so the row's tenant is a join away. That is why this follows
mozilla-ai/otari#875's shape (a ``/v1/organizations/me/*`` router over the same
rows) without following its migration.

**What stops a tenant widening their own access.** A policy or an alias decides
which real model a name reaches, so an unconstrained write is a way to name a
model the tenant may not call. Every target here is put through the same
allow-list the catalog is filtered by
(``services/tenancy/organization_model_access``), so an admin can only point a
name at something their organization already reaches. That check, not the
operator gate, is what makes these writes safe to open.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy import Select, false, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.api.routes.aliases import (
    AliasRequest,
    AliasResponse,
    delete_alias_in_workspace,
    upsert_alias_in_workspace,
)
from gateway.api.routes.routing import (
    PolicyRequest,
    PolicyResponse,
    delete_policy_in_workspace,
    upsert_policy_in_workspace,
    validated_spec,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelAlias, RoutingPolicy
from gateway.models.routing import PolicySpec
from gateway.models.tenancy import User as TenancyUser
from gateway.models.tenancy import Workspace
from gateway.services.alias_service import all_alias_names
from gateway.services.model_access import is_model_allowed
from gateway.services.policy_store import all_policy_names
from gateway.services.provider_kwargs import normalize_pricing_key
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import (
    VisibleWorkspaceScope,
    resolve_visible_workspace_scope,
    resolve_workspace_in_organization,
)
from gateway.services.tenancy.organization_model_access import resolve_session_model_allowlist

# Authentication only, like the rest of the ``/v1/organizations/me`` surface.
# What the caller may read or write is decided per request below, which is the
# pattern the tenant-scoped routers already follow and the reason the deployment
# operator gate does not belong here.
policies_router = APIRouter(
    prefix="/v1/organizations/me/routing-policies",
    tags=["routing"],
    dependencies=[Depends(verify_master_key)],
)

aliases_router = APIRouter(
    prefix="/v1/organizations/me/aliases",
    tags=["aliases"],
    dependencies=[Depends(verify_master_key)],
)

# The cap on either list, and its default. High enough that no real organization's
# routing configuration is truncated by it, low enough that one row per name is
# never an unbounded read.
_MAX_ROWS = 1000


async def _writable_workspace_id(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: TenancyUser,
    workspace_id: uuid.UUID | None,
    targets: list[str],
) -> uuid.UUID:
    """Resolve the workspace a tenant write lands in, refusing what it may not do.

    Three refusals, in the order that keeps each one from leaking what the next
    would have told the caller. Management access first, so a member learns
    nothing about which workspaces exist. Then the workspace, resolved inside the
    caller's own organization, so another tenant's id is a 404 rather than a 403.
    Then the targets, against the organization's own reach.
    """
    organizations = OrganizationService(db)
    organization = await organizations.get_active_organization_for_user(user)
    await organizations.require_active_organization_management_access(user=user, organization=organization)
    if workspace_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="workspace_id is required: an organization's routing entries belong to one of its workspaces.",
        )
    workspace = await resolve_workspace_in_organization(
        db,
        user=user,
        workspace_id=workspace_id,
        organization=organization,
        organizations=organizations,
    )
    await _require_reachable_targets(db, config, user=user, targets=targets)
    return workspace.id


async def _require_reachable_targets(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    user: TenancyUser,
    targets: list[str],
) -> None:
    """Refuse a target the caller's organization cannot already reach.

    Without this, writing a policy would be a way to reach a provider the
    organization holds no key for: the name is the tenant's to choose, and
    resolution follows the name. Answered as a 400 naming the target, because it
    is a statement about the body rather than about the caller's role, and the
    catalog the dashboard offers already excludes these.

    A target that names another alias or policy is left alone, because the write
    helpers refuse chaining a step later and say so precisely. Checking it here
    first would answer "not available to this organization", which is true of an
    indirection name and useless for fixing it.
    """
    if not targets:
        return
    indirections = all_alias_names(config) | all_policy_names(config)
    allowlist = await resolve_session_model_allowlist(db, config, user=user)
    unreachable = [
        target
        for target in targets
        if target not in indirections and not is_model_allowed(allowlist, normalize_pricing_key(config, target))
    ]
    if unreachable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Target(s) {', '.join(sorted(unreachable))} are not available to this organization. "
                "Add a provider key for them first, or pick a model from the catalog."
            ),
        )


def _reject_user_scope(user_id: str | None) -> None:
    """Refuse a user-scoped entry on the tenant surface.

    A user-scoped row names a ``users.user_id``, which is a deployment-wide
    identifier with nothing in it saying whose it is, so accepting one would make
    this an existence oracle across tenants. The operator router keeps that scope;
    an organization admin writes workspace-wide, which is what the roles matrix
    asks of them.
    """
    if user_id is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id is not accepted here: an organization's entries are workspace-wide.",
        )


def _visible_workspace_ids(scope: VisibleWorkspaceScope) -> Select[tuple[uuid.UUID]]:
    """The workspace ids whose stored rows this caller may be shown.

    A subquery rather than a clause over one table, so the policies list and the
    aliases list express one rule over their two. An owner or admin is a
    predicate over ``organization_id`` rather than an ``IN`` list, so it does not
    grow with the tenant.
    """
    statement = select(col(Workspace.id))
    if scope.sees_every_workspace:
        return statement.where(col(Workspace.organization_id) == scope.organization.id)
    if scope.workspace_ids:
        return statement.where(col(Workspace.id).in_(scope.workspace_ids))
    # Belongs to no workspace yet: no stored row is theirs to see, and
    # deliberately not a 403 (nothing was refused). The config entries below are
    # deployment-wide, so they are still listed.
    return statement.where(false())


_LIMIT = Query(ge=1, le=_MAX_ROWS, description="Maximum entries to return, stored and config-file together.")


@policies_router.get("")
async def list_visible_routing_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
    limit: Annotated[int, _LIMIT] = _MAX_ROWS,
) -> list[PolicyResponse]:
    """List the routing policies in force in the workspaces this caller may see.

    Stored policies from the caller's visible workspaces plus the config-file
    policies, which are deployment-wide and resolve in every workspace. The
    response is the shape ``GET /v1/routing/policies`` answers, narrowed to the
    caller's own organization.
    """
    scope = await resolve_visible_workspace_scope(db, user=current_identity, organizations=OrganizationService(db))
    statement = select(RoutingPolicy).where(
        col(RoutingPolicy.workspace_id).in_(_visible_workspace_ids(scope)),
        # Workspace-wide only; see the module docstring for why a user-scoped row
        # is neither this surface's to show nor its to act on.
        col(RoutingPolicy.user_id).is_(None),
    )
    rows = (await db.execute(statement.order_by(RoutingPolicy.name).limit(limit))).scalars().all()
    policies = []
    for row in rows:
        try:
            parsed = PolicySpec.model_validate(row.spec)
        except ValidationError:
            # Listed rather than hidden, matching the operator view: hiding a row
            # this build cannot parse would make the dashboard disagree with what
            # actually resolves.
            logger.warning("Stored routing policy %r does not validate; listing it as-is", row.name)
            policies.append(PolicyResponse.from_model(row, is_dynamic=False))
            continue
        policies.append(PolicyResponse.from_model(row, is_dynamic=parsed.is_dynamic))
    # Config last, matching the operator list: deployment-wide, in force in every
    # workspace, listed once and unscoped.
    for name, spec in config.routing.policies.items():
        if len(policies) >= limit:
            break
        policies.append(
            PolicyResponse(
                name=name,
                spec=spec.model_dump(mode="json", exclude_none=True),
                source="config",
                is_dynamic=spec.is_dynamic,
            )
        )
    return sorted(policies, key=lambda policy: (policy.name, str(policy.workspace_id or ""), policy.user_id or ""))


@policies_router.post("")
async def set_organization_routing_policy(
    request: PolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
) -> PolicyResponse:
    """Create or update a stored policy in one of the organization's workspaces.

    Organization owners and admins only. ``workspace_id`` is required and must
    name a workspace of the caller's own organization; ``user_id`` is not
    accepted here.
    """
    _reject_user_scope(request.user_id)
    workspace_id = await _writable_workspace_id(
        db,
        config,
        user=current_identity,
        workspace_id=request.workspace_id,
        targets=validated_spec(request.name, request.spec).static_selectors(),
    )
    return await upsert_policy_in_workspace(request, db, config, workspace_id=workspace_id)


@policies_router.delete("/{name:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization_routing_policy(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Delete the policy in this workspace of the caller's organization."),
    ] = None,
) -> None:
    """Delete a stored policy from one of the organization's workspaces. Owners and admins only."""
    resolved = await _writable_workspace_id(
        db,
        config,
        user=current_identity,
        workspace_id=workspace_id,
        targets=[],
    )
    await delete_policy_in_workspace(name, db, config, workspace_id=resolved, user_id=None)


@aliases_router.get("")
async def list_visible_aliases(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
    limit: Annotated[int, _LIMIT] = _MAX_ROWS,
) -> list[AliasResponse]:
    """List the aliases in force in the workspaces this caller may see.

    The policies list's sibling, over ``model_aliases``, and scoped the same way:
    stored rows from the caller's visible workspaces, plus the config-file
    aliases, which are deployment-wide.
    """
    scope = await resolve_visible_workspace_scope(db, user=current_identity, organizations=OrganizationService(db))
    statement = select(ModelAlias).where(
        col(ModelAlias.workspace_id).in_(_visible_workspace_ids(scope)),
        col(ModelAlias.user_id).is_(None),
    )
    rows = (await db.execute(statement.order_by(ModelAlias.name).limit(limit))).scalars().all()
    aliases = [AliasResponse.from_model(row) for row in rows]
    for name, target in config.aliases.items():
        if len(aliases) >= limit:
            break
        aliases.append(AliasResponse(name=name, target=target, source="config"))
    return sorted(aliases, key=lambda alias: (alias.name, str(alias.workspace_id or ""), alias.user_id or ""))


@aliases_router.post("")
async def set_organization_alias(
    request: AliasRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
) -> AliasResponse:
    """Create or update a stored alias in one of the organization's workspaces.

    Organization owners and admins only, with the same two scope rules the policy
    write has: ``workspace_id`` is required and resolved inside the caller's
    organization, and ``user_id`` is not accepted.
    """
    _reject_user_scope(request.user_id)
    workspace_id = await _writable_workspace_id(
        db,
        config,
        user=current_identity,
        workspace_id=request.workspace_id,
        targets=[request.target],
    )
    return await upsert_alias_in_workspace(request, db, config, workspace_id=workspace_id)


@aliases_router.delete("/{name:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization_alias(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Delete the alias in this workspace of the caller's organization."),
    ] = None,
) -> None:
    """Delete a stored alias from one of the organization's workspaces. Owners and admins only."""
    resolved = await _writable_workspace_id(
        db,
        config,
        user=current_identity,
        workspace_id=workspace_id,
        targets=[],
    )
    await delete_alias_in_workspace(name, db, config, workspace_id=resolved, user_id=None)
