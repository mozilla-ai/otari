"""The routing policies in force where the caller may see, for a tenant who does not operate the deployment.

``/v1/routing/policies`` is deployment-wide and operator-only, which is right:
it lists every tenant's stored policies and its ``workspace_id`` parameter is a
filter the client supplies, so nothing but the operator gate stands between a
signed-in member and another organization's routing (the escalation
otari-ai#1880 closed). The roles matrix still wants a member to *view* the
policies that shape their own requests (otari-ai#1942), so this router is a
second, narrower reading of the same rows, shaped like ``organization_usage.py``
(mozilla-ai/otari#837):

* **Scope is derived, never accepted.** It comes from the caller's own
  ``active_organization_id`` by way of ``resolve_visible_workspace_scope``,
  which refuses a pointer with no live membership behind it. No request here
  names an organization or a workspace.
* **How much of the organization** follows the rule the workspace list already
  uses: an owner, an admin or a superuser reads every workspace in it, and a
  member or viewer reads the ones they actively belong to. A member who belongs
  to no workspace still gets the config-file policies, which are deployment-wide
  and in force in every workspace they could ever join.

Reads only, one route. Writing a policy stays operator-only on the router above:
a policy decides which models a name reaches, so a caller who could write one
could widen their own access.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import ValidationError
from sqlalchemy import false, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.api.routes.routing import PolicyResponse
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import RoutingPolicy
from gateway.models.routing import PolicySpec
from gateway.models.tenancy import Workspace
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import resolve_visible_workspace_scope

router = APIRouter(
    prefix="/v1/organizations/me/routing-policies",
    tags=["routing"],
    # Authentication only, like the rest of the ``/v1/organizations/me`` surface.
    # What the caller may read is decided per request by the scope below, which
    # is the pattern the tenant-scoped routers already follow and the reason the
    # deployment operator gate does not belong here.
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def list_visible_routing_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    current_identity: CurrentIdentity,
) -> list[PolicyResponse]:
    """List the routing policies in force in the workspaces this caller may see.

    Stored policies from the caller's visible workspaces plus the config-file
    policies, which are deployment-wide and resolve in every workspace. The
    response is the shape ``GET /v1/routing/policies`` answers, narrowed to the
    caller's own organization; only an operator can write any of it.
    """
    scope = await resolve_visible_workspace_scope(db, user=current_identity, organizations=OrganizationService(db))
    statement = select(RoutingPolicy)
    if scope.sees_every_workspace:
        statement = statement.where(
            RoutingPolicy.workspace_id.in_(
                select(col(Workspace.id)).where(col(Workspace.organization_id) == scope.organization.id)
            )
        )
    elif scope.workspace_ids:
        statement = statement.where(RoutingPolicy.workspace_id.in_(scope.workspace_ids))
    else:
        # Belongs to no workspace yet: no stored row is theirs to see, and
        # deliberately not a 403 (nothing was refused). The config policies
        # below still apply to every workspace, so they are still listed.
        statement = statement.where(false())
    rows = (await db.execute(statement.order_by(RoutingPolicy.name))).scalars().all()
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
        policies.append(
            PolicyResponse(
                name=name,
                spec=spec.model_dump(mode="json", exclude_none=True),
                source="config",
                is_dynamic=spec.is_dynamic,
            )
        )
    return sorted(policies, key=lambda policy: (policy.name, str(policy.workspace_id or ""), policy.user_id or ""))
