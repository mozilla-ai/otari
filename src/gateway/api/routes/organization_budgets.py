"""The caller's organization's spend budgets and ceilings (not hybrid mode).

Mounted by ``_register_core_routers`` for standalone **and hosted** deployments;
only hybrid mode has no management API. Spelled out rather than the "standalone
mode only" shorthand the neighbouring ``/api/v1/organizations/me`` routers use,
because on this surface the shorthand would name the wrong audience: a hosted
organization's admin is exactly who otari-ai#1943 added it for.

Thin composition over `gateway.services.tenancy.organization_budget_service`:
resolve the caller's identity, call the service, return its typed result. The
role gate, the scope resolution and the cross-tenant rules live there, and the
domain errors it raises carry their own statuses (see
`gateway.services.tenancy.errors`), so nothing here catches them.

Two routers in one module, as `routes/org_provider_keys.py` does, because they
are one feature: a budget is the figure and a ceiling is where it applies, and
splitting them across files would put the two halves of one page in two places.

Scoped to ``/me`` for the same reason `routes/organizations.py` is: a request
cannot name an organization at all, because the caller's identity already points
at one, so there is no parameter that could be confused with an authorization
decision.

These sit *beside* ``/api/v1/budgets`` and ``/api/v1/scoped-budgets``, which stay the
deployment's own surface behind ``require_deployment_operator``. The tables are
shared; what differs is which rows a caller may reach. A budget with no
``organization_id`` is the deployment's, and no route here lists, offers or
repoints one.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.services.tenancy.organization_budget_service import (
    OrganizationBudgetCreate,
    OrganizationBudgetPublic,
    OrganizationBudgetService,
    OrganizationBudgetsPublic,
    OrganizationBudgetUpdate,
    OrganizationScopedBudgetCreate,
    OrganizationScopedBudgetPublic,
    OrganizationScopedBudgetsPublic,
    OrganizationScopedBudgetUpdate,
)

# Master key on the router, as every standalone management router declares it.
# The role gate is a separate question answered in the service: the credential
# says a request is authenticated, the membership says whether that identity may
# set what this organization's members are allowed to spend.
budgets_router = APIRouter(
    prefix="/organizations/me/budgets",
    tags=["organization-budgets"],
    dependencies=[Depends(verify_master_key)],
)

ceilings_router = APIRouter(
    prefix="/organizations/me/spend-ceilings",
    tags=["organization-budgets"],
    dependencies=[Depends(verify_master_key)],
)


def get_organization_budget_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationBudgetService:
    """Build the service on the request's session."""
    return OrganizationBudgetService(db)


ServiceDep = Annotated[OrganizationBudgetService, Depends(get_organization_budget_service)]


@budgets_router.get("")
async def list_organization_budgets(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrganizationBudgetsPublic:
    """List the budgets this organization has defined. Owners and admins only."""
    return await service.list_budgets(user=current_identity, skip=skip, limit=limit)


@budgets_router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_budget(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationBudgetCreate,
) -> OrganizationBudgetPublic:
    """Define a budget owned by this organization. Owners and admins only."""
    return await service.create_budget(user=current_identity, request=body)


@budgets_router.patch("/{budget_id}")
async def update_organization_budget(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    budget_id: str,
    body: OrganizationBudgetUpdate,
) -> OrganizationBudgetPublic:
    """Change a budget's label, figure or period.

    Every ceiling naming it is held to the new figure from here on, which is the
    point of naming a budget rather than typing an amount per place it applies.
    """
    return await service.update_budget(user=current_identity, budget_id=budget_id, request=body)


@budgets_router.delete("/{budget_id}")
async def delete_organization_budget(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    budget_id: str,
) -> Message:
    """Delete a budget, refused with 409 while a ceiling or workspace default names it."""
    await service.delete_budget(user=current_identity, budget_id=budget_id)
    return Message(message="Budget deleted")


@ceilings_router.get("")
async def list_organization_spend_ceilings(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrganizationScopedBudgetsPublic:
    """List the ceilings capping identities inside this organization. Owners and admins only.

    A ceiling whose budget this organization does not own is listed with
    ``manageable`` false rather than omitted: it is enforcing against this
    organization's spend, so leaving it out would let the page read as uncapped.
    """
    return await service.list_ceilings(user=current_identity, skip=skip, limit=limit)


@ceilings_router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_spend_ceiling(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationScopedBudgetCreate,
) -> OrganizationScopedBudgetPublic:
    """Cap one identity in this organization at one of its budgets.

    Answers 404 when the scope names nothing in this organization, rather than
    creating a ceiling that can never bind, and 404 when the budget is not this
    organization's.
    """
    return await service.create_ceiling(user=current_identity, request=body)


@ceilings_router.patch("/{ceiling_id}")
async def update_organization_spend_ceiling(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    ceiling_id: str,
    body: OrganizationScopedBudgetUpdate,
) -> OrganizationScopedBudgetPublic:
    """Relabel a ceiling, or point it at a different budget of this organization's.

    The scope and the provider narrowing are not editable: changing either would
    move the ceiling to a different identity while carrying its spend, which is a
    delete and a create.
    """
    return await service.update_ceiling(user=current_identity, ceiling_id=ceiling_id, request=body)


@ceilings_router.delete("/{ceiling_id}")
async def delete_organization_spend_ceiling(
    service: ServiceDep,
    current_identity: CurrentIdentity,
    ceiling_id: str,
) -> Message:
    """Remove a ceiling inside this organization."""
    await service.delete_ceiling(user=current_identity, ceiling_id=ceiling_id)
    return Message(message="Spend ceiling deleted")


__all__ = ["budgets_router", "ceilings_router"]
