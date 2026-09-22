"""The caller's organization's spend budgets and ceilings.

Both routers are mounted in standalone and hosted modes, and not in hybrid mode.
A budget is the figure and a ceiling is where it applies, so both routers live in one module.
The routes sit under ``/me``, because the caller's identity names the organization and no request parameter can.
No route here catches a domain error, because each error carries its own status.
A budget with no ``organization_id`` belongs to the deployment, and no route here lists, offers or repoints one.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import BudgetServiceDep, CurrentIdentity, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.schemas.budgets import (
    OrganizationBudgetCreate,
    OrganizationBudgetPublic,
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


@budgets_router.get("")
async def list_organization_budgets(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrganizationBudgetsPublic:
    """List the budgets this organization has defined. Owners and admins only."""
    return await service.list_organization_budgets(user=current_identity, skip=skip, limit=limit)


@budgets_router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_budget(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationBudgetCreate,
) -> OrganizationBudgetPublic:
    """Define a budget owned by this organization. Owners and admins only."""
    return await service.create_organization_budget(user=current_identity, request=body)


@budgets_router.patch("/{budget_id}")
async def update_organization_budget(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    budget_id: str,
    body: OrganizationBudgetUpdate,
) -> OrganizationBudgetPublic:
    """Change a budget's label, figure or period.

    Every ceiling naming it is held to the new figure from here on, which is the
    point of naming a budget rather than typing an amount per place it applies.
    """
    return await service.update_organization_budget(user=current_identity, budget_id=budget_id, request=body)


@budgets_router.delete("/{budget_id}")
async def delete_organization_budget(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    budget_id: str,
) -> Message:
    """Delete a budget, refused with 409 while a ceiling or workspace default names it."""
    await service.delete_organization_budget(user=current_identity, budget_id=budget_id)
    return Message(message="Budget deleted")


@ceilings_router.get("")
async def list_organization_spend_ceilings(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrganizationScopedBudgetsPublic:
    """List the ceilings capping identities inside this organization. Owners and admins only.

    A ceiling whose budget this organization does not own is listed with
    ``manageable`` false rather than omitted: it is enforcing against this
    organization's spend, so leaving it out would let the page read as uncapped.
    """
    return await service.list_organization_ceilings(user=current_identity, skip=skip, limit=limit)


@ceilings_router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_spend_ceiling(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationScopedBudgetCreate,
) -> OrganizationScopedBudgetPublic:
    """Cap one identity in this organization at one of its budgets.

    Answers 404 when the scope names nothing in this organization, rather than
    creating a ceiling that can never bind, and 404 when the budget is not this
    organization's.
    """
    return await service.create_organization_ceiling(user=current_identity, request=body)


@ceilings_router.patch("/{ceiling_id}")
async def update_organization_spend_ceiling(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    ceiling_id: str,
    body: OrganizationScopedBudgetUpdate,
) -> OrganizationScopedBudgetPublic:
    """Relabel a ceiling, or point it at a different budget of this organization's.

    The scope and the provider narrowing are not editable: changing either would
    move the ceiling to a different identity while carrying its spend, which is a
    delete and a create.
    """
    return await service.update_organization_ceiling(user=current_identity, ceiling_id=ceiling_id, request=body)


@ceilings_router.delete("/{ceiling_id}")
async def delete_organization_spend_ceiling(
    service: BudgetServiceDep,
    current_identity: CurrentIdentity,
    ceiling_id: str,
) -> Message:
    """Remove a ceiling inside this organization."""
    await service.delete_organization_ceiling(user=current_identity, ceiling_id=ceiling_id)
    return Message(message="Spend ceiling deleted")


__all__ = ["budgets_router", "ceilings_router"]
