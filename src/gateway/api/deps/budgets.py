"""How a request reaches its budget service, and how startup work builds one of its own."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    get_unit_of_work,
)
from gateway.api.deps.api_keys import get_api_key_service
from gateway.api.deps.organizations import get_user_access
from gateway.core.database import get_db
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.budgets import BudgetRepositories
from gateway.services.budgets import BudgetService
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess


def get_budget_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetService:
    """Build the request's budget service on the request's Unit of Work."""
    organizations = OrganizationService(db, membership_listener=None)
    return BudgetService(
        uow,
        BudgetRepositories.on(uow),
        organizations,
        get_api_key_service(uow),
        WorkspaceAccess(db, organizations),
        users=get_user_access(uow),
    )


BudgetServiceDep = Annotated[BudgetService, Depends(get_budget_service)]


def build_budget_service(session: AsyncSession) -> BudgetService:
    """Build the budget service for startup work, on a Unit of Work over a session of its own."""
    return get_budget_service(get_unit_of_work(session), session)
