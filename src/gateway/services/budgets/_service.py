from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import User
from gateway.repositories.budgets import BudgetRepositories
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
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets._organization_surface import _OrganizationSurface
from gateway.services.budgets._scopes import ScopeOwnership
from gateway.services.tenancy.organization_service import OrganizationService


class BudgetService:
    """The budgets domain's use cases: an organization's budgets and the spend ceilings that enforce them.

    Each public method is one business step, run in a block of the Unit of Work it was built on.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: BudgetRepositories,
        organizations: OrganizationService,
        api_keys: ApiKeyService,
    ) -> None:
        self._uow = uow
        self._organization = _OrganizationSurface(repositories, ScopeOwnership(organizations, api_keys), organizations)

    async def create_organization_budget(
        self, *, user: User, request: OrganizationBudgetCreate
    ) -> OrganizationBudgetPublic:
        """Create a budget owned by the caller's organization."""
        async with self._uow:
            return await self._organization.create_budget(user=user, request=request)

    async def create_organization_ceiling(
        self, *, user: User, request: OrganizationScopedBudgetCreate
    ) -> OrganizationScopedBudgetPublic:
        """Cap one identity inside the caller's organization at one of its budgets."""
        async with self._uow:
            return await self._organization.create_ceiling(user=user, request=request)

    async def delete_organization_budget(self, *, user: User, budget_id: str) -> None:
        """Delete a budget the caller's organization owns, unless something still names it."""
        async with self._uow:
            await self._organization.delete_budget(user=user, budget_id=budget_id)

    async def delete_organization_ceiling(self, *, user: User, ceiling_id: str) -> None:
        """Remove a ceiling inside the caller's organization."""
        async with self._uow:
            await self._organization.delete_ceiling(user=user, ceiling_id=ceiling_id)

    async def list_organization_budgets(
        self, *, user: User, skip: int = 0, limit: int = 100
    ) -> OrganizationBudgetsPublic:
        """Return a page of the caller's organization's budgets, with how many ceilings hold each."""
        async with self._uow:
            return await self._organization.list_budgets(user=user, skip=skip, limit=limit)

    async def list_organization_ceilings(
        self, *, user: User, skip: int = 0, limit: int = 100
    ) -> OrganizationScopedBudgetsPublic:
        """Return a page of the ceilings capping identities inside the caller's organization.

        A ceiling on a deployment budget is included, because it caps this organization's spend.
        """
        async with self._uow:
            return await self._organization.list_ceilings(user=user, skip=skip, limit=limit)

    async def update_organization_budget(
        self, *, user: User, budget_id: str, request: OrganizationBudgetUpdate
    ) -> OrganizationBudgetPublic:
        """Change a budget the caller's organization owns, and hold every ceiling naming it to the new figure."""
        async with self._uow:
            return await self._organization.update_budget(user=user, budget_id=budget_id, request=request)

    async def update_organization_ceiling(
        self, *, user: User, ceiling_id: str, request: OrganizationScopedBudgetUpdate
    ) -> OrganizationScopedBudgetPublic:
        """Relabel a ceiling inside the caller's organization, or point it at another budget the organization owns."""
        async with self._uow:
            return await self._organization.update_ceiling(user=user, ceiling_id=ceiling_id, request=request)


__all__ = ["BudgetService"]
