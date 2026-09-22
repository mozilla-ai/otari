"""A stored ceiling whose scope type this build does not know is refused before its scope is resolved."""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.exceptions.budget_exceptions import OrganizationScopedBudgetNotFoundError
from gateway.models.budgets import ScopedBudget
from gateway.models.tenancy import Organization
from gateway.repositories.budgets import BudgetRepositories, BudgetRepository, ScopedBudgetRepository
from gateway.services.budgets._organization_surface import _OrganizationSurface
from gateway.services.budgets._scopes import ScopeOwnership
from gateway.services.tenancy.organization_service import OrganizationService


@pytest.mark.asyncio
async def test_a_stored_scope_type_this_build_does_not_know_is_not_found() -> None:
    organization = Organization(id=uuid.uuid4(), name="Acme", slug="acme")
    ceiling = ScopedBudget(scope_type="galaxy", scope_id=str(organization.id), budget_id="budget-1")
    ceilings = Mock(spec=ScopedBudgetRepository)
    ceilings.get = AsyncMock(return_value=ceiling)
    scopes = Mock(spec=ScopeOwnership)
    surface = _OrganizationSurface(
        BudgetRepositories(budgets=Mock(spec=BudgetRepository), ceilings=ceilings),
        scopes,
        Mock(spec=OrganizationService),
    )

    with pytest.raises(OrganizationScopedBudgetNotFoundError):
        await surface._require_own_ceiling(organization=organization, ceiling_id=ceiling.id)

    scopes.get_organization_id_for.assert_not_called()
