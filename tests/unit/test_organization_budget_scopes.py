"""An organization's ceiling refuses a scope type enforcement cannot resolve."""

import pytest
from pydantic import ValidationError

from gateway.services.tenancy.organization_budget_service import OrganizationScopedBudgetCreate


def test_the_create_body_refuses_an_unknown_scope() -> None:
    with pytest.raises(ValidationError):
        OrganizationScopedBudgetCreate.model_validate({"scope_type": "team", "scope_id": "x", "budget_id": "b"})
