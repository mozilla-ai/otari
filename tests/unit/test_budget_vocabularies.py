"""Guards on the budget vocabularies.

The named constants cover each ``Literal``.
The deployment route's scope table has a row for every scope type.
Both ceiling create bodies refuse a scope type outside the vocabulary.
"""

import pytest
from pydantic import BaseModel, ValidationError

from gateway.api.routes.scoped_budgets import _SCOPE_SUBJECTS
from gateway.models.budgets import (
    ALIGN_DAY,
    ALIGN_MONTH,
    ALIGN_WEEK,
    RESERVATION_ACTIVE,
    RESERVATION_EXPIRED,
    RESERVATION_RELEASED,
    RESERVATION_SETTLED,
    RESERVATION_STATUSES,
    RESET_ALIGNMENTS,
    SCOPE_API_TOKEN,
    SCOPE_ORG_MEMBER,
    SCOPE_ORGANIZATION,
    SCOPE_TYPES,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
)
from gateway.schemas.budgets import CreateScopedBudgetRequest, OrganizationScopedBudgetCreate


def test_the_alignment_constants_cover_the_literal() -> None:
    assert {ALIGN_DAY, ALIGN_WEEK, ALIGN_MONTH} == set(RESET_ALIGNMENTS)


def test_the_scope_constants_cover_the_literal() -> None:
    named = {SCOPE_ORGANIZATION, SCOPE_WORKSPACE, SCOPE_WORKSPACE_MEMBER, SCOPE_ORG_MEMBER, SCOPE_API_TOKEN}
    assert named == set(SCOPE_TYPES)


def test_the_reservation_status_constants_cover_the_literal() -> None:
    named = {RESERVATION_ACTIVE, RESERVATION_SETTLED, RESERVATION_RELEASED, RESERVATION_EXPIRED}
    assert named == set(RESERVATION_STATUSES)


def test_the_deployment_route_resolves_every_scope_type() -> None:
    assert set(_SCOPE_SUBJECTS) == set(SCOPE_TYPES)


@pytest.mark.parametrize("create_body", [CreateScopedBudgetRequest, OrganizationScopedBudgetCreate])
def test_a_ceiling_create_body_refuses_an_unknown_scope(create_body: type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        create_body.model_validate({"scope_type": "team", "scope_id": "x", "budget_id": "b"})
