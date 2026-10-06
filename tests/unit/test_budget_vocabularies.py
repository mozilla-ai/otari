"""Guards on the budget vocabularies.

The named constants cover each ``Literal``.
The deployment route's scope table has a row for every scope type.
Both ceiling create bodies refuse a scope type outside the vocabulary.
"""

import pytest
from pydantic import BaseModel, ValidationError

from gateway.api.routes.scoped_budgets import _SCOPE_SUBJECTS
from gateway.models.budgets import (
    CYCLE_DAILY,
    CYCLE_DAYS,
    CYCLE_HOURS,
    CYCLE_MONTHLY,
    CYCLE_WEEKLY,
    CYCLE_YEARLY,
    INTERVAL_CYCLES,
    RESERVATION_ACTIVE,
    RESERVATION_EXPIRED,
    RESERVATION_RELEASED,
    RESERVATION_SETTLED,
    RESERVATION_STATUSES,
    RESET_CYCLES,
    SCOPE_API_TOKEN,
    SCOPE_ORG_MEMBER,
    SCOPE_ORGANIZATION,
    SCOPE_TYPES,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
)
from gateway.schemas.budgets import CreateScopedBudgetRequest, OrganizationScopedBudgetCreate


def test_the_cycle_constants_cover_the_literal() -> None:
    named = {CYCLE_HOURS, CYCLE_DAYS, CYCLE_DAILY, CYCLE_WEEKLY, CYCLE_MONTHLY, CYCLE_YEARLY}
    assert named == set(RESET_CYCLES)


def test_the_interval_cycles_are_cycles() -> None:
    # They are the two that carry an interval and an anchor, and the periods
    # module branches on membership rather than on the names, so a cycle added
    # to the tuple and not to the Literal would branch on nothing.
    assert set(INTERVAL_CYCLES) <= set(RESET_CYCLES)
    assert set(INTERVAL_CYCLES) == {CYCLE_HOURS, CYCLE_DAYS}


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
