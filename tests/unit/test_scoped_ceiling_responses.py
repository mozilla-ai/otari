"""Guards on the two scoped ceiling response models and the two update bodies."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from gateway.models.budgets import Budget, ScopedBudget
from gateway.schemas.budgets import (
    OrganizationScopedBudgetPublic,
    OrganizationScopedBudgetUpdate,
    ScopedBudgetResponse,
    UpdateScopedBudgetRequest,
)

_SHARED_FIELDS = [
    "id",
    "scope_type",
    "scope_id",
    "provider_key_id",
    "budget_id",
    "name",
    "max_budget",
    "current_spend",
    "reserved_spend",
    "token_limit",
    "current_tokens",
    "reserved_tokens",
    "request_limit",
    "current_requests",
    "reserved_requests",
    "budget_duration_sec",
    "reset_alignment",
    "period_start",
    "period_end",
    "created_at",
    "updated_at",
]
_OWNER = uuid.uuid4()


def _a_ceiling() -> ScopedBudget:
    return ScopedBudget(
        id="c1",
        scope_type="workspace",
        scope_id="w1",
        provider_key_id="openai",
        budget_id="b1",
        name="Team cap",
        current_spend=Decimal("1.25"),
        reserved_spend=Decimal("0.5"),
        current_tokens=10,
        reserved_tokens=2,
        current_requests=3,
        reserved_requests=1,
        period_start=datetime(2026, 9, 1, tzinfo=UTC),
        period_end=None,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 2, tzinfo=UTC),
    )


def _a_budget() -> Budget:
    return Budget(
        budget_id="b1",
        organization_id=_OWNER,
        max_budget=Decimal("20"),
        token_limit=1000,
        request_limit=None,
        budget_duration_sec=None,
        reset_alignment="month",
    )


def test_the_published_field_order_is_fixed() -> None:
    deployment = ScopedBudgetResponse.model_json_schema()
    organization = OrganizationScopedBudgetPublic.model_json_schema()

    assert list(deployment["properties"]) == _SHARED_FIELDS
    assert list(organization["properties"]) == [*_SHARED_FIELDS, "manageable"]
    assert deployment["required"] == list(deployment["properties"])
    assert organization["required"] == list(organization["properties"])


def test_a_ceiling_reads_its_counters_from_the_row_and_its_limits_from_the_budget() -> None:
    assert ScopedBudgetResponse.from_model(_a_ceiling(), _a_budget()).model_dump() == {
        "id": "c1",
        "scope_type": "workspace",
        "scope_id": "w1",
        "provider_key_id": "openai",
        "budget_id": "b1",
        "name": "Team cap",
        "max_budget": 20.0,
        "current_spend": 1.25,
        "reserved_spend": 0.5,
        "token_limit": 1000,
        "current_tokens": 10,
        "reserved_tokens": 2,
        "request_limit": None,
        "current_requests": 3,
        "reserved_requests": 1,
        "budget_duration_sec": None,
        "reset_alignment": "month",
        "period_start": "2026-09-01T00:00:00+00:00",
        "period_end": None,
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": "2026-08-02T00:00:00+00:00",
    }


def test_the_organization_ceiling_adds_only_manageable() -> None:
    deployment = ScopedBudgetResponse.from_model(_a_ceiling(), _a_budget()).model_dump()
    owned = OrganizationScopedBudgetPublic.from_model(_a_ceiling(), _a_budget(), organization_id=_OWNER)
    foreign = OrganizationScopedBudgetPublic.from_model(_a_ceiling(), _a_budget(), organization_id=uuid.uuid4())

    assert owned.model_dump() == {**deployment, "manageable": True}
    assert foreign.model_dump() == {**deployment, "manageable": False}


def test_the_two_update_bodies_publish_the_same_fields() -> None:
    deployment = UpdateScopedBudgetRequest.model_json_schema()["properties"]
    organization = OrganizationScopedBudgetUpdate.model_json_schema()["properties"]

    assert deployment == organization
    assert list(deployment) == ["budget_id", "name"]
