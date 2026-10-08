"""The tenant-scoped budget and spend-ceiling endpoints, and their tenant boundary.

Two halves, for the reason `test_organization_pricing_routes.py` gives.

The **HTTP surface** is exercised through the client, which can only ever act as
the one superuser operator identity a standalone deployment provisions. That is
enough for the statuses, the shapes, and the rules that do not depend on who is
asking (a period stated twice, a duplicate ceiling, a delete refused while
something names the budget).

The **rules that decide who may reach what** are exercised at the service layer,
with identities built at whatever role and in whatever organization a case needs.
That is where the interesting failures live, and they are the point of this
surface existing: before otari-ai#1943 a tenant could reach neither budgets nor
ceilings, and the risk in opening them is that an admin reaches *another*
tenant's.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyValidationError
from gateway.exceptions.budget_exceptions import (
    OrganizationBudgetHeldElsewhereError,
    OrganizationBudgetInUseError,
    OrganizationBudgetNotFoundError,
    OrganizationScopedBudgetAlreadyExistsError,
    OrganizationScopedBudgetNotFoundError,
    OrganizationScopeNotFoundError,
)
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget, BudgetResetLog, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.models.users import User as ApiUser
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.schemas.budgets import (
    AppliedEntity,
    OrganizationBudgetCreate,
    OrganizationBudgetUpdate,
    OrganizationScopedBudgetCreate,
    OrganizationScopedBudgetUpdate,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.workspace_listener import NullWorkspaceListener

_BUDGETS = f"{API_ROOT}/organizations/me/budgets"
_CEILINGS = f"{API_ROOT}/organizations/me/spend-ceilings"

# `HTTP_422_UNPROCESSABLE_CONTENT`, not `..._ENTITY`, in the schema-refusal
# assertions below. The two are both 422; `_ENTITY` is the deprecated alias and
# reading it emits a `StarletteDeprecationWarning` (verified against Starlette
# 1.3.1). Noted here because the swap has been suggested twice in review, in the
# wrong direction each time.


# =============================================================================
# The HTTP surface
# =============================================================================


def _budget_body(**overrides: Any) -> dict[str, Any]:
    """A create body, monthly unless a case names another cycle.

    Naming a cycle replaces the cadence rather than merging into it: each cycle
    carries exactly its own settings, so a default day-of-month carried onto a
    weekly body is the stranding the server refuses, and every case would have to
    remember to clear it.
    """
    body: dict[str, Any] = {"name": "Engineering monthly", "max_budget": 250.0}
    if "reset_cycle" in overrides:
        return {**body, **overrides}
    return {**body, "reset_cycle": "monthly", "reset_month_day": 1, **overrides}


def test_a_budget_is_created_listed_changed_and_deleted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    created = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    budget = created.json()
    assert budget["max_budget"] == 250.0
    assert budget["reset_cycle"] == "monthly"
    # Nothing names it yet, which is what makes it deletable below.
    assert budget["ceiling_count"] == 0
    # Stamped with the caller's organization rather than left for the client to
    # claim: the request cannot name one at all.
    assert budget["organization_id"]

    listed = client.get(_BUDGETS, headers=master_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    assert listed.json()["count"] == 1
    assert listed.json()["data"][0]["budget_id"] == budget["budget_id"]

    changed = client.patch(
        f"{_BUDGETS}/{budget['budget_id']}",
        json={"max_budget": 500.0, "name": "Engineering monthly (raised)"},
        headers=master_key_header,
    )
    assert changed.status_code == status.HTTP_200_OK, changed.text
    assert changed.json()["max_budget"] == 500.0
    # Untouched by a patch that did not mention it.
    assert changed.json()["reset_cycle"] == "monthly"

    deleted = client.delete(f"{_BUDGETS}/{budget['budget_id']}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_200_OK, deleted.text
    assert client.get(_BUDGETS, headers=master_key_header).json()["count"] == 0


def test_a_budget_refuses_another_cycles_settings(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The CHECKs as a 400 naming the offending field, not a 500.

    This replaces the old two-period-sources case: a budget no longer carries a
    duration and an alignment, it carries a cycle, and the state worth refusing
    is a cycle holding a setting it does not take.
    """
    refused = client.post(
        _BUDGETS,
        json=_budget_body(reset_cycle="daily", reset_month_day=15),
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
    assert "does not take reset_month_day" in refused.json()["detail"]


def test_a_budget_refuses_a_cycle_missing_its_own_settings(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The other direction, which is the one a form gets wrong."""
    refused = client.post(
        _BUDGETS,
        json=_budget_body(reset_cycle="weekly"),
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
    assert "needs reset_weekdays" in refused.json()["detail"]


def test_a_patch_that_changes_the_cycle_clears_the_old_one_s_settings(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The obvious request is the one that has to work.

    A budget moved from monthly to daily keeps no day-of-month: carrying it would
    refuse the write for a field the caller never mentioned, and leaving it
    stored would be what the budget silently reverts to on the next switch back.
    """
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    assert budget["reset_month_day"] == 1

    changed = client.patch(
        f"{_BUDGETS}/{budget['budget_id']}",
        json={"reset_cycle": "daily"},
        headers=master_key_header,
    )
    assert changed.status_code == status.HTTP_200_OK, changed.text
    assert changed.json()["reset_cycle"] == "daily"
    assert changed.json()["reset_month_day"] is None


def test_a_patch_that_keeps_the_cycle_keeps_its_settings(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The other half: an omitted field contributes what is stored."""
    budget = client.post(
        _BUDGETS,
        json=_budget_body(reset_cycle="monthly", reset_month_day=15),
        headers=master_key_header,
    ).json()

    renamed = client.patch(
        f"{_BUDGETS}/{budget['budget_id']}",
        json={"name": "Renamed"},
        headers=master_key_header,
    )
    assert renamed.status_code == status.HTTP_200_OK, renamed.text
    assert renamed.json()["reset_month_day"] == 15


@pytest.mark.parametrize("alignment", ["calendar_month", "fortnightly", "DAILY", ""])
def test_an_unrecognized_reset_cycle_is_refused_on_the_request(
    client: TestClient,
    master_key_header: dict[str, str],
    alignment: str,
) -> None:
    """422 on the request that introduces it, not 500 on the next window derivation.

    An unrecognized alignment stores happily and then raises out of
    ``period_window`` the first time a window is derived from it, which is
    creating a ceiling or retiming one after a cadence change. "weekly" is the
    plausible mistake: it is a period name a caller would reasonably try, and it
    is not one of the three calendar boundaries.
    """
    refused = client.post(
        _BUDGETS,
        json={"name": "Monthly", "max_budget": 100.0, "reset_cycle": alignment},
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


def test_an_unrecognized_reset_cycle_is_refused_on_an_update(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The update path too, which is the one that reaches the retiming."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    refused = client.patch(
        f"{_BUDGETS}/{budget['budget_id']}",
        json={"reset_cycle": "fortnightly"},
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


def test_every_recognized_cycle_is_accepted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The other half, so the CHECKs cannot creep into refusing a valid cycle."""
    for cycle, settings in (
        ("daily", {}),
        ("weekly", {"reset_weekdays": 0b10001}),
        ("monthly", {"reset_month_day": 15}),
        ("yearly", {"reset_month": 3, "reset_month_day": 1}),
        ("every_n_hours", {"reset_every_n": 6, "reset_anchor_at": "2026-01-01T00:00:00Z"}),
        ("every_n_days", {"reset_every_n": 14, "reset_anchor_at": "2026-01-01T00:00:00Z"}),
    ):
        created = client.post(
            _BUDGETS,
            json={"name": cycle, "max_budget": 10.0, "reset_cycle": cycle, **settings},
            headers=master_key_header,
        )
        assert created.status_code == status.HTTP_201_CREATED, created.text
        assert created.json()["reset_cycle"] == cycle


def test_a_gateway_user_may_not_be_created_on_an_organizations_budget(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """404, the same one an id naming nothing gets.

    A ``users`` row is deployment-wide with no tenancy column, so a tenant's
    budget is never the right cap for one: the admin editing its figure would be
    moving what a gateway user may spend, and neither side can see the coupling.
    """
    tenant = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    user_id = f"holder-{uuid.uuid4().hex[:8]}"

    refused = client.post(
        f"{API_ROOT}/users",
        json={"user_id": user_id, "budget_id": tenant["budget_id"]},
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text
    assert refused.json()["detail"] == f"Budget with id '{tenant['budget_id']}' not found"

    # Refused whole, rather than the user landing uncapped.
    assert client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header).status_code == status.HTTP_404_NOT_FOUND


def test_a_gateway_user_may_not_be_moved_onto_an_organizations_budget(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The other assignment site, and the cap it already had survives the refusal."""
    deployment = client.post(f"{API_ROOT}/budgets", json={"max_budget": 10.0}, headers=master_key_header).json()
    tenant = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    user_id = f"holder-{uuid.uuid4().hex[:8]}"
    created = client.post(
        f"{API_ROOT}/users",
        json={"user_id": user_id, "budget_id": deployment["budget_id"]},
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_200_OK, created.text

    refused = client.patch(
        f"{API_ROOT}/users/{user_id}",
        json={"budget_id": tenant["budget_id"]},
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text

    still_capped = client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header)
    assert still_capped.json()["budget_id"] == deployment["budget_id"]


def test_an_unknown_budget_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    missing = client.patch(f"{_BUDGETS}/nope", json={"name": "x"}, headers=master_key_header)
    assert missing.status_code == status.HTTP_404_NOT_FOUND, missing.text


def test_a_ceiling_is_created_listed_relabeled_and_deleted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    organization_id = budget["organization_id"]

    created = client.post(
        _CEILINGS,
        json={
            "scope_type": "organization",
            "scope_id": organization_id,
            "budget_id": budget["budget_id"],
            "name": "Whole org",
        },
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    ceiling = created.json()
    # The figure and the period are read through the budget and carried here, so
    # a page can render a ceiling without fetching every budget to resolve one id.
    assert ceiling["max_budget"] == 250.0
    assert ceiling["reset_cycle"] == "monthly"
    # A calendar-aligned budget opens a window immediately, rather than on first
    # spend, so a periodic cap has a defined end before any request arrives.
    assert ceiling["period_start"] is not None
    assert ceiling["period_end"] is not None
    # This organization's own budget, so its figure is changeable here.
    assert ceiling["manageable"] is True

    listed = client.get(_CEILINGS, headers=master_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    assert [row["id"] for row in listed.json()["data"]] == [ceiling["id"]]

    relabeled = client.patch(
        f"{_CEILINGS}/{ceiling['id']}",
        json={"name": None},
        headers=master_key_header,
    )
    assert relabeled.status_code == status.HTTP_200_OK, relabeled.text
    # An explicit null clears the label back to unnamed, which a create can produce.
    assert relabeled.json()["name"] is None

    deleted = client.delete(f"{_CEILINGS}/{ceiling['id']}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_200_OK, deleted.text
    assert client.get(_CEILINGS, headers=master_key_header).json()["count"] == 0


def test_the_ceiling_list_narrows_to_one_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The budget detail page reads only its own budget's ceilings, counted on the server."""
    whole_org = client.post(_BUDGETS, json=_budget_body(name="Whole org"), headers=master_key_header).json()
    other = client.post(_BUDGETS, json=_budget_body(name="Other"), headers=master_key_header).json()
    created = client.post(
        _CEILINGS,
        json={
            "scope_type": "organization",
            "scope_id": whole_org["organization_id"],
            "budget_id": whole_org["budget_id"],
        },
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text

    mine = client.get(_CEILINGS, params={"budget_id": whole_org["budget_id"]}, headers=master_key_header).json()
    none = client.get(_CEILINGS, params={"budget_id": other["budget_id"]}, headers=master_key_header).json()
    unknown = client.get(_CEILINGS, params={"budget_id": "no-such-budget"}, headers=master_key_header).json()

    assert (mine["count"], [row["id"] for row in mine["data"]]) == (1, [created.json()["id"]])
    assert (none["count"], none["data"]) == (0, [])
    assert (unknown["count"], unknown["data"]) == (0, [])


def test_deleting_a_budget_another_organization_is_capped_by_answers_409(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The status a refusal reaches the client with, and a message naming no other tenant."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    created = client.post(
        f"{API_ROOT}/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": str(uuid.uuid4()), "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    if created.status_code != status.HTTP_200_OK:
        pytest.skip("the operator route refuses an unknown scope here; the service-level test covers this")

    refused = client.delete(f"{_BUDGETS}/{budget['budget_id']}", headers=master_key_header)
    assert refused.status_code == status.HTTP_409_CONFLICT, refused.text
    assert "outside this organization" in refused.json()["detail"]


def test_a_second_ceiling_on_one_scope_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The entity unique index, reported as words a caller can act on."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    body = {
        "scope_type": "organization",
        "scope_id": budget["organization_id"],
        "budget_id": budget["budget_id"],
    }
    assert client.post(_CEILINGS, json=body, headers=master_key_header).status_code == status.HTTP_201_CREATED

    refused = client.post(_CEILINGS, json=body, headers=master_key_header)
    assert refused.status_code == status.HTTP_409_CONFLICT, refused.text


def test_each_model_of_a_provider_carries_its_own_ceiling(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """One ceiling per entity: the provider and each of its models are distinct entities, a repeat is not."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    scope = {"scope_type": "organization", "scope_id": budget["organization_id"], "budget_id": budget["budget_id"]}

    for narrowing in (
        {"provider_key_id": "openai"},
        *({"provider_key_id": "openai", "model": m} for m in ("gpt-4o", "o3")),
    ):
        created = client.post(_CEILINGS, json={**scope, **narrowing}, headers=master_key_header)
        assert created.status_code == status.HTTP_201_CREATED, created.text
        assert created.json()["model"] == narrowing.get("model")

    repeat = {**scope, "provider_key_id": "openai", "model": "gpt-4o"}
    assert client.post(_CEILINGS, json=repeat, headers=master_key_header).status_code == status.HTTP_409_CONFLICT
    orphan = {**scope, "model": "gpt-4o"}
    refused = client.post(_CEILINGS, json=orphan, headers=master_key_header)
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


def test_a_ceiling_on_an_unknown_scope_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Refused rather than created, because a scope naming nothing never binds.

    A ceiling on a typo is created, listed, and silently unenforced, with nothing
    anywhere to surface it. Same rule ``POST /api/v1/scoped-budgets`` states.
    """
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    refused = client.post(
        _CEILINGS,
        json={
            "scope_type": "workspace",
            "scope_id": str(uuid.uuid4()),
            "budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text


def test_an_unknown_scope_type_is_refused_by_the_schema(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The ``Literal`` is what publishes the values and refuses the rest."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    refused = client.post(
        _CEILINGS,
        json={"scope_type": "galaxy", "scope_id": "x", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_provider_narrowing_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    blank: str,
) -> None:
    """A ceiling narrowed to nothing would be created, listed, and never enforced.

    ``applicable_budgets`` matches ``provider_key_id == provider_instance OR IS
    NULL``, and a blank string is neither: it stores as a narrowed row and binds
    to no request ever. That is the same permissive-direction failure a scope
    naming nothing has, so it is refused at the schema rather than normalized,
    since folding it into null would quietly cap *more* than the caller asked for.
    """
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    refused = client.post(
        _CEILINGS,
        json={
            "scope_type": "organization",
            "scope_id": budget["organization_id"],
            "provider_key_id": blank,
            "budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


def test_an_omitted_provider_narrowing_still_caps_every_provider(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The other half of the rule above: absent is the aggregate cap, and stays so."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    created = client.post(
        _CEILINGS,
        json={
            "scope_type": "organization",
            "scope_id": budget["organization_id"],
            "budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    assert created.json()["provider_key_id"] is None


def test_deleting_a_budget_removes_the_ceilings_applying_it(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """One step: the dashboard confirms what stops being capped, and the delete does the rest."""
    budget = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()
    client.post(
        _CEILINGS,
        json={
            "scope_type": "organization",
            "scope_id": budget["organization_id"],
            "budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    )

    deleted = client.delete(f"{_BUDGETS}/{budget['budget_id']}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_200_OK, deleted.text
    assert client.get(_CEILINGS, headers=master_key_header).json()["count"] == 0
    assert client.get(_BUDGETS, headers=master_key_header).json()["count"] == 0


def test_the_deployment_budget_list_is_not_this_one(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A budget defined here is the organization's, and one defined there is not.

    The two surfaces share the table, so this pins the filter rather than
    assuming it: ``/api/v1/budgets`` sees everything, and this list sees only rows
    carrying the caller's organization.
    """
    deployment = client.post(f"{API_ROOT}/budgets", json={"name": "Deployment wide"}, headers=master_key_header)
    assert deployment.status_code == status.HTTP_200_OK, deployment.text
    tenant = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    scoped = client.get(_BUDGETS, headers=master_key_header).json()
    assert [row["budget_id"] for row in scoped["data"]] == [tenant["budget_id"]]

    everything = client.get(f"{API_ROOT}/budgets", headers=master_key_header).json()
    assert {row["budget_id"] for row in everything} == {
        deployment.json()["budget_id"],
        tenant["budget_id"],
    }
    # And each row says which it is, so the operator's assignment control can
    # withhold the one `POST /api/v1/users` would refuse.
    owners = {row["budget_id"]: row["organization_id"] for row in everything}
    assert owners[deployment.json()["budget_id"]] is None
    assert owners[tenant["budget_id"]] == tenant["organization_id"]


def test_a_ceiling_may_not_name_a_deployment_budget(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """404, and the same 404 an id naming nothing gets.

    This is the rule that keeps the surface honest: a deployment budget is not
    the organization's to enforce with, because editing its figure afterwards
    would move a cap the deployment set.
    """
    deployment = client.post(f"{API_ROOT}/budgets", json={"name": "Deployment wide"}, headers=master_key_header).json()
    tenant = client.post(_BUDGETS, json=_budget_body(), headers=master_key_header).json()

    refused = client.post(
        _CEILINGS,
        json={
            "scope_type": "organization",
            "scope_id": tenant["organization_id"],
            "budget_id": deployment["budget_id"],
        },
        headers=master_key_header,
    )
    assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text


# =============================================================================
# Who may reach what
#
# At the service layer, because the API can only act as the deployment's one
# operator identity, which is a superuser and an owner everywhere.
# =============================================================================


async def _seeded[Row](db: AsyncSession, row: Row) -> Row:
    # A step that fails rolls the session back, which expires every attached row, so the tests hold detached ones.
    await db.commit()
    db.expunge(row)
    return row


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    organization = await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )
    return await _seeded(db, organization)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
        is_superuser=False,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return await _seeded(db, user)


async def _workspace(db: AsyncSession, organization: Organization, *, name: str, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return await _seeded(db, workspace)


def _create(**overrides: Any) -> OrganizationBudgetCreate:
    """A create request, monthly unless a case names another cycle.

    Naming a cycle replaces the cadence rather than merging into it, for the
    reason `_budget_body` gives.
    """
    fields: dict[str, Any] = {"name": "Monthly", "max_budget": 100.0}
    if "reset_cycle" in overrides:
        return OrganizationBudgetCreate(**{**fields, **overrides})
    return OrganizationBudgetCreate(**{**fields, "reset_cycle": "monthly", "reset_month_day": 1, **overrides})


def _service(async_db: AsyncSession) -> BudgetService:
    uow = UnitOfWork(async_db)
    return BudgetService(
        uow,
        BudgetRepositories.on(uow),
        OrganizationService(async_db, membership_listener=None, workspace_listener=NullWorkspaceListener()),
        ApiKeyService(ApiKeyRepository(uow)),
        WorkspaceAccess(
            async_db,
            OrganizationService(async_db, membership_listener=None, workspace_listener=NullWorkspaceListener()),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["owner", "admin"])
async def test_a_management_role_may_define_a_budget(async_db: AsyncSession, role: str) -> None:
    organization = await _organization(async_db, slug=f"acme-write-{role}")
    identity = await _member(async_db, organization, role=role, full_name=f"{role} person")

    created = await _service(async_db).create_organization_budget(user=identity, request=_create())

    assert created.organization_id == organization.id
    assert created.max_budget == 100.0


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_a_non_management_role_may_not_define_a_budget(async_db: AsyncSession, role: str) -> None:
    organization = await _organization(async_db, slug=f"acme-refuse-{role}")
    identity = await _member(async_db, organization, role=role, full_name=f"{role} person")

    with pytest.raises(NotAuthorizedError):
        await _service(async_db).create_organization_budget(user=identity, request=_create())


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_a_non_management_role_may_not_even_read_them(async_db: AsyncSession, role: str) -> None:
    """Stricter than the pricing overrides, and deliberately.

    The roles matrix puts Spend & budgets at Hidden for a member, where Model
    pricing is a read they get: a rate is what *you* are billed at, and a cap is
    a statement about what colleagues may spend.
    """
    organization = await _organization(async_db, slug=f"acme-read-{role}")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    reader = await _member(async_db, organization, role=role, full_name=f"{role} reader")
    service = _service(async_db)
    await service.create_organization_budget(user=owner, request=_create())

    with pytest.raises(NotAuthorizedError):
        await service.list_organization_budgets(user=reader)

    with pytest.raises(NotAuthorizedError):
        await service.list_organization_ceilings(user=reader)


@pytest.mark.asyncio
async def test_an_admin_may_not_reach_another_organizations_budget(async_db: AsyncSession) -> None:
    """The core cross-tenant rule on the budget half.

    Not found rather than forbidden: another tenant's row must be
    indistinguishable from one that was never created, or the response is an
    existence oracle over their spend configuration.
    """
    theirs = await _organization(async_db, slug="globex-budget")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their owner")
    mine = await _organization(async_db, slug="acme-budget")
    my_admin = await _member(async_db, mine, role="admin", full_name="My admin")
    service = _service(async_db)
    their_budget = await service.create_organization_budget(user=their_owner, request=_create())

    with pytest.raises(OrganizationBudgetNotFoundError):
        await service.update_organization_budget(
            user=my_admin,
            budget_id=their_budget.budget_id,
            request=OrganizationBudgetUpdate(max_budget=1.0),
        )

    with pytest.raises(OrganizationBudgetNotFoundError):
        await service.delete_organization_budget(user=my_admin, budget_id=their_budget.budget_id)

    # And it is not in their list either, which is the read half of the same rule.
    assert (await service.list_organization_budgets(user=my_admin)).count == 0


@pytest.mark.asyncio
async def test_an_admin_may_not_cap_another_organizations_workspace(async_db: AsyncSession) -> None:
    """The core cross-tenant rule on the ceilings half.

    A scope id travels as a bare uuid with nothing in it saying whose it is, so
    an unresolved one is the whole of the risk here.
    """
    theirs = await _organization(async_db, slug="globex-scope")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their owner")
    their_workspace = await _workspace(async_db, theirs, name="Theirs", owner=their_owner)
    mine = await _organization(async_db, slug="acme-scope")
    my_admin = await _member(async_db, mine, role="admin", full_name="My admin")
    service = _service(async_db)
    my_budget = await service.create_organization_budget(user=my_admin, request=_create())

    with pytest.raises(OrganizationScopeNotFoundError):
        await service.create_organization_ceiling(
            user=my_admin,
            request=OrganizationScopedBudgetCreate(
                scope_type="workspace",
                scope_id=str(their_workspace.id),
                budget_id=my_budget.budget_id,
            ),
        )


@pytest.mark.asyncio
async def test_every_scope_kind_resolves_to_its_organization(async_db: AsyncSession) -> None:
    """All five, because each resolves through a different table.

    A scope kind whose resolution is wrong in the permissive direction is a
    cross-tenant hole, and one wrong in the strict direction is a cap an admin
    cannot set. Both are worth one assertion each.
    """
    organization = await _organization(async_db, slug="acme-scopes")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    membership = (
        (await async_db.execute(select(OrganizationMember).where(col(OrganizationMember.user_id) == owner.id)))
        .scalars()
        .first()
    )
    assert membership is not None
    workspace_member = (
        (await async_db.execute(select(WorkspaceMember).where(col(WorkspaceMember.workspace_id) == workspace.id)))
        .scalars()
        .first()
    )
    assert workspace_member is not None
    key = APIKey(id="sk-scope-test", key_hash="hash-scope-test", workspace_id=workspace.id)
    async_db.add(key)
    await async_db.flush()

    service = _service(async_db)
    scopes = {
        "organization": str(organization.id),
        "workspace": str(workspace.id),
        "org_member": str(membership.id),
        "workspace_member": str(workspace_member.id),
        "api_token": key.id,
    }
    for scope_type, scope_id in scopes.items():
        budget = await service.create_organization_budget(user=owner, request=_create(name=f"For {scope_type}"))
        created = await service.create_organization_ceiling(
            user=owner,
            request=OrganizationScopedBudgetCreate(
                scope_type=scope_type,  # type: ignore[arg-type]
                scope_id=scope_id,
                budget_id=budget.budget_id,
            ),
        )
        assert created.scope_type == scope_type
        assert created.manageable is True

    assert (await service.list_organization_ceilings(user=owner)).count == len(scopes)


@pytest.mark.asyncio
async def test_a_listed_budget_names_what_it_applies_to(async_db: AsyncSession) -> None:
    """Named where the entity has a name, narrowed where it is narrowed, and never another tenant's scope."""
    organization = await _organization(async_db, slug="acme-applied")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Research", owner=owner)
    membership = (
        (await async_db.execute(select(OrganizationMember).where(col(OrganizationMember.user_id) == owner.id)))
        .scalars()
        .one()
    )
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    unused = await service.create_organization_budget(user=owner, request=_create(name="Unused"))
    for scope_type, scope_id, provider in [
        ("organization", str(organization.id), None),
        ("workspace", str(workspace.id), "openai"),
        ("org_member", str(membership.id), None),
    ]:
        await service.create_organization_ceiling(
            user=owner,
            request=OrganizationScopedBudgetCreate(
                scope_type=scope_type,  # type: ignore[arg-type]
                scope_id=scope_id,
                provider_key_id=provider,
                budget_id=budget.budget_id,
            ),
        )

    # An operator may point another organization's workspace at this budget; that ID must not reach this admin.
    other = await _organization(async_db, slug="other-applied")
    other_owner = await _member(async_db, other, role="owner", full_name="Other owner")
    foreign = await _workspace(async_db, other, name="Elsewhere", owner=other_owner)
    async_db.add(ScopedBudget(scope_type="workspace", scope_id=str(foreign.id), budget_id=budget.budget_id))
    await async_db.commit()

    listed = {row.budget_id: row for row in (await service.list_organization_budgets(user=owner)).data}

    assert [(e.scope_type, e.provider_key_id, e.name) for e in listed[budget.budget_id].applied_to] == [
        ("organization", None, "Acme-Applied"),
        ("workspace", "openai", "Research"),
        ("org_member", None, None),
    ]
    assert listed[budget.budget_id].ceiling_count == 4
    assert listed[unused.budget_id].applied_to == []


@pytest.mark.asyncio
async def test_giving_a_cadence_to_a_budget_that_had_none_retimes_its_ceilings(async_db: AsyncSession) -> None:
    """The case that is an enforcement bug rather than a cosmetic one.

    ``_roll_expired_periods`` only ever updates a row whose ``period_end`` is not
    null, so a ceiling left with a NULL window under a periodic budget never rolls
    at all: it accumulates spend forever while this API reports the new cadence.
    """
    organization = await _organization(async_db, slug="acme-cadence-none")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(
        user=owner,
        request=_create(name="No reset", reset_cycle=None, max_budget=50.0),
    )
    ceiling = await service.create_organization_ceiling(
        user=owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(organization.id),
            budget_id=budget.budget_id,
        ),
    )
    assert ceiling.period_start is None
    assert ceiling.period_end is None

    await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(reset_cycle="monthly", reset_month_day=1),
    )

    retimed = (await service.list_organization_ceilings(user=owner)).data[0]
    assert retimed.period_start is not None
    assert retimed.period_end is not None
    assert retimed.reset_cycle == "monthly"


@pytest.mark.asyncio
async def test_taking_a_cadence_away_clears_the_window(async_db: AsyncSession) -> None:
    """The other direction, which would otherwise roll once at a stale boundary.

    A ceiling keeping an old ``period_end`` under a budget that no longer resets
    would zero its spend the moment that boundary passed, for no reason anyone
    could point at.
    """
    organization = await _organization(async_db, slug="acme-cadence-drop")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    await service.create_organization_ceiling(
        user=owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(organization.id),
            budget_id=budget.budget_id,
        ),
    )

    await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(reset_cycle=None),
    )

    cleared = (await service.list_organization_ceilings(user=owner)).data[0]
    assert cleared.period_start is None
    assert cleared.period_end is None


@pytest.mark.asyncio
async def test_retiming_keeps_the_spend_already_recorded(async_db: AsyncSession) -> None:
    """A cadence change is not a reset.

    Same rule repointing a ceiling at a different budget already follows: the
    window moves, the counters do not, so the allowance is held to a different
    figure from here on rather than handed back.
    """
    organization = await _organization(async_db, slug="acme-cadence-spend")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    created = await service.create_organization_ceiling(
        user=owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(organization.id),
            budget_id=budget.budget_id,
        ),
    )
    stored = await async_db.get(ScopedBudget, created.id)
    assert stored is not None
    stored.current_spend = Decimal("7.5")
    stored.reserved_spend = Decimal("1.25")
    await async_db.flush()

    await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(reset_cycle="daily"),
    )

    kept = (await service.list_organization_ceilings(user=owner)).data[0]
    assert kept.current_spend == 7.5
    # Untouched, so a hold taken before the change still releases against the
    # counter it was taken from.
    assert kept.reserved_spend == 1.25


@pytest.mark.asyncio
async def test_a_change_that_is_not_the_cadence_leaves_the_window_alone(async_db: AsyncSession) -> None:
    """Renaming or repricing must not restart a period.

    Retiming on every update would throw away the part of the period a ceiling
    had already spent, every time somebody fixed a typo in a budget's name.
    """
    organization = await _organization(async_db, slug="acme-cadence-stable")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    created = await service.create_organization_ceiling(
        user=owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(organization.id),
            budget_id=budget.budget_id,
        ),
    )

    await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(name="Renamed", max_budget=999.0),
    )

    unchanged = (await service.list_organization_ceilings(user=owner)).data[0]
    assert unchanged.period_start == created.period_start
    assert unchanged.period_end == created.period_end
    # The figure did move, which is the whole point of naming a budget.
    assert unchanged.max_budget == 999.0


@pytest.mark.asyncio
async def test_a_ceiling_on_a_deployment_budget_is_listed_but_not_manageable(async_db: AsyncSession) -> None:
    """The state the otari-ai cutover leaves behind, described honestly.

    It writes ceilings naming budgets it shares across tenants by shape, so they
    carry no owner. Omitting them would let the page read as uncapped while the
    organization was in fact capped; reporting them as manageable would offer an
    edit the surface refuses.
    """
    organization = await _organization(async_db, slug="acme-migrated")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    deployment_budget = Budget(name="Shaped by the cutover", max_budget=None, reset_cycle="daily")
    async_db.add(deployment_budget)
    await async_db.flush()
    async_db.add(
        ScopedBudget(
            scope_type="organization",
            scope_id=str(organization.id),
            budget_id=deployment_budget.budget_id,
        )
    )
    await async_db.flush()
    service = _service(async_db)

    listed = await service.list_organization_ceilings(user=owner)

    assert listed.count == 1
    assert listed.data[0].manageable is False
    # Its real figures, so the page is not lying about what binds.
    assert listed.data[0].reset_cycle == "daily"


@pytest.mark.asyncio
async def test_such_a_ceiling_can_be_moved_onto_the_organizations_own_budget(async_db: AsyncSession) -> None:
    """The way out of that state, per ceiling, without touching the shared budget.

    Repointing never edits the budget the ceiling used to name: enforcement reads
    through a budget, so editing one would silently re-cap every other ceiling
    naming it, possibly in another tenant.
    """
    organization = await _organization(async_db, slug="acme-repoint")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    deployment_budget = Budget(name="Shared", max_budget=None, reset_cycle="daily")
    async_db.add(deployment_budget)
    await async_db.flush()
    ceiling = ScopedBudget(
        scope_type="organization",
        scope_id=str(organization.id),
        budget_id=deployment_budget.budget_id,
    )
    async_db.add(ceiling)
    await async_db.flush()
    service = _service(async_db)
    mine = await service.create_organization_budget(user=owner, request=_create())

    moved = await service.update_organization_ceiling(
        user=owner,
        ceiling_id=ceiling.id,
        request=OrganizationScopedBudgetUpdate(budget_id=mine.budget_id),
    )

    assert moved.manageable is True
    assert moved.max_budget == 100.0
    # The budget it stopped naming is untouched, still holding what it held.
    still_there = await async_db.get(Budget, deployment_budget.budget_id)
    assert still_there is not None
    assert still_there.reset_cycle == "daily"


@pytest.mark.asyncio
async def test_an_admin_may_not_repoint_a_ceiling_at_a_foreign_budget(async_db: AsyncSession) -> None:
    """The same rule on the update path, which is the one an attacker would try.

    Creating a ceiling checks the budget; so must repointing one, or the check is
    a formality that one extra request walks around.
    """
    theirs = await _organization(async_db, slug="globex-repoint")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their owner")
    mine = await _organization(async_db, slug="acme-repoint-refuse")
    my_owner = await _member(async_db, mine, role="owner", full_name="My owner")
    service = _service(async_db)
    their_budget = await service.create_organization_budget(user=their_owner, request=_create())
    my_budget = await service.create_organization_budget(user=my_owner, request=_create())
    ceiling = await service.create_organization_ceiling(
        user=my_owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(mine.id),
            budget_id=my_budget.budget_id,
        ),
    )

    with pytest.raises(OrganizationBudgetNotFoundError):
        await service.update_organization_ceiling(
            user=my_owner,
            ceiling_id=ceiling.id,
            request=OrganizationScopedBudgetUpdate(budget_id=their_budget.budget_id),
        )


@pytest.mark.asyncio
async def test_an_admin_may_not_delete_another_organizations_ceiling(async_db: AsyncSession) -> None:
    theirs = await _organization(async_db, slug="globex-ceiling")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their owner")
    mine = await _organization(async_db, slug="acme-ceiling")
    my_admin = await _member(async_db, mine, role="admin", full_name="My admin")
    service = _service(async_db)
    their_budget = await service.create_organization_budget(user=their_owner, request=_create())
    theirs_ceiling = await service.create_organization_ceiling(
        user=their_owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(theirs.id),
            budget_id=their_budget.budget_id,
        ),
    )

    with pytest.raises(OrganizationScopedBudgetNotFoundError):
        await service.delete_organization_ceiling(user=my_admin, ceiling_id=theirs_ceiling.id)

    with pytest.raises(OrganizationScopedBudgetNotFoundError):
        await service.update_organization_ceiling(
            user=my_admin,
            ceiling_id=theirs_ceiling.id,
            request=OrganizationScopedBudgetUpdate(name="mine now"),
        )

    # And it is not in their list, which is what the page would show.
    assert (await service.list_organization_ceilings(user=my_admin)).count == 0


@pytest.mark.asyncio
async def test_a_delete_is_refused_while_a_workspace_default_names_the_budget(async_db: AsyncSession) -> None:
    """The other RESTRICT holder, counted separately so the message can say which."""
    organization = await _organization(async_db, slug="acme-default-hold")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    async_db.add(WorkspaceBudgetDefault(workspace_id=workspace.id, budget_id=budget.budget_id))
    await async_db.flush()

    with pytest.raises(OrganizationBudgetInUseError, match="workspace member default"):
        await service.delete_organization_budget(user=owner, budget_id=budget.budget_id)


@pytest.mark.asyncio
async def test_a_concurrent_duplicate_ceiling_is_a_conflict_not_a_crash(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The race the pre-check cannot close, translated rather than escaping as a 500.

    The pre-check is a read, so two creates can both pass it and one commit then
    loses to the partial unique index. Simulated deterministically by inserting
    the colliding row after the pre-check would have run and before the commit,
    which is the same ordering a concurrent writer produces.
    """
    organization = await _organization(async_db, slug="acme-race")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    request = OrganizationScopedBudgetCreate(
        scope_type="organization",
        scope_id=str(organization.id),
        budget_id=budget.budget_id,
    )

    original = service._organization._require_no_existing_ceiling

    async def insert_the_winner(candidate: OrganizationScopedBudgetCreate) -> None:
        await original(candidate)
        # The competing writer, landing between the check and this call's commit.
        async_db.add(
            ScopedBudget(
                scope_type=candidate.scope_type,
                scope_id=candidate.scope_id,
                budget_id=candidate.budget_id,
            )
        )
        await async_db.flush()

    # Through `monkeypatch` rather than a bare assignment, which mypy refuses on
    # a bound method and which would leave the patch in place if the assertion
    # below raised.
    monkeypatch.setattr(service._organization, "_require_no_existing_ceiling", insert_the_winner)

    with pytest.raises(OrganizationScopedBudgetAlreadyExistsError):
        await service.create_organization_ceiling(user=owner, request=request)


@pytest.mark.asyncio
async def test_an_explicit_null_clears_the_cap_as_the_schema_says(async_db: AsyncSession) -> None:
    """The behavior the update model's description used to deny.

    Each field is tri-state on `model_fields_set`, so an explicit null is a value:
    it clears the cap back to uncapped rather than being ignored. The description
    said clearing was a delete, which is published in the OpenAPI schema and would
    have sent a client to the wrong endpoint.
    """
    organization = await _organization(async_db, slug="acme-clear-cap")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create(max_budget=100.0))
    assert budget.max_budget == 100.0

    cleared = await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(max_budget=None),
    )

    assert cleared.max_budget is None
    # And an omitted field is still left alone, which is what makes it a patch.
    renamed = await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(name="Uncapped"),
    )
    assert renamed.max_budget is None
    assert renamed.name == "Uncapped"


@pytest.mark.asyncio
async def test_a_delete_is_refused_while_a_gateway_user_holds_the_budget(async_db: AsyncSession) -> None:
    """The hold the assignment sites no longer create, and still have to refuse.

    Since otari#881 neither `/api/v1/users` site will point a gateway user at a
    tenant's budget, so a row in this shape was assigned before that. Seeded
    directly for the same reason. `Budget.users` is a plain relationship, so an
    unchecked delete does not fail on it: the ORM nulls the column out, and an
    admin who cannot see that table removes the operator's assignment with it.
    """
    organization = await _organization(async_db, slug="acme-user-hold")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    async_db.add(ApiUser(user_id="capped-user", budget_id=budget.budget_id))
    await async_db.flush()

    with pytest.raises(OrganizationBudgetHeldElsewhereError):
        await service.delete_organization_budget(user=owner, budget_id=budget.budget_id)


@pytest.mark.asyncio
async def test_a_refused_delete_leaves_the_ceilings_applying_the_budget(async_db: AsyncSession) -> None:
    """A refused delete leaves the ceilings applying the budget where they were."""
    organization = await _organization(async_db, slug="acme-delete-atomic")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(
        user=owner,
        request=_create(applied_to=[AppliedEntity(scope_type="organization", scope_id=str(organization.id))]),
    )
    async_db.add(ApiUser(user_id="capped-elsewhere", budget_id=budget.budget_id))
    await async_db.flush()

    with pytest.raises(OrganizationBudgetHeldElsewhereError):
        await service.delete_organization_budget(user=owner, budget_id=budget.budget_id)

    assert len((await service.list_organization_ceilings(user=owner)).data) == 1


@pytest.mark.asyncio
async def test_a_delete_clears_the_budget_reset_history(async_db: AsyncSession) -> None:
    """A reset record outlives the assignment that produced it, and goes with the budget.

    A user can detach after a reset, leaving no live `users.budget_id` while the
    `budget_reset_logs` row remains. Refusing on it would strand the budget: the
    deployment route will not delete a tenant's budget, so no one could.
    """
    organization = await _organization(async_db, slug="acme-reset-hold")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    async_db.add(ApiUser(user_id="detached-user", budget_id=None))
    await async_db.flush()
    async_db.add(
        BudgetResetLog(
            user_id="detached-user",
            budget_id=budget.budget_id,
            previous_spend=Decimal("1.5"),
            reset_at=datetime.now(UTC),
        )
    )
    await async_db.flush()

    await service.delete_organization_budget(user=owner, budget_id=budget.budget_id)

    remaining = await async_db.execute(select(BudgetResetLog).where(BudgetResetLog.budget_id == budget.budget_id))
    assert remaining.all() == []


@pytest.mark.asyncio
async def test_a_scope_id_that_is_not_a_uuid_is_not_found_rather_than_a_crash(async_db: AsyncSession) -> None:
    """A typo and another tenant's row have to be the same answer, including a malformed id."""
    organization = await _organization(async_db, slug="acme-malformed")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())

    with pytest.raises(OrganizationScopeNotFoundError):
        await service.create_organization_ceiling(
            user=owner,
            request=OrganizationScopedBudgetCreate(
                scope_type="workspace",
                scope_id="not-a-uuid",
                budget_id=budget.budget_id,
            ),
        )


@pytest.mark.asyncio
async def test_a_second_ceiling_narrowed_to_a_provider_is_allowed(async_db: AsyncSession) -> None:
    """Two axes, so one scope may hold an aggregate cap and a per-provider one.

    A request must pass every row that applies to it, which is why the aggregate
    and the narrowed row are separate ceilings rather than a conflict.
    """
    organization = await _organization(async_db, slug="acme-two-axes")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    budget = await service.create_organization_budget(user=owner, request=_create())
    await service.create_organization_ceiling(
        user=owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(organization.id),
            budget_id=budget.budget_id,
        ),
    )

    narrowed = await service.create_organization_ceiling(
        user=owner,
        request=OrganizationScopedBudgetCreate(
            scope_type="organization",
            scope_id=str(organization.id),
            provider_key_id="openai-eu",
            budget_id=budget.budget_id,
        ),
    )

    assert narrowed.provider_key_id == "openai-eu"
    assert (await service.list_organization_ceilings(user=owner)).count == 2

    with pytest.raises(OrganizationScopedBudgetAlreadyExistsError):
        await service.create_organization_ceiling(
            user=owner,
            request=OrganizationScopedBudgetCreate(
                scope_type="organization",
                scope_id=str(organization.id),
                provider_key_id="openai-eu",
                budget_id=budget.budget_id,
            ),
        )


@pytest.mark.asyncio
async def test_an_unknown_scope_type_reaching_the_service_is_a_validation_error(async_db: AsyncSession) -> None:
    """Unreachable through the routes, where the ``Literal`` refuses it first.

    Asserted because the service is importable on its own and a caller inside
    this process is not held to the request schema.
    """
    organization = await _organization(async_db, slug="acme-bad-scope")
    service = _service(async_db)

    with pytest.raises(TenancyValidationError):
        await service._organization._require_scope_in_organization(
            organization=organization,
            scope_type="galaxy",
            scope_id=str(organization.id),
        )


# =============================================================================
# One save for a budget and the entities it applies to
# =============================================================================


async def _applied(service: BudgetService, owner: User) -> set[tuple[str, str, str | None, str | None]]:
    ceilings = (await service.list_organization_ceilings(user=owner)).data
    return {(c.scope_type, c.scope_id, c.provider_key_id, c.model) for c in ceilings}


@pytest.mark.asyncio
async def test_a_budget_is_created_with_its_entities_in_one_step(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-applied-create")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Platform", owner=owner)
    service = _service(async_db)

    created = await service.create_organization_budget(
        user=owner,
        request=_create(
            applied_to=[
                AppliedEntity(scope_type="workspace", scope_id=str(workspace.id)),
                AppliedEntity(
                    scope_type="organization", scope_id=str(organization.id), provider_key_id="openai", model="gpt-4o"
                ),
            ]
        ),
    )

    assert created.ceiling_count == 2
    assert await _applied(service, owner) == {
        ("workspace", str(workspace.id), None, None),
        ("organization", str(organization.id), "openai", "gpt-4o"),
    }


@pytest.mark.asyncio
async def test_an_update_replaces_the_entities_and_keeps_the_spend_of_those_it_keeps(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-applied-update")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Platform", owner=owner)
    service = _service(async_db)
    whole_org = AppliedEntity(scope_type="organization", scope_id=str(organization.id))
    platform = AppliedEntity(scope_type="workspace", scope_id=str(workspace.id))
    budget = await service.create_organization_budget(user=owner, request=_create(applied_to=[whole_org, platform]))
    kept = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.scope_type == "organization"))).scalar_one()
    kept.current_spend = Decimal("7.5")
    await async_db.flush()

    on_openai = AppliedEntity(scope_type="workspace", scope_id=str(workspace.id), provider_key_id="openai")
    updated = await service.update_organization_budget(
        user=owner, budget_id=budget.budget_id, request=OrganizationBudgetUpdate(applied_to=[whole_org, on_openai])
    )

    assert updated.ceiling_count == 2
    assert await _applied(service, owner) == {
        ("organization", str(organization.id), None, None),
        ("workspace", str(workspace.id), "openai", None),
    }
    still = await async_db.get(ScopedBudget, kept.id)
    assert still is not None
    assert still.current_spend == Decimal("7.5")


@pytest.mark.asyncio
async def test_an_update_without_entities_leaves_them_alone(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-applied-omitted")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    whole_org = AppliedEntity(scope_type="organization", scope_id=str(organization.id))
    budget = await service.create_organization_budget(user=owner, request=_create(applied_to=[whole_org]))

    await service.update_organization_budget(
        user=owner, budget_id=budget.budget_id, request=OrganizationBudgetUpdate(name="Renamed")
    )

    assert await _applied(service, owner) == {("organization", str(organization.id), None, None)}


@pytest.mark.asyncio
async def test_a_refused_entity_refuses_the_whole_create(async_db: AsyncSession) -> None:
    """No budget is left behind applying to part of what was asked."""
    organization = await _organization(async_db, slug="acme-applied-atomic")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    with pytest.raises(OrganizationScopeNotFoundError):
        await service.create_organization_budget(
            user=owner,
            request=_create(
                applied_to=[
                    AppliedEntity(scope_type="organization", scope_id=str(organization.id)),
                    AppliedEntity(scope_type="workspace", scope_id=str(uuid.uuid4())),
                ]
            ),
        )

    assert (await service.list_organization_budgets(user=owner)).count == 0
    assert await _applied(service, owner) == set()


@pytest.mark.asyncio
async def test_an_entity_another_budget_carries_refuses_the_whole_update(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-applied-taken")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    whole_org = AppliedEntity(scope_type="organization", scope_id=str(organization.id))
    await service.create_organization_budget(user=owner, request=_create(name="First", applied_to=[whole_org]))
    second = await service.create_organization_budget(user=owner, request=_create(name="Second"))

    with pytest.raises(OrganizationScopedBudgetAlreadyExistsError):
        await service.update_organization_budget(
            user=owner,
            budget_id=second.budget_id,
            request=OrganizationBudgetUpdate(name="Renamed", applied_to=[whole_org]),
        )

    names = {budget.name for budget in (await service.list_organization_budgets(user=owner)).data}
    assert names == {"First", "Second"}


def test_an_entity_named_twice_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    entity = {"scope_type": "workspace", "scope_id": "w1"}
    refused = client.post(_BUDGETS, json=_budget_body(applied_to=[entity, entity]), headers=master_key_header)
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


@pytest.mark.asyncio
async def test_a_non_canonical_scope_id_is_refused(async_db: AsyncSession) -> None:
    """An uppercase UUID parses, but it would store a ceiling that nothing matches and nothing can remove."""
    organization = await _organization(async_db, slug="acme-applied-canonical")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Platform", owner=owner)

    with pytest.raises(OrganizationScopeNotFoundError):
        await _service(async_db).create_organization_budget(
            user=owner,
            request=_create(applied_to=[AppliedEntity(scope_type="workspace", scope_id=str(workspace.id).upper())]),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["workspace", "org_member", "workspace_member", "api_token"])
async def test_an_update_naming_another_organizations_scope_changes_nothing(
    async_db: AsyncSession, scope_type: str
) -> None:
    """The refusal also takes back the removals made before it, with their spend."""
    organization = await _organization(async_db, slug=f"acme-applied-foreign-{scope_type}")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    other = await _organization(async_db, slug=f"other-applied-foreign-{scope_type}")
    other_owner = await _member(async_db, other, role="owner", full_name="Other owner")
    foreign_workspace = await _workspace(async_db, other, name="Elsewhere", owner=other_owner)
    key = APIKey(
        id=f"sk-foreign-{scope_type}", key_hash=f"hash-foreign-{scope_type}", workspace_id=foreign_workspace.id
    )
    async_db.add(key)
    await async_db.commit()
    other_owner_id = other_owner.id
    foreign_ids = {
        "workspace": str(foreign_workspace.id),
        "org_member": str(
            (
                await async_db.execute(
                    select(col(OrganizationMember.id)).where(col(OrganizationMember.user_id) == other_owner_id)
                )
            ).scalar_one()
        ),
        "workspace_member": str(
            (
                await async_db.execute(
                    select(col(WorkspaceMember.id)).where(col(WorkspaceMember.workspace_id) == foreign_workspace.id)
                )
            ).scalar_one()
        ),
        "api_token": key.id,
    }
    service = _service(async_db)
    whole_org = AppliedEntity(scope_type="organization", scope_id=str(organization.id))
    budget = await service.create_organization_budget(user=owner, request=_create(applied_to=[whole_org]))
    held = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.budget_id == budget.budget_id))).scalar_one()
    held.current_spend = Decimal("7.5")
    held_id = held.id
    await async_db.commit()

    with pytest.raises(OrganizationScopeNotFoundError):
        await service.update_organization_budget(
            user=owner,
            budget_id=budget.budget_id,
            request=OrganizationBudgetUpdate(
                applied_to=[AppliedEntity(scope_type=scope_type, scope_id=foreign_ids[scope_type])]  # type: ignore[arg-type]
            ),
        )

    async_db.expire_all()
    still = await async_db.get(ScopedBudget, held_id)
    assert still is not None
    assert still.current_spend == Decimal("7.5")


@pytest.mark.asyncio
async def test_an_update_changing_the_cycle_and_the_entities_puts_both_on_the_new_window(
    async_db: AsyncSession,
) -> None:
    organization = await _organization(async_db, slug="acme-applied-cycle")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, name="Platform", owner=owner)
    service = _service(async_db)
    whole_org = AppliedEntity(scope_type="organization", scope_id=str(organization.id))
    platform = AppliedEntity(scope_type="workspace", scope_id=str(workspace.id))
    budget = await service.create_organization_budget(user=owner, request=_create(applied_to=[whole_org]))

    await service.update_organization_budget(
        user=owner,
        budget_id=budget.budget_id,
        request=OrganizationBudgetUpdate(reset_cycle="daily", applied_to=[whole_org, platform]),
    )

    async_db.expire_all()
    ends = {
        ceiling.scope_type: ceiling.period_end
        for ceiling in (
            await async_db.execute(select(ScopedBudget).where(ScopedBudget.budget_id == budget.budget_id))
        ).scalars()
    }
    tomorrow = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    assert {key: end.astimezone(UTC) if end else None for key, end in ends.items()} == {
        "organization": tomorrow,
        "workspace": tomorrow,
    }


def test_the_routes_answer_a_refused_entity_with_its_status(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    unknown = {"scope_type": "workspace", "scope_id": str(uuid.uuid4())}
    refused = client.post(_BUDGETS, json=_budget_body(applied_to=[unknown]), headers=master_key_header)
    assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text


@pytest.mark.asyncio
async def test_a_delete_is_refused_while_another_organizations_ceiling_names_the_budget(
    async_db: AsyncSession,
) -> None:
    """Deleting would uncap another tenant's scope, which is not this admin's to decide."""
    organization = await _organization(async_db, slug="acme-delete-foreign")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    other = await _organization(async_db, slug="other-delete-foreign")
    other_owner = await _member(async_db, other, role="owner", full_name="Other owner")
    elsewhere = await _workspace(async_db, other, name="Elsewhere", owner=other_owner)
    service = _service(async_db)
    budget = await service.create_organization_budget(
        user=owner,
        request=_create(applied_to=[AppliedEntity(scope_type="organization", scope_id=str(organization.id))]),
    )
    budget_id = budget.budget_id
    async_db.add(ScopedBudget(scope_type="workspace", scope_id=str(elsewhere.id), budget_id=budget_id))
    await async_db.commit()

    with pytest.raises(OrganizationBudgetHeldElsewhereError):
        await service.delete_organization_budget(user=owner, budget_id=budget_id)

    async_db.expire_all()
    remaining = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.budget_id == budget_id))).scalars()
    assert len(list(remaining)) == 2
