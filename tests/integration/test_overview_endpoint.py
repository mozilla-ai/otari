"""The dashboard overview's summary endpoint.

Two halves, the way ``test_organization_budgets.py`` splits them.

The **HTTP surface** goes through the client, which can only act as the one
superuser operator a standalone deployment provisions. That covers the counts,
the scoping and the judgment.

The **rules that decide who sees which strip** are exercised at the service
layer, with identities built at whatever role a case needs, because the API
cannot act as a plain member.
"""

import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget, ScopedBudget
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.users import User as ApiUser
from gateway.repositories.overview.overview_repository import Allocation, OverviewRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.budgets import WorkspaceBudgetDefaultService
from gateway.services.overview.overview_service import OverviewService, judge
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.workspace_service import WorkspaceService

_ENDPOINT = f"{API_ROOT}/overview"


# =============================================================================
# The judgment, which is pure
# =============================================================================


def _allocation(spent: float, allocated: float, name: str = "Row") -> Allocation:
    return Allocation(name=name, budget_id=f"b-{name}", spent=spent, allocated=allocated)


def test_nothing_capped_is_neither_over_nor_near() -> None:
    health = judge([], total_count=3)

    assert (health.over_count, health.near_count, health.capped_count) == (0, 0, 0)
    assert health.total_count == 3
    assert health.worst is None


def test_a_row_at_its_limit_counts_as_over_rather_than_near() -> None:
    """The boundary is inclusive, so a budget exactly spent reads as over."""

    health = judge([_allocation(spent=100, allocated=100)], total_count=1)

    assert (health.over_count, health.near_count) == (1, 0)


def test_eighty_percent_is_where_a_row_becomes_near() -> None:
    below = judge([_allocation(spent=79, allocated=100)], total_count=1)
    at = judge([_allocation(spent=80, allocated=100)], total_count=1)

    assert below.near_count == 0
    assert at.near_count == 1


def test_a_zero_allowance_with_spend_against_it_is_over() -> None:
    """It admits nothing, so anything spent is past it. The share has no finite
    value, and reporting it as a full 1.0 keeps the row comparable."""

    health = judge([_allocation(spent=5, allocated=0)], total_count=1)

    assert health.over_count == 1
    assert health.worst is not None
    assert health.worst.allocated == 0


def test_the_worst_row_is_the_one_furthest_through_its_allowance() -> None:
    health = judge(
        [
            _allocation(spent=50, allocated=100, name="Half"),
            _allocation(spent=190, allocated=100, name="Worst"),
            _allocation(spent=95, allocated=100, name="Near"),
        ],
        total_count=3,
    )

    assert health.worst is not None
    assert health.worst.name == "Worst"


# =============================================================================
# The HTTP surface
# =============================================================================


def _caller_organization_id(client: TestClient, headers: dict[str, str]) -> uuid.UUID:
    """The organization the caller is acting in.

    The counts are scoped to it, so a fixture that builds rows in an
    organization of its own is testing the tenant guard rather than the counts.
    """

    context = client.get(f"{API_ROOT}/organizations/me", headers=headers)
    assert context.status_code == 200
    return uuid.UUID(context.json()["organization"]["id"])


def test_an_operator_sees_both_strips(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Deployment budgets and organization ceilings are different authorities,
    and the deployment's own operator holds both."""

    response = client.get(_ENDPOINT, headers=master_key_header)

    assert response.status_code == 200
    body = response.json()
    assert body["budgets"] is not None
    assert body["ceilings"] is not None
    assert body["active_members"] == 0


@pytest.mark.asyncio
async def test_only_active_keys_are_counted(
    client: TestClient,
    master_key_header: dict[str, str],
    async_db: AsyncSession,
) -> None:
    """The rail says "active keys", and the keys page filters the same way."""

    organization_id = _caller_organization_id(client, master_key_header)
    owner = await _member_of(async_db, organization_id, role="owner", full_name="Owner")
    workspace = await _workspace_in(async_db, organization_id, name="Engineering", owner=owner)
    async_db.add(APIKey(id="sk-live", key_hash="h1", workspace_id=workspace.id, is_active=True))
    async_db.add(APIKey(id="sk-revoked", key_hash="h2", workspace_id=workspace.id, is_active=False))
    await async_db.commit()

    response = client.get(_ENDPOINT, params={"workspace_id": str(workspace.id)}, headers=master_key_header)

    assert response.status_code == 200
    assert response.json()["active_keys"] == 1


def test_an_unknown_workspace_is_ignored_too(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(_ENDPOINT, params={"workspace_id": str(uuid.uuid4())}, headers=master_key_header)

    assert response.status_code == 200
    assert response.json()["active_members"] == 0


@pytest.mark.asyncio
async def test_a_budget_is_judged_against_its_users_share_of_the_cap(
    client: TestClient,
    master_key_header: dict[str, str],
    async_db: AsyncSession,
) -> None:
    """``max_budget`` is a per-user cap the attached users share, so the honest
    allowance is the cap times the number of them, as the budgets page shows."""

    budget = Budget(budget_id="b-shared", name="Shared", max_budget=Decimal(100))
    async_db.add(budget)
    async_db.add(ApiUser(user_id="u1", budget_id="b-shared", spend=Decimal(90), reserved=Decimal(0)))
    async_db.add(ApiUser(user_id="u2", budget_id="b-shared", spend=Decimal(80), reserved=Decimal(0)))
    await async_db.commit()

    body = client.get(_ENDPOINT, headers=master_key_header).json()

    # 170 spent of 200 allowed is near, not over: judged per budget, not per user.
    assert body["budgets"]["over_count"] == 0
    assert body["budgets"]["near_count"] == 1
    assert body["budgets"]["worst"]["allocated"] == 200
    assert body["budgets"]["worst"]["spent"] == 170


@pytest.mark.asyncio
async def test_an_uncapped_budget_counts_toward_the_total_and_not_the_judgment(
    client: TestClient,
    master_key_header: dict[str, str],
    async_db: AsyncSession,
) -> None:
    """The page tells "no budgets" from "none caps spend", so it needs both."""

    async_db.add(Budget(budget_id="b-open", name="Unlimited", max_budget=None))
    async_db.add(ApiUser(user_id="u3", budget_id="b-open", spend=Decimal(10), reserved=Decimal(0)))
    await async_db.commit()

    body = client.get(_ENDPOINT, headers=master_key_header).json()

    assert body["budgets"]["total_count"] == 1
    assert body["budgets"]["capped_count"] == 0
    assert body["budgets"]["worst"] is None


@pytest.mark.asyncio
async def test_reserved_spend_counts_against_a_ceiling(
    client: TestClient,
    master_key_header: dict[str, str],
    async_db: AsyncSession,
) -> None:
    """A hold is money already committed, so a ceiling that ignored it would
    report headroom a request has been told it cannot have."""

    organization_id = _caller_organization_id(client, master_key_header)
    budget = Budget(budget_id="b-ceiling", name="Cap", max_budget=Decimal(100), organization_id=organization_id)
    async_db.add(budget)
    async_db.add(
        ScopedBudget(
            id="sb-1",
            scope_type="workspace",
            scope_id=str(uuid.uuid4()),
            budget_id="b-ceiling",
            name="Engineering",
            current_spend=Decimal(60),
            reserved_spend=Decimal(45),
        )
    )
    await async_db.commit()

    response = client.get(_ENDPOINT, headers=master_key_header)

    assert response.status_code == 200
    # 105 of 100 once the hold is counted, which is over rather than near.
    ceilings = response.json()["ceilings"]
    assert ceilings["over_count"] == 1
    assert ceilings["worst"]["spent"] == 105


# =============================================================================
# Who sees which strip
#
# At the service layer, because the API can only act as the deployment's one
# operator identity, which is a superuser and an owner everywhere.
# =============================================================================


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )


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
    return user


async def _workspace(db: AsyncSession, organization: Organization, *, name: str, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


async def _member_of(db: AsyncSession, organization_id: uuid.UUID, *, role: str, full_name: str) -> User:
    """A member of an organization that already exists, the caller's own."""

    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization_id,
        is_superuser=False,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization_id,
        user_id=user.id,
        role=role,
    )
    return user


async def _workspace_in(db: AsyncSession, organization_id: uuid.UUID, *, name: str, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization_id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


def _service(db: AsyncSession) -> OverviewService:
    return OverviewService(
        OverviewRepository(db),
        OrganizationService(db, membership_listener=None),
        DeploymentUserService(db),
        WorkspaceService(db, membership_listener=WorkspaceBudgetDefaultService(db)),
    )


@pytest.mark.asyncio
async def test_a_member_gets_no_ceiling_strip_and_no_budget_strip(
    async_db: AsyncSession,
) -> None:
    """Withheld rather than emptied: the page draws a different strip for
    "nothing to judge" than for "not yours to see"."""

    organization = await _organization(async_db, slug="acme-member")
    member = await _member(async_db, organization, role="member", full_name="Member")

    summary = await _service(async_db).summary(identity=member, workspace_id=None)

    assert summary.ceilings is None
    assert summary.budgets is None


@pytest.mark.asyncio
async def test_an_admin_sees_the_ceilings_but_not_the_deployment_budgets(
    async_db: AsyncSession,
) -> None:
    """Two different authorities: ceilings are the organization's, and
    deployment budgets are the operator's."""

    organization = await _organization(async_db, slug="acme-admin")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")

    summary = await _service(async_db).summary(identity=admin, workspace_id=None)

    assert summary.ceilings is not None
    assert summary.budgets is None


@pytest.mark.asyncio
async def test_a_workspace_of_another_organization_is_ignored_rather_than_refused(
    async_db: AsyncSession,
) -> None:
    """The id comes from a switcher whose contents can go stale, so a stale one
    falls back to the unscoped answer rather than failing the page. What it must
    not do is answer for the other tenant, which is the leak."""

    mine = await _organization(async_db, slug="acme-scope-mine")
    theirs = await _organization(async_db, slug="acme-scope-theirs")
    member = await _member(async_db, mine, role="owner", full_name="Mine")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Theirs")
    their_workspace = await _workspace(async_db, theirs, name="Theirs", owner=their_owner)
    await async_db.flush()

    summary = await _service(async_db).summary(identity=member, workspace_id=their_workspace.id)

    # Their workspace has one active member; ours is not scoped to it.
    assert summary.active_members == 0


@pytest.mark.asyncio
async def test_a_member_cannot_count_a_workspace_they_are_not_in(
    async_db: AsyncSession,
) -> None:
    """Belonging to the organization is not enough, which is the rule the
    workspace routes this replaces already applied.

    An owner or admin sees every workspace in their organization; anybody else
    only the ones they are an active member of. Without this a member could read
    the key and member counts of a workspace they were never added to, by naming
    its id.
    """

    organization = await _organization(async_db, slug="acme-visible")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    outsider = await _member(async_db, organization, role="member", full_name="Outsider")
    workspace = await _workspace(async_db, organization, name="Private", owner=owner)
    async_db.add(APIKey(id="sk-private", key_hash="h9", workspace_id=workspace.id, is_active=True))
    await async_db.flush()

    seen_by_owner = await _service(async_db).summary(identity=owner, workspace_id=workspace.id)
    seen_by_outsider = await _service(async_db).summary(identity=outsider, workspace_id=workspace.id)

    assert seen_by_owner.active_members == 1
    # Not scoped to it at all, so the workspace's own roster never answers.
    assert seen_by_outsider.active_members == 0


@pytest.mark.asyncio
async def test_a_ceiling_keeps_the_scope_it_caps(
    async_db: AsyncSession,
) -> None:
    """A ceiling nobody named is named on screen after what it caps, so the
    scope has to survive the reduction rather than the row falling back to an
    id fingerprint."""

    organization = await _organization(async_db, slug="acme-scope-kept")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    scope_id = str(uuid.uuid4())
    async_db.add(
        Budget(budget_id="b-unnamed", name="Cap", max_budget=Decimal(10), organization_id=organization.id)
    )
    async_db.add(
        ScopedBudget(
            id="sb-unnamed",
            scope_type="workspace",
            scope_id=scope_id,
            budget_id="b-unnamed",
            name=None,
            current_spend=Decimal(9),
            reserved_spend=Decimal(0),
        )
    )
    await async_db.flush()

    summary = await _service(async_db).summary(identity=owner, workspace_id=None)

    assert summary.ceilings is not None
    assert summary.ceilings.worst is not None
    assert summary.ceilings.worst.name is None
    assert summary.ceilings.worst.scope_type == "workspace"
    assert summary.ceilings.worst.scope_id == scope_id


@pytest.mark.asyncio
async def test_a_members_key_count_stays_inside_their_organization(
    async_db: AsyncSession,
) -> None:
    """There is no organization column on a key, so this is the join that
    stands in for one. Without it a member would count the deployment."""

    mine = await _organization(async_db, slug="acme-mine")
    theirs = await _organization(async_db, slug="acme-theirs")
    member = await _member(async_db, mine, role="member", full_name="Member")
    my_owner = await _member(async_db, mine, role="owner", full_name="My Owner")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their Owner")
    my_workspace = await _workspace(async_db, mine, name="Mine", owner=my_owner)
    their_workspace = await _workspace(async_db, theirs, name="Theirs", owner=their_owner)
    async_db.add(APIKey(id="sk-mine", key_hash="h4", workspace_id=my_workspace.id, is_active=True))
    async_db.add(APIKey(id="sk-theirs-2", key_hash="h5", workspace_id=their_workspace.id, is_active=True))
    await async_db.flush()

    summary = await _service(async_db).summary(identity=member, workspace_id=None)

    assert summary.active_keys == 1
