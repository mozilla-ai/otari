"""The roster row carries what the members page renders.

The page used to fetch seven collections and stitch them together in the browser:
the roster, every gateway identity, every workspace, a roster read per workspace,
every budget and every spend ceiling (otari#1381). The joins are the server's
work, and doing them there is also what lets the page ask for one row at a time.
"""

import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.budgets import Budget, ScopedBudget
from gateway.models.tenancy import (
    ActiveOrganizationMemberPublic,
    ActiveOrganizationMembersPublic,
    Organization,
    User,
    Workspace,
    WorkspaceMember,
)
from gateway.models.users import User as ApiUser
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.organization_service import OrganizationService


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )


async def _member(db: AsyncSession, organization: Organization, *, full_name: str, role: str = "member") -> User:
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


def _service(db: AsyncSession) -> OrganizationService:
    return OrganizationService(db, membership_listener=None)


async def _roster(db: AsyncSession, caller: User) -> ActiveOrganizationMembersPublic:
    return await _service(db).list_active_organization_members_for_user(user=caller)


def _row_for(page: ActiveOrganizationMembersPublic, user: User) -> ActiveOrganizationMemberPublic:
    return next(row for row in page.data if row.user_id == user.id)


@pytest.mark.asyncio
async def test_a_row_carries_the_workspaces_its_member_is_in(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-join-workspaces")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    await _workspace(async_db, organization, name="Engineering", owner=owner)
    await _workspace(async_db, organization, name="Research", owner=owner)

    row = _row_for(await _roster(async_db, owner), owner)

    assert [placement.workspace_name for placement in row.workspaces] == [
        "Engineering",
        "Research",
    ]
    assert {placement.role for placement in row.workspaces} == {"owner"}


@pytest.mark.asyncio
async def test_a_member_of_no_workspace_carries_an_empty_list(async_db: AsyncSession) -> None:
    """Not null: the page renders "no workspaces", which is a fact rather than
    an unanswered read."""

    organization = await _organization(async_db, slug="acme-join-none")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    outsider = await _member(async_db, organization, full_name="Outsider")

    row = _row_for(await _roster(async_db, owner), outsider)

    assert row.workspaces == []


@pytest.mark.asyncio
async def test_a_placement_carries_the_ceiling_on_that_membership(async_db: AsyncSession) -> None:
    """A ceiling is keyed on the membership, not the person, which is why the
    roster had to be resolved before the ceilings could be matched to it."""

    organization = await _organization(async_db, slug="acme-join-ceiling")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    membership = (await _roster(async_db, owner)).data[0].workspaces[0].workspace_member_id
    async_db.add(Budget(budget_id="b-ceiling", name="Cap", max_budget=Decimal(250)))
    async_db.add(
        ScopedBudget(
            id="sb-member",
            scope_type="workspace_member",
            scope_id=str(membership),
            budget_id="b-ceiling",
        )
    )
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    placement = next(p for p in row.workspaces if p.workspace_id == workspace.id)
    assert placement.ceiling is not None
    assert placement.ceiling.budget_id == "b-ceiling"
    assert placement.ceiling.max_budget == 250


@pytest.mark.asyncio
async def test_a_member_in_two_workspaces_holds_a_ceiling_in_each(async_db: AsyncSession) -> None:
    """Two memberships, so two ceilings. Keying them on the person would collapse
    the pair and report one workspace's cap on both."""

    organization = await _organization(async_db, slug="acme-join-two")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    await _workspace(async_db, organization, name="Engineering", owner=owner)
    await _workspace(async_db, organization, name="Research", owner=owner)
    placements = _row_for(await _roster(async_db, owner), owner).workspaces
    async_db.add(Budget(budget_id="b-eng", name="Engineering cap", max_budget=Decimal(100)))
    async_db.add(Budget(budget_id="b-res", name="Research cap", max_budget=Decimal(50)))
    for placement, budget_id in zip(placements, ("b-eng", "b-res"), strict=True):
        async_db.add(
            ScopedBudget(
                id=f"sb-{budget_id}",
                scope_type="workspace_member",
                scope_id=str(placement.workspace_member_id),
                budget_id=budget_id,
            )
        )
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert [p.ceiling.max_budget for p in row.workspaces if p.ceiling] == [100, 50]


@pytest.mark.asyncio
async def test_a_suspended_workspace_membership_is_not_a_placement(async_db: AsyncSession) -> None:
    """Somebody suspended in a workspace is no longer in it, and listing them
    there would put them somewhere they cannot act. ``get_workspaces_for_user``
    answers the same question the same way."""

    organization = await _organization(async_db, slug="acme-join-suspended")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    membership = (
        await async_db.execute(
            select(WorkspaceMember).where(col(WorkspaceMember.workspace_id) == workspace.id)
        )
    ).scalars().one()
    membership.status = "suspended"
    async_db.add(membership)
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert row.workspaces == []


@pytest.mark.asyncio
async def test_a_ceiling_narrowed_to_one_provider_is_not_the_memberships(async_db: AsyncSession) -> None:
    """A ceiling carrying a provider key caps that credential, not the
    membership. The roster reports what the member may spend at all, so the
    aggregate row is the one it wants, and reading either would make the figure
    depend on which row the database returned first."""

    organization = await _organization(async_db, slug="acme-join-narrowed")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    await _workspace(async_db, organization, name="Engineering", owner=owner)
    membership = (await _roster(async_db, owner)).data[0].workspaces[0].workspace_member_id
    async_db.add(Budget(budget_id="b-narrow", name="One provider", max_budget=Decimal(5)))
    async_db.add(
        ScopedBudget(
            id="sb-narrowed",
            scope_type="workspace_member",
            scope_id=str(membership),
            budget_id="b-narrow",
            provider_key_id="pk-1",
        )
    )
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert row.workspaces[0].ceiling is None


@pytest.mark.asyncio
async def test_the_aggregate_ceiling_wins_over_a_narrowed_one(async_db: AsyncSession) -> None:
    """Both rows can exist at once, and only one of them is the member's cap."""

    organization = await _organization(async_db, slug="acme-join-both")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    await _workspace(async_db, organization, name="Engineering", owner=owner)
    membership = (await _roster(async_db, owner)).data[0].workspaces[0].workspace_member_id
    async_db.add(Budget(budget_id="b-all", name="Everything", max_budget=Decimal(100)))
    async_db.add(Budget(budget_id="b-one", name="One provider", max_budget=Decimal(5)))
    async_db.add(
        ScopedBudget(
            id="sb-all",
            scope_type="workspace_member",
            scope_id=str(membership),
            budget_id="b-all",
        )
    )
    async_db.add(
        ScopedBudget(
            id="sb-one",
            scope_type="workspace_member",
            scope_id=str(membership),
            budget_id="b-one",
            provider_key_id="pk-1",
        )
    )
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert row.workspaces[0].ceiling is not None
    assert row.workspaces[0].ceiling.budget_id == "b-all"


@pytest.mark.asyncio
async def test_a_placement_stays_inside_the_callers_organization(async_db: AsyncSession) -> None:
    """A person can belong to two organizations. The roster of one must not name
    the workspaces of the other."""

    mine = await _organization(async_db, slug="acme-join-mine")
    theirs = await _organization(async_db, slug="acme-join-theirs")
    owner = await _member(async_db, mine, full_name="Owner", role="owner")
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=theirs.id, user_id=owner.id, role="member"
    )
    await _workspace(async_db, mine, name="Mine", owner=owner)
    elsewhere = await _workspace(async_db, theirs, name="Theirs", owner=owner)

    row = _row_for(await _roster(async_db, owner), owner)

    assert [p.workspace_name for p in row.workspaces] == ["Mine"]
    assert elsewhere.id not in {p.workspace_id for p in row.workspaces}


@pytest.mark.asyncio
async def test_the_spend_figures_are_withheld_from_a_non_operator(async_db: AsyncSession) -> None:
    """Deployment-wide facts. `/api/v1/users` refuses them, so the roster does
    too, and withholds rather than zeroes: a zero would read as a member who has
    spent nothing."""

    organization = await _organization(async_db, slug="acme-join-spend")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    async_db.add(ApiUser(user_id=str(owner.id), spend=Decimal(7), reserved=Decimal(1)))
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert row.attribution is None


@pytest.mark.asyncio
async def test_the_roster_asks_for_the_placements_of_its_page_only(async_db: AsyncSession) -> None:
    """The window still bounds the work: a page of one resolves one member's
    workspaces, not the organization's."""

    organization = await _organization(async_db, slug="acme-join-window")
    owner = await _member(async_db, organization, full_name="Aaa Owner", role="owner")
    for index in range(5):
        await _member(async_db, organization, full_name=f"Bbb Member {index}")
    await _workspace(async_db, organization, name="Engineering", owner=owner)

    page = await _service(async_db).list_active_organization_members_for_user(user=owner, limit=1)

    assert page.count == 6
    assert len(page.data) == 1
    assert [p.workspace_name for p in page.data[0].workspaces] == ["Engineering"]


@pytest.mark.asyncio
async def test_a_ceiling_on_another_scope_is_not_read_as_a_members(async_db: AsyncSession) -> None:
    """`scoped_budgets` holds ceilings for several scope kinds keyed on one
    column, so matching without the scope type would hand a workspace's own cap
    to whichever membership shared its id."""

    organization = await _organization(async_db, slug="acme-join-scope")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    async_db.add(Budget(budget_id="b-workspace", name="Workspace cap", max_budget=Decimal(9)))
    async_db.add(
        ScopedBudget(
            id="sb-workspace",
            scope_type="workspace",
            scope_id=str(workspace.id),
            budget_id="b-workspace",
        )
    )
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert row.workspaces[0].ceiling is None


@pytest.mark.asyncio
async def test_an_unknown_membership_id_matches_no_ceiling(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-join-orphan")
    owner = await _member(async_db, organization, full_name="Owner", role="owner")
    await _workspace(async_db, organization, name="Engineering", owner=owner)
    async_db.add(Budget(budget_id="b-orphan", name="Orphan", max_budget=Decimal(5)))
    async_db.add(
        ScopedBudget(
            id="sb-orphan",
            scope_type="workspace_member",
            scope_id=str(uuid.uuid4()),
            budget_id="b-orphan",
        )
    )
    await async_db.flush()

    row = _row_for(await _roster(async_db, owner), owner)

    assert row.workspaces[0].ceiling is None


def test_an_operator_is_given_the_spend_figures(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The other half of the gate. The deployment's own operator may read
    `/api/v1/users`, so the roster carries what that read was fetched for."""

    listed = client.get(f"{API_ROOT}/organizations/me/members", headers=master_key_header)

    assert listed.status_code == 200
    rows = listed.json()["data"]
    assert rows, "the deployment provisions one operator identity"
    # Present rather than populated: the figures are whatever the seeded
    # operator has spent, and the point is that the field is not withheld.
    assert all("attribution" in row for row in rows)
    assert any(row["attribution"] is not None for row in rows)
