"""Admitting an invited person: the membership, the workspace grants, their ceilings and the active organization."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import col

from gateway.exceptions.organizations_exceptions import (
    MembershipUpdateError,
    OrganizationMemberAddressUnprovenError,
)
from gateway.models.budgets import ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import (
    OrganizationMember,
    User,
    WorkspaceAssignmentRequest,
    WorkspaceMember,
)
from gateway.models.users import User as BillingUser
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.organizations import AdmittedMember, MembershipAdmission, MembershipListener
from gateway.services.tenancy import OrganizationService

from .conftest import _to_async_url
from .tenancy_helpers import (
    create_budget,
    create_member,
    create_organization,
    create_workspace,
    membership_writes,
)

pytestmark = pytest.mark.asyncio


class _RecordingListener:
    """The budget listener, with every organization-member announcement recorded."""

    def __init__(self, inner: MembershipListener) -> None:
        self._inner = inner
        self.changed: list[tuple[uuid.UUID, str, int]] = []
        self.role_changed: list[tuple[uuid.UUID, str]] = []

    async def member_joined(self, member: WorkspaceMember) -> None:
        await self._inner.member_joined(member)

    async def member_removed(self, member: WorkspaceMember) -> None:
        await self._inner.member_removed(member)

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        self.role_changed.append((member.id, member.role))
        await self._inner.member_role_changed(member)

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        self.changed.append((membership.id, membership.role, len(workspace_ids)))
        await self._inner.organization_member_changed(membership, workspace_ids)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        await self._inner.workspace_deleted(workspace_id, member_ids)

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        await self._inner.member_lost_workspaces(user_id, workspace_ids)


async def _invitee(db: AsyncSession, *, active_organization_id: uuid.UUID, email: str) -> User:
    """An identity that is a member of nothing, active somewhere else."""
    return await UserRepository(db).create_local_identity(
        full_name="Invitee", active_organization_id=active_organization_id, email=email
    )


async def _ceilings_for(db: AsyncSession, scope_id: uuid.UUID) -> list[str]:
    rows = await db.execute(select(ScopedBudget.budget_id).where(ScopedBudget.scope_id == str(scope_id)))
    return list(rows.scalars().all())


async def test_admitting_a_new_member_gives_them_the_ceiling_a_workspace_policy_owes(async_db: AsyncSession) -> None:
    writes = membership_writes(async_db)
    organization = await create_organization(async_db, slug="admit-ceiling")
    elsewhere = await create_organization(async_db, slug="admit-ceiling-other")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    budget_id = await create_budget(async_db, max_budget=25.0)
    async_db.add(WorkspaceBudgetDefault(workspace_id=workspace.id, budget_id=budget_id))
    await async_db.flush()
    invitee = await _invitee(async_db, active_organization_id=elsewhere.id, email="ada@example.com")
    service = OrganizationService(async_db, **writes)

    async with writes["uow"]:
        admitted = await service.admit(
            organization_id=organization.id,
            role="member",
            user_id=invitee.id,
            workspaces=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member")],
        )

    assert admitted.created is True
    membership = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(workspace.id, invitee.id)
    assert membership is not None
    assert await _ceilings_for(async_db, membership.id) == [budget_id]


async def test_admitting_a_suspended_member_revives_them_with_the_offered_role(async_db: AsyncSession) -> None:
    writes = membership_writes(async_db)
    organization = await create_organization(async_db, slug="admit-revive")
    await create_member(async_db, organization, role="owner", full_name="Owner")
    returning = await create_member(async_db, organization, role="viewer", full_name="Returning")
    members = OrganizationMemberRepository(async_db)
    suspended = await members.get_by_organization_and_user(organization.id, returning.id)
    assert suspended is not None
    await members.update_membership(suspended, {"status": "suspended"})
    await async_db.flush()
    service = OrganizationService(async_db, **writes)

    async with writes["uow"]:
        admitted = await service.admit(
            organization_id=organization.id, role="admin", user_id=returning.id, workspaces=[]
        )

    assert admitted.created is False
    revived = await members.get_by_organization_and_user(organization.id, returning.id)
    assert revived is not None
    assert (revived.role, revived.status) == ("admin", "active")


async def test_admitting_drops_a_workspace_that_no_longer_exists(async_db: AsyncSession) -> None:
    writes = membership_writes(async_db)
    organization = await create_organization(async_db, slug="admit-vanished")
    elsewhere = await create_organization(async_db, slug="admit-vanished-other")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    invitee = await _invitee(async_db, active_organization_id=elsewhere.id, email="grace@example.com")
    service = OrganizationService(async_db, **writes)

    async with writes["uow"]:
        admitted = await service.admit(
            organization_id=organization.id,
            role="member",
            user_id=invitee.id,
            workspaces=[
                WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member"),
                WorkspaceAssignmentRequest(workspace_id=uuid.uuid4(), role="member"),
            ],
        )

    assert admitted.created is True
    placements = (
        (
            await async_db.execute(
                select(col(WorkspaceMember.workspace_id)).where(col(WorkspaceMember.user_id) == invitee.id)
            )
        )
        .scalars()
        .all()
    )
    assert list(placements) == [workspace.id]


async def test_admitting_refuses_an_offer_that_would_leave_no_active_owner(async_db: AsyncSession) -> None:
    writes = membership_writes(async_db)
    organization = await create_organization(async_db, slug="admit-last-owner")
    only_owner = await create_member(async_db, organization, role="owner", full_name="Only Owner")
    service = OrganizationService(async_db, **writes)

    with pytest.raises(MembershipUpdateError, match="at least one active owner"):
        async with writes["uow"]:
            await service.admit(organization_id=organization.id, role="member", user_id=only_owner.id, workspaces=[])


async def test_admitting_makes_the_organization_active_and_mints_the_attribution_row(async_db: AsyncSession) -> None:
    writes = membership_writes(async_db)
    organization = await create_organization(async_db, slug="admit-active-org")
    elsewhere = await create_organization(async_db, slug="admit-active-org-other")
    await create_member(async_db, organization, role="owner", full_name="Owner")
    invitee = await _invitee(async_db, active_organization_id=elsewhere.id, email="ida@example.com")
    service = OrganizationService(async_db, **writes)

    async with writes["uow"]:
        await service.admit(organization_id=organization.id, role="member", user_id=invitee.id, workspaces=[])

    active = (
        await async_db.execute(select(col(User.active_organization_id)).where(col(User.id) == invitee.id))
    ).scalar_one()
    assert active == organization.id
    attribution = (
        await async_db.execute(select(BillingUser.user_id).where(BillingUser.user_id == str(invitee.id)))
    ).scalar_one_or_none()
    assert attribution is not None


async def test_admitting_an_active_member_announces_the_role_it_applied(async_db: AsyncSession) -> None:
    """A demotion through an invitation is announced, so a viewer's keys are disabled as the admin path does."""
    writes = membership_writes(async_db)
    listener = _RecordingListener(writes["membership_listener"])
    organization = await create_organization(async_db, slug="admit-announces")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    active = await create_member(async_db, organization, role="admin", full_name="Active Admin")
    await create_workspace(async_db, organization, name="Engineering", owner=owner)
    service = OrganizationService(async_db, membership_listener=listener, uow=writes["uow"])

    async with writes["uow"]:
        admitted = await service.admit(organization_id=organization.id, role="viewer", user_id=active.id, workspaces=[])

    assert admitted.created is False
    assert [(member_id, role) for member_id, role, _ in listener.changed] == [(admitted.member_id, "viewer")]
    assert listener.changed[0][2] >= 1


async def test_admitting_announces_a_workspace_role_the_invitation_changes(async_db: AsyncSession) -> None:
    """A workspace demotion through an invitation is announced, so the member's keys follow the role they hold."""
    writes = membership_writes(async_db)
    listener = _RecordingListener(writes["membership_listener"])
    organization = await create_organization(async_db, slug="admit-workspace-role")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    member = await create_member(async_db, organization, role="member", full_name="Member")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    placement = await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id, user_id=member.id, role="admin"
    )
    await async_db.flush()
    service = OrganizationService(async_db, membership_listener=listener, uow=writes["uow"])

    async with writes["uow"]:
        await service.admit(
            organization_id=organization.id,
            role="member",
            user_id=member.id,
            workspaces=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="viewer")],
        )

    assert listener.role_changed == [(placement.id, "viewer")]


async def test_admitting_says_nothing_about_a_workspace_role_that_did_not_change(async_db: AsyncSession) -> None:
    """A second acceptance of the same offer changes nothing, so it announces nothing."""
    writes = membership_writes(async_db)
    listener = _RecordingListener(writes["membership_listener"])
    organization = await create_organization(async_db, slug="admit-workspace-role-same")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    member = await create_member(async_db, organization, role="member", full_name="Member")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    await async_db.flush()
    service = OrganizationService(async_db, membership_listener=listener, uow=writes["uow"])

    async with writes["uow"]:
        await service.admit(
            organization_id=organization.id,
            role="member",
            user_id=member.id,
            workspaces=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member")],
        )

    assert listener.role_changed == []


async def test_admitting_compares_a_workspace_role_read_under_its_lock(
    async_db: AsyncSession, postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role edit committing before the lock is taken is seen, so the offered role is applied and announced."""
    writes = membership_writes(async_db)
    listener = _RecordingListener(writes["membership_listener"])
    organization = await create_organization(async_db, slug="admit-workspace-role-raced")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    member = await create_member(async_db, organization, role="member", full_name="Member")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    placement = await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id, user_id=member.id, role="member"
    )
    # Committed, so the edit below is a second transaction rather than this one.
    await async_db.commit()
    organization_id, member_id, placement_id, workspace_id = organization.id, member.id, placement.id, workspace.id
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    real_lock = WorkspaceRepository.lock
    edited = False

    async def lock_once_an_admin_has_changed_the_role(self: WorkspaceRepository, target: uuid.UUID) -> None:
        nonlocal edited
        if not edited:
            edited = True
            async with sessions() as admin:
                # scalar_one, so the edit is known to have reached the row this test set up.
                changed = await admin.execute(
                    update(WorkspaceMember)
                    .where(col(WorkspaceMember.id) == placement_id)
                    .values(role="admin")
                    .returning(col(WorkspaceMember.role))
                    .execution_options(synchronize_session=False)
                )
                assert changed.scalar_one() == "admin"
                await admin.commit()
        await real_lock(self, target)

    monkeypatch.setattr(WorkspaceRepository, "lock", lock_once_an_admin_has_changed_the_role)
    service = OrganizationService(async_db, membership_listener=listener, uow=writes["uow"])

    try:
        async with writes["uow"]:
            await service.admit(
                organization_id=organization_id,
                role="member",
                user_id=member_id,
                workspaces=[WorkspaceAssignmentRequest(workspace_id=workspace_id, role="member")],
            )
        async with sessions() as check:
            stored = await check.execute(
                select(col(WorkspaceMember.role)).where(col(WorkspaceMember.id) == placement_id)
            )
            role = stored.scalar_one()
    finally:
        await engine.dispose()

    assert edited
    assert role == "member"
    assert listener.role_changed == [(placement_id, "member")]


async def test_a_refused_offer_leaves_no_membership_behind(async_db: AsyncSession) -> None:
    writes = membership_writes(async_db)
    organization = await create_organization(async_db, slug="admit-rolled-back")
    elsewhere = await create_organization(async_db, slug="admit-rolled-back-other")
    owner = await create_member(async_db, organization, role="owner", full_name="Only Owner")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    invitee = await _invitee(async_db, active_organization_id=elsewhere.id, email="rolled@example.com")
    # Committed before the block under test, so the rollback cannot take the fixtures with it.
    await async_db.commit()
    organization_id, owner_id, invitee_id, workspace_id = organization.id, owner.id, invitee.id, workspace.id
    service = OrganizationService(async_db, **writes)

    admitted: AdmittedMember | None = None
    with pytest.raises(MembershipUpdateError):
        async with writes["uow"]:
            admitted = await service.admit(
                organization_id=organization_id,
                role="member",
                user_id=invitee_id,
                workspaces=[WorkspaceAssignmentRequest(workspace_id=workspace_id, role="member")],
            )
            # The refusal below is what the test is about: it must take the membership above with it.
            await service.admit(organization_id=organization_id, role="member", user_id=owner_id, workspaces=[])

    assert admitted is not None, "the refusal under test must follow a membership that was created"
    members = OrganizationMemberRepository(async_db)
    assert await members.get_by_organization_and_user(organization_id, invitee_id) is None
    assert await members.get_by_organization_and_user(organization_id, owner_id) is not None


# The service is the collaborator acceptance holds, checked by mypy rather than by a call.
_CONFORMS: MembershipAdmission = OrganizationService(None, membership_listener=None)  # type: ignore[arg-type]


async def test_admitting_an_identity_only_another_organization_vouched_for_is_refused(
    async_db: AsyncSession,
) -> None:
    """A vouched identity carries a password the organization that vouched could have chosen."""
    organization = await create_organization(async_db, slug="admit-vouched")
    elsewhere = await create_organization(async_db, slug="admit-vouched-other")
    await create_member(async_db, organization, role="owner", full_name="Owner")
    invitee = await _invitee(async_db, active_organization_id=elsewhere.id, email="vouched@example.com")
    invitee.email_vouched_at = datetime.now(UTC)
    async_db.add(invitee)
    await async_db.flush()
    writes = membership_writes(async_db)
    service = OrganizationService(async_db, **writes)

    with pytest.raises(OrganizationMemberAddressUnprovenError):
        async with writes["uow"]:
            await service.admit(organization_id=organization.id, role="member", user_id=invitee.id, workspaces=[])


async def test_admitting_sees_a_vouch_committed_after_the_identity_was_resolved(postgres_url: str) -> None:
    """The guard reads the identity under its own lock, so a vouch committed in between is not missed."""
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="admit-late-vouch")
            elsewhere = await create_organization(setup, slug="admit-late-vouch-other")
            await create_member(setup, organization, role="owner", full_name="Owner")
            invitee = await _invitee(setup, active_organization_id=elsewhere.id, email="late@example.com")
            await setup.commit()
            organization_id, invitee_id = organization.id, invitee.id

        async with sessions() as admitting, sessions() as vouching:
            resolved = await UserRepository(admitting).get(invitee_id)
            assert resolved is not None
            assert resolved.email_vouched_at is None

            await vouching.execute(
                update(User).where(col(User.id) == invitee_id).values(email_vouched_at=datetime.now(UTC))
            )
            await vouching.commit()

            writes = membership_writes(admitting)
            service = OrganizationService(admitting, **writes)
            with pytest.raises(OrganizationMemberAddressUnprovenError):
                async with writes["uow"]:
                    await service.admit(
                        organization_id=organization_id, role="member", user_id=invitee_id, workspaces=[]
                    )
    finally:
        await engine.dispose()


async def test_admitting_somebody_already_on_this_roster_is_not_refused_for_being_vouched(
    async_db: AsyncSession,
) -> None:
    """The rule is about a second organization taking the identity, so its own roster is left alone."""
    organization = await create_organization(async_db, slug="admit-vouched-member")
    await create_member(async_db, organization, role="owner", full_name="Owner")
    member = await create_member(async_db, organization, role="member", full_name="Suspended")
    member.email_vouched_at = datetime.now(UTC)
    async_db.add(member)
    members = OrganizationMemberRepository(async_db)
    membership = await members.get_by_organization_and_user(organization.id, member.id)
    assert membership is not None
    await members.update_membership(membership, {"status": "suspended"})
    await async_db.flush()
    writes = membership_writes(async_db)
    service = OrganizationService(async_db, **writes)

    async with writes["uow"]:
        admitted = await service.admit(organization_id=organization.id, role="member", user_id=member.id, workspaces=[])

    assert admitted.created is False
