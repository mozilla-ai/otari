"""The budgets domain reacts to workspace membership changes through a listener.

Each test drives a listener method directly, so the contract the organizations
domain calls is exercised without the services that call it.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.budgets import ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    Workspace,
    WorkspaceAssignmentRequest,
    WorkspaceCreate,
    WorkspaceMember,
)
from gateway.repositories.tenancy import UserRepository, WorkspaceMemberRepository
from gateway.services.budgets import WorkspaceBudgetDefaultService
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.services.tenancy.membership_listener import MembershipListener
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity

from .tenancy_helpers import create_budget, create_member, create_organization, create_workspace

pytestmark = pytest.mark.asyncio


class _Case(NamedTuple):
    workspace: Workspace
    default: WorkspaceBudgetDefault
    member: WorkspaceMember


async def _case(db: AsyncSession, *, slug: str) -> _Case:
    """A workspace carrying one default, and a membership nothing has materialized yet."""
    organization = await create_organization(db, slug=slug)
    owner = await create_member(db, organization, role="owner", full_name="Owner")
    joiner = await create_member(db, organization, role="member", full_name="Joiner")
    workspace = await create_workspace(db, organization, name="Engineering", owner=owner)

    default = WorkspaceBudgetDefault(
        workspace_id=workspace.id,
        budget_id=await create_budget(db, max_budget=25.0),
    )
    db.add(default)
    await db.flush()

    member = await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=joiner.id, role="member")
    return _Case(workspace, default, member)


async def _ceilings_for(db: AsyncSession, scope_id: uuid.UUID) -> list[ScopedBudget]:
    stmt = select(ScopedBudget).where(ScopedBudget.scope_id == str(scope_id))
    return list((await db.execute(stmt)).scalars().all())


async def test_member_joined_materializes_the_workspace_defaults(async_db: AsyncSession) -> None:
    case = await _case(async_db, slug="acme-joined")
    listener = WorkspaceBudgetDefaultService(async_db)

    await listener.member_joined(case.member)
    await async_db.flush()

    ceilings = await _ceilings_for(async_db, case.member.id)
    assert [ceiling.budget_id for ceiling in ceilings] == [case.default.budget_id]


async def test_member_removed_deletes_the_member_ceilings(async_db: AsyncSession) -> None:
    case = await _case(async_db, slug="acme-removed")
    listener = WorkspaceBudgetDefaultService(async_db)
    await listener.member_joined(case.member)
    await async_db.flush()

    await listener.member_removed(case.member)
    await async_db.flush()

    assert await _ceilings_for(async_db, case.member.id) == []


async def test_workspace_deleted_deletes_workspace_and_member_ceilings(async_db: AsyncSession) -> None:
    case = await _case(async_db, slug="acme-deleted")
    listener = WorkspaceBudgetDefaultService(async_db)
    await listener.member_joined(case.member)
    async_db.add(
        ScopedBudget(
            scope_type="workspace",
            scope_id=str(case.workspace.id),
            budget_id=case.default.budget_id,
        )
    )
    await async_db.flush()

    await listener.workspace_deleted(case.workspace.id, [case.member.id])
    await async_db.flush()

    assert (await async_db.execute(select(ScopedBudget))).scalars().all() == []


def _as_listener(service: WorkspaceBudgetDefaultService) -> MembershipListener:
    """Checked by mypy only: the service satisfies the listener contract structurally."""
    return service


@dataclass
class RecordingListener:
    """Records what it was told, so a test can assert the announcement rather than its effect."""

    joined: list[uuid.UUID] = field(default_factory=list)
    removed: list[uuid.UUID] = field(default_factory=list)
    deleted: list[tuple[uuid.UUID, list[uuid.UUID]]] = field(default_factory=list)

    async def member_joined(self, member: WorkspaceMember) -> None:
        self.joined.append(member.id)

    async def member_removed(self, member: WorkspaceMember) -> None:
        self.removed.append(member.id)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        self.deleted.append((workspace_id, sorted(member_ids)))


async def _membership_id(db: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
    member = await WorkspaceMemberRepository(db).get_by_workspace_and_user(workspace_id, user_id)
    assert member is not None
    return member.id


async def test_add_member_announces_the_new_member(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme-add")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    joiner = await create_member(async_db, organization, role="member", full_name="Joiner")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    listener = RecordingListener()

    added = await WorkspaceService(async_db, membership_listener=listener).add_member(
        user=owner, workspace_id=workspace.id, user_id=joiner.id
    )

    assert listener.joined == [added.id]


async def test_create_workspace_announces_the_creator(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme-create")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    listener = RecordingListener()

    created = await WorkspaceService(async_db, membership_listener=listener).create_workspace(
        user=owner, workspace_create=WorkspaceCreate(name="Engineering")
    )

    assert listener.joined == [await _membership_id(async_db, created.id, owner.id)]


async def test_remove_member_announces_the_removal_before_the_row_goes(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme-remove")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    leaver = await create_member(async_db, organization, role="member", full_name="Leaver")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    listener = RecordingListener()
    service = WorkspaceService(async_db, membership_listener=listener)
    added = await service.add_member(user=owner, workspace_id=workspace.id, user_id=leaver.id)

    await service.remove_member(user=owner, workspace_id=workspace.id, user_id=leaver.id)

    assert listener.removed == [added.id]


async def test_delete_workspace_announces_the_workspace_and_its_members(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme-delete")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    await create_workspace(async_db, organization, name="Keep", owner=owner)
    doomed = await create_workspace(async_db, organization, name="Doomed", owner=owner)
    doomed_membership = await _membership_id(async_db, doomed.id, owner.id)
    listener = RecordingListener()

    await WorkspaceService(async_db, membership_listener=listener).delete_workspace(user=owner, workspace_id=doomed.id)

    assert listener.deleted == [(doomed.id, [doomed_membership])]


async def test_workspace_assignment_announces_new_and_revived_members_only(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme-assign")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    fresh = await create_workspace(async_db, organization, name="Fresh", owner=owner)
    revived = await create_workspace(async_db, organization, name="Revived", owner=owner)
    already = await create_workspace(async_db, organization, name="Already", owner=owner)

    target = await UserRepository(async_db).create_local_identity(
        full_name="Target",
        email="target@example.test",
        active_organization_id=organization.id,
    )
    members = WorkspaceMemberRepository(async_db)
    suspended = await members.create(workspace_id=revived.id, user_id=target.id, role="member", status="suspended")
    active = await members.create(workspace_id=already.id, user_id=target.id, role="member", status="active")
    await async_db.commit()
    listener = RecordingListener()

    await OrganizationService(async_db, membership_listener=listener).create_active_organization_member_for_user(
        user=owner,
        request=ActiveOrganizationMemberCreateRequest(
            email="target@example.test",
            role="member",
            workspace_assignments=[
                WorkspaceAssignmentRequest(workspace_id=workspace_id, role="member")
                for workspace_id in (fresh.id, revived.id, already.id)
            ],
        ),
    )

    assert active.id not in listener.joined, "an already-active membership is not a join"
    assert sorted(listener.joined) == sorted([await _membership_id(async_db, fresh.id, target.id), suspended.id])


async def test_an_organization_service_without_a_listener_refuses_membership_changes(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme-mute")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)

    with pytest.raises(RuntimeError):
        await OrganizationService(async_db, membership_listener=None).create_active_organization_member_for_user(
            user=owner,
            request=ActiveOrganizationMemberCreateRequest(
                email="new-hire@example.test",
                role="member",
                workspace_assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member")],
            ),
        )


async def test_bootstrap_provisioning_announces_the_operator_membership(async_db: AsyncSession) -> None:
    listener = RecordingListener()

    operator = await ensure_bootstrap_identity(async_db, membership_listener=listener)

    memberships = (
        (await async_db.execute(select(WorkspaceMember).where(col(WorkspaceMember.user_id) == operator.id)))
        .scalars()
        .all()
    )
    assert listener.joined == [membership.id for membership in memberships]
