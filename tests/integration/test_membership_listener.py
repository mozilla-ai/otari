"""The budgets domain reacts to workspace membership changes through a listener.

Each test drives a listener method directly, so the contract the organizations
domain calls is exercised without the services that call it.
"""

import uuid
from typing import NamedTuple

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.budgets import ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import Workspace, WorkspaceMember
from gateway.repositories.tenancy import WorkspaceMemberRepository
from gateway.services.tenancy.membership_listener import MembershipListener
from gateway.services.tenancy.workspace_budget_default_service import WorkspaceBudgetDefaultService

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
