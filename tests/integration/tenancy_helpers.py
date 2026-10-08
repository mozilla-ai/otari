"""Builders for the tenancy rows a service-layer test needs.

Each one writes through the repositories rather than a service, so a test can
start from whatever organization, workspace and membership shape it needs
without first satisfying authorization rules it is not exercising.
"""

from typing import TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.budgets import Budget
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetMembershipListener, BudgetService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.membership_listener import MembershipListener
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.workspace_listener import NullWorkspaceListener, WorkspaceListener


class MembershipWrites(TypedDict):
    """What a service that changes membership is built with: a Unit of Work and the listeners writing through it."""

    uow: UnitOfWork
    membership_listener: MembershipListener
    workspace_listener: WorkspaceListener


def membership_writes(db: AsyncSession) -> MembershipWrites:
    """Build a Unit of Work over ``db`` and the listeners on it, as a standalone deployment's request path does."""
    uow = UnitOfWork(db)
    return {
        "uow": uow,
        "membership_listener": BudgetMembershipListener(BudgetRepositories.on(uow)),
        "workspace_listener": NullWorkspaceListener(),
    }


def budget_service(db: AsyncSession) -> BudgetService:
    """Build the budget service over ``db``, as ``get_budget_service`` does."""
    uow = UnitOfWork(db)
    organizations = OrganizationService(db, membership_listener=None, workspace_listener=NullWorkspaceListener())
    return BudgetService(
        uow,
        BudgetRepositories.on(uow),
        organizations,
        ApiKeyService(ApiKeyRepository(uow)),
        WorkspaceAccess(db, organizations),
    )


async def create_organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def create_member(
    db: AsyncSession,
    organization: Organization,
    *,
    role: str,
    full_name: str,
) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return user


async def create_workspace(db: AsyncSession, organization: Organization, *, name: str, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


async def create_budget(
    db: AsyncSession,
    *,
    max_budget: float | None = None,
    budget_duration_sec: int | None = None,
    reset_alignment: str | None = None,
    name: str | None = None,
) -> str:
    """A budget for a default to hand out, returning its id.

    A default names a ``budgets`` row rather than carrying a limit of its own,
    which is what lets the Budgets page say a limit is a workspace's default.
    """
    budget = Budget(
        name=name,
        max_budget=max_budget,
        budget_duration_sec=budget_duration_sec,
        reset_alignment=reset_alignment,
    )
    db.add(budget)
    await db.flush()
    return budget.budget_id
