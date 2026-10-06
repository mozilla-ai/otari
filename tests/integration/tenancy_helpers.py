"""Builders for the tenancy rows a service-layer test needs.

Each one writes through the repositories rather than a service, so a test can
start from whatever organization, workspace and membership shape it needs
without first satisfying authorization rules it is not exercising.
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.budgets import Budget
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
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
    reset_cycle: str | None = None,
    reset_every_n: int | None = None,
    reset_anchor_at: datetime | None = None,
    reset_weekdays: int | None = None,
    reset_month_day: int | None = None,
    reset_month: int | None = None,
    name: str | None = None,
) -> str:
    """A budget for a default to hand out, returning its id.

    A default names a ``budgets`` row rather than carrying a limit of its own,
    which is what lets the Budgets page say a limit is a workspace's default.
    """
    budget = Budget(
        name=name,
        max_budget=max_budget,
        reset_cycle=reset_cycle,
        reset_every_n=reset_every_n,
        reset_anchor_at=reset_anchor_at,
        reset_weekdays=reset_weekdays,
        reset_month_day=reset_month_day,
        reset_month=reset_month,
    )
    db.add(budget)
    await db.flush()
    return budget.budget_id
