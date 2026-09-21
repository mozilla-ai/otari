"""Counts and allocations for the dashboard overview.

Every query here answers with a number or a handful of rows, never a
collection. That is the point: the overview used to read four whole tables in
the browser to render three integers and one worst-case row (otari#1425).
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget, ScopedBudget
from gateway.models.tenancy import Workspace, WorkspaceMember
from gateway.models.users import User

_ZERO = Decimal(0)


@dataclass(frozen=True)
class Allocation:
    """One row with a finite cap: what it is called, spent, and may spend.

    ``scope_type`` and ``scope_id`` are a ceiling's and are None for a
    deployment budget, which caps no scope. A ceiling nobody named is named on
    screen after what it caps ("A workspace"), so the scope has to survive the
    reduction or the row would fall back to an id fingerprint.
    """

    name: str | None
    budget_id: str
    spent: float
    allocated: float
    scope_type: str | None = None
    scope_id: str | None = None


class OverviewRepository:
    """Aggregate reads behind the overview's counts and health strips."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def count_active_keys(self, *, workspace_id: uuid.UUID | None, organization_id: uuid.UUID | None) -> int:
        """Active keys, narrowed the way the keys page narrows its own list.

        A workspace is the tighter scope and wins when both are given, which is
        what keeps this equal to the count the API keys page would show: the
        rail it feeds links there, so the two cannot disagree. With neither, the
        count is deployment-wide, which only an operator is allowed to ask for.

        There is no organization column on a key, so the organization scope is
        the workspaces that belong to it. That is the same route the key list
        takes to answer the question.
        """

        stmt = select(func.count()).select_from(APIKey).where(APIKey.is_active.is_(True))
        if workspace_id is not None:
            return int(await self.db.scalar(stmt.where(APIKey.workspace_id == workspace_id)) or 0)
        if organization_id is not None:
            stmt = stmt.join(Workspace, APIKey.workspace_id == col(Workspace.id)).where(
                col(Workspace.organization_id) == organization_id
            )
        return int(await self.db.scalar(stmt) or 0)

    async def count_active_workspace_members(self, workspace_id: uuid.UUID) -> int:
        """Active members of one workspace.

        One workspace, never the organization: the rail this feeds is headed
        "This workspace", and an organization member need not be a member of
        every workspace, so an organization-wide count would overcount it and
        stay put when the switcher moves.
        """

        stmt = (
            select(func.count())
            .select_from(WorkspaceMember)
            .where(col(WorkspaceMember.workspace_id) == workspace_id, col(WorkspaceMember.status) == "active")
        )
        return int(await self.db.scalar(stmt) or 0)

    async def budget_allocations(self) -> list[Allocation]:
        """The deployment's capped budgets, with the spend against each.

        ``max_budget`` on a deployment budget is a per-user cap that its users
        share, so the honest allocation is the cap times the number of active
        users holding it, which is what the budgets page shows. A budget with no
        cap or no users has no utilization to judge and is left out.

        One grouped pass rather than a count per budget: this runs on every
        overview load, and a query per row is what makes a summary cost more
        than the collection it replaced.
        """

        rollup = (
            select(
                User.budget_id.label("budget_id"),
                func.count().label("user_count"),
                # ``Decimal`` defaults rather than 0.0, for the reason
                # ``routes/budgets.py`` gives: in PostgreSQL
                # ``coalesce(numeric, double precision)`` resolves the sum as
                # double precision and rolls exact counters up through a float.
                func.coalesce(func.sum(User.spend), _ZERO).label("spend"),
                func.coalesce(func.sum(User.reserved), _ZERO).label("reserved"),
            )
            .where(User.budget_id.is_not(None), User.deleted_at.is_(None))
            .group_by(User.budget_id)
            .subquery()
        )
        stmt = (
            select(Budget, rollup.c.user_count, rollup.c.spend, rollup.c.reserved)
            .join(rollup, Budget.budget_id == rollup.c.budget_id)
            .where(Budget.max_budget.is_not(None))
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            Allocation(
                name=budget.name,
                budget_id=budget.budget_id,
                spent=float(spend) + float(reserved),
                allocated=float(budget.max_budget) * user_count,
            )
            for budget, user_count, spend, reserved in rows
            if user_count > 0
        ]

    async def ceiling_allocations(self, organization_id: uuid.UUID) -> list[Allocation]:
        """One organization's capped spend ceilings, with the spend against each.

        A ceiling carries its own counters, so the allocation is the budget's
        ``max_budget`` itself rather than a per-user cap multiplied out.
        """

        stmt = (
            select(ScopedBudget, Budget.max_budget)
            .join(Budget, ScopedBudget.budget_id == Budget.budget_id)
            .where(Budget.organization_id == organization_id, Budget.max_budget.is_not(None))
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            Allocation(
                name=ceiling.name,
                budget_id=ceiling.budget_id,
                spent=float(ceiling.current_spend) + float(ceiling.reserved_spend),
                allocated=float(max_budget),
                scope_type=ceiling.scope_type,
                scope_id=ceiling.scope_id,
            )
            for ceiling, max_budget in rows
        ]

    async def count_ceilings(self, organization_id: uuid.UUID) -> int:
        """Every ceiling the organization has, capped or not.

        Separate from :meth:`ceiling_allocations` because the page tells the two
        apart: no ceilings at all reads differently from ceilings that cap
        tokens or requests rather than dollars.
        """

        stmt = (
            select(func.count())
            .select_from(ScopedBudget)
            .join(Budget, ScopedBudget.budget_id == Budget.budget_id)
            .where(Budget.organization_id == organization_id)
        )
        return int(await self.db.scalar(stmt) or 0)

    async def count_budgets(self) -> int:
        """Every deployment budget, capped or not, for the same distinction."""

        return int(await self.db.scalar(select(func.count()).select_from(Budget)) or 0)

    async def workspace_organization(self, workspace_id: uuid.UUID) -> uuid.UUID | None:
        """Which organization owns this workspace, or None where none does.

        The caller names the workspace, so without this the counts above would
        report another organization's rail to anyone who guessed an id.
        """

        owner: uuid.UUID | None = await self.db.scalar(
            select(col(Workspace.organization_id)).where(col(Workspace.id) == workspace_id)
        )
        return owner
