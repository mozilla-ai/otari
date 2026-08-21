"""Workspace-level templates for per-member ``scoped_budgets`` ceilings.

A default (``WorkspaceBudgetDefault``) is a workspace-level template for a
per-member spend limit. When a member joins the workspace (or when a default
is created on a workspace that already has members) it is **eagerly
materialized** into a per-member :class:`ScopedBudget` row, so the ceiling is
visible immediately rather than on first spend.

The wire DTOs keep the ``WorkspaceMemberBudgetPolicy*`` names from
``otari-ai``'s ``budget_policy_service`` (the hosted equivalent this is ported
from), so the generated client stays recognizable across both trees; the
stored fields follow this repo's own ``ScopedBudget`` vocabulary
(``max_budget``, ``budget_duration_sec``) rather than otari-ai's
(``budget_limit``, ``spend_period``), since OSS budgets are USD/seconds-only
with no token or request dimension.

Materialization methods (``materialize_for_member``, ``materialize_for_default``)
are flush-only: they do not commit, so they fold into the enclosing
transaction at each call site (workspace creation, adding a member,
organization-member creation with workspace assignments). CRUD methods commit
on success.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.entities import Budget, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import User, Workspace, WorkspaceMember
from gateway.repositories.tenancy import WorkspaceMemberRepository, WorkspaceRepository
from gateway.services.budget_periods import period_window
from gateway.services.tenancy import authorization
from gateway.services.tenancy.errors import (
    WorkspaceBudgetDefaultAlreadyExistsError,
    WorkspaceBudgetDefaultBudgetNotFoundError,
    WorkspaceBudgetDefaultNotFoundError,
)
from gateway.services.tenancy.organization_service import OrganizationService

# The scope name is spelled out rather than imported from
# `scoped_budget_service`: that module imports `workspace_scope`, which imports
# `tenancy.provisioning_service`, which imports `tenancy/__init__`, which
# imports `workspace_service`, which imports this module. See
# `WorkspaceService`'s own docstring on `_delete_scoped_budgets_for` for the
# same avoidance. `tests/unit/test_service_module_imports.py` pins the graph
# staying acyclic.
#
# The *period derivation* used to be duplicated for the same reason, and that
# copy only understood durations, so a calendar-aligned budget materialized a
# ceiling with no window and never reset. It lives in
# `gateway.services.budget_periods` now, a leaf both sides import.
_SCOPE_WORKSPACE_MEMBER = "workspace_member"

# Page size for fanning a new default out across a workspace's active members,
# and the ceiling a list read pages at.
_MATERIALIZE_PAGE_SIZE = 500
_MAX_LIST_LIMIT = 1000



class WorkspaceMemberBudgetPolicyCreate(BaseModel):
    """Request body for creating a default."""

    budget_id: str = Field(
        min_length=1,
        max_length=255,
        description="The budget this workspace hands to every member",
    )
    provider_key_id: str | None = Field(
        default=None,
        max_length=255,
        description="Narrow the default to one provider instance; null applies to every provider",
    )


class WorkspaceMemberBudgetPolicyUpdate(BaseModel):
    """Request body for pointing a default at a different budget.

    Members already materialized from this default keep the budget they were
    given: their ceiling names it directly, and this only changes what a member
    joining afterwards is handed. Editing the *budget* is the retroactive act,
    and it moves everyone naming it, in this workspace and outside it.
    """

    budget_id: str = Field(min_length=1, max_length=255)


class WorkspaceMemberBudgetPolicyPublic(BaseModel):
    """One default and its template values."""

    id: str
    workspace_id: uuid.UUID
    budget_id: str
    provider_key_id: str | None
    # Read off the budget, not stored here. Carried on the wire so the dashboard
    # can render a default without fetching every budget to resolve one id, and
    # so this shape stays what it was before the limit moved onto the budget.
    name: str | None
    max_budget: float | None
    budget_duration_sec: int | None
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, default: WorkspaceBudgetDefault, budget: Budget) -> WorkspaceMemberBudgetPolicyPublic:
        return cls(
            id=default.id,
            workspace_id=default.workspace_id,
            budget_id=default.budget_id,
            provider_key_id=default.provider_key_id,
            name=budget.name,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            created_at=default.created_at.isoformat(),
            updated_at=default.updated_at.isoformat(),
        )


class WorkspaceMemberBudgetPoliciesPublic(BaseModel):
    data: list[WorkspaceMemberBudgetPolicyPublic]
    count: int


class WorkspaceBudgetDefaultService:
    """Materializer + CRUD for workspace per-member budget defaults."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db)
        self.workspaces = WorkspaceRepository(db)
        self.workspace_members = WorkspaceMemberRepository(db)

    # ------------------------------------------------------------------
    # Materialization (flush-only, no auth: called from within an
    # already-authorized membership-creation transaction)
    # ------------------------------------------------------------------

    async def materialize_for_member(self, member: WorkspaceMember) -> list[ScopedBudget]:
        """Create a per-member ``ScopedBudget`` for every default on the member's workspace.

        Skips any ``provider_key_id`` the member already has a ceiling for; a
        member-specific override always wins over the template. Never raises
        ``IntegrityError``: a collision on one default (see
        :meth:`_insert_member_budgets`) is swallowed there, precisely so this
        method's callers (``WorkspaceService.add_member``,
        ``OrganizationService._apply_workspace_assignments``) can keep their
        own ``except IntegrityError`` narrow to the membership row they
        actually insert, without a materialization-time collision escaping
        and being misreported as a duplicate membership.
        """
        defaults = (
            (
                await self.db.execute(
                    select(WorkspaceBudgetDefault).where(
                        col(WorkspaceBudgetDefault.workspace_id) == member.workspace_id
                    )
                )
            )
            .scalars()
            .all()
        )
        created: list[ScopedBudget] = []
        for default in defaults:
            if await self._existing_member_budget(member.id, default.provider_key_id) is not None:
                continue
            created.extend(await self._insert_member_budgets([member.id], default, await self._budget_for(default)))
        return created

    async def materialize_for_default(
        self, default: WorkspaceBudgetDefault, budget: Budget | None = None
    ) -> list[ScopedBudget]:
        """Create a per-member ``ScopedBudget`` for every active member of the default's workspace.

        Members who already have a ceiling for this ``provider_key_id`` are
        left untouched, same precedence rule as :meth:`materialize_for_member`.
        Pages through members so a large workspace is fully materialized.

        One existence query and one flush per page, not per member: for a
        page of up to ``_MATERIALIZE_PAGE_SIZE`` members that would otherwise
        be two round trips each (an existence check, then an insert), which on
        a workspace with hundreds of members is the difference between one
        query pair and hundreds of them.
        """
        # Resolved once for the whole fan-out rather than per page: every member
        # of this workspace is handed the same budget, and it cannot change under
        # us inside the transaction.
        template = budget if budget is not None else await self._budget_for(default)
        created: list[ScopedBudget] = []
        skip = 0
        while True:
            members, total = await self.workspace_members.get_by_workspace(
                default.workspace_id, skip=skip, limit=_MATERIALIZE_PAGE_SIZE
            )
            if not members:
                break
            active_ids = [member.id for member in members if member.status == "active"]
            if active_ids:
                created.extend(await self._materialize_batch(active_ids, default, template))
            skip += len(members)
            if skip >= total:
                break
        return created

    async def _materialize_batch(
        self, member_ids: list[uuid.UUID], default: WorkspaceBudgetDefault, budget: Budget
    ) -> list[ScopedBudget]:
        """Materialize ``default`` onto every id in ``member_ids`` that does not already have one.

        One query for the whole batch's existing rows: the insert itself is
        :meth:`_insert_member_budgets`, which is what actually tolerates a
        collision on any one id.
        """
        existing_stmt = select(ScopedBudget.scope_id).where(
            ScopedBudget.scope_type == _SCOPE_WORKSPACE_MEMBER,
            col(ScopedBudget.scope_id).in_([str(member_id) for member_id in member_ids]),
        )
        existing_stmt = existing_stmt.where(
            ScopedBudget.provider_key_id == default.provider_key_id
            if default.provider_key_id is not None
            else ScopedBudget.provider_key_id.is_(None)
        )
        already_covered = {row for row in (await self.db.execute(existing_stmt)).scalars().all()}
        candidates = [member_id for member_id in member_ids if str(member_id) not in already_covered]
        if not candidates:
            return []
        return await self._insert_member_budgets(candidates, default, budget)

    async def _insert_member_budgets(
        self, member_ids: list[uuid.UUID], default: WorkspaceBudgetDefault, budget: Budget
    ) -> list[ScopedBudget]:
        """Insert a ``ScopedBudget`` for each id in ``member_ids``, tolerating a uniqueness collision on any one.

        The existence check the two callers run first is a separate round
        trip with nothing locking the gap after it, and the one surface that
        can land there (a direct ``POST /v1/scoped-budgets`` for one of these
        members, racing this insert) takes no lock on the workspace:
        `routes/scoped_budgets.py` is a master-key admin surface with no
        notion of one. One flush for the whole batch in the common case;
        falls back to one row at a time, each in its own savepoint, on
        conflict, so a single colliding id costs one skipped row rather than
        the whole batch.

        Each attempt's ``add()`` happens *inside* its ``begin_nested()``
        block, not before it: ``AsyncSession.begin_nested()`` unconditionally
        flushes whatever is already pending before it opens the ``SAVEPOINT``,
        so an add staged beforehand is flushed *outside* any savepoint, and a
        conflict there fails the outer transaction rather than just rolling
        back to the point the savepoint was meant to protect. The same reason
        rules out an explicit ``expunge`` in the ``except`` branch: a
        savepoint rollback already expunges the row it was protecting, so a
        second, manual expunge finds nothing there and raises
        ``InvalidRequestError`` instead.
        """
        try:
            async with self.db.begin_nested():
                ceilings = [self._build_member_budget(member_id, default, budget) for member_id in member_ids]
                self.db.add_all(ceilings)
                await self.db.flush()
            return ceilings
        except IntegrityError:
            pass  # fall back below; the savepoint already rolled the batch back

        created: list[ScopedBudget] = []
        for member_id in member_ids:
            try:
                async with self.db.begin_nested():
                    ceiling = self._build_member_budget(member_id, default, budget)
                    self.db.add(ceiling)
                    await self.db.flush()
                created.append(ceiling)
            except IntegrityError:
                pass  # this id's savepoint rolled back and expunged it; move on
        return created

    @staticmethod
    def _build_member_budget(
        member_id: uuid.UUID, default: WorkspaceBudgetDefault, budget: Budget
    ) -> ScopedBudget:
        """One member's ceiling, naming the budget the default hands out.

        Named rather than copied: the limit and the period are read through the
        budget on every request, so editing that budget moves everyone already
        holding a ceiling from it. That is the point of a budget being a named
        thing rather than a figure duplicated per member.

        Only the period *window* is stamped here, because a window is this
        member's own: two people materialized a week apart from one rolling
        monthly budget are each a month from their own start. A calendar-aligned
        budget lands them both on the same boundary, which is what the shared
        derivation is for.
        """
        window = period_window(
            datetime.now(UTC),
            duration=budget.budget_duration_sec,
            alignment=budget.reset_alignment,
        )
        period_start, period_end = window if window is not None else (None, None)
        return ScopedBudget(
            scope_type=_SCOPE_WORKSPACE_MEMBER,
            scope_id=str(member_id),
            provider_key_id=default.provider_key_id,
            budget_id=budget.budget_id,
            period_start=period_start,
            period_end=period_end,
        )

    async def _budget_for(self, default: WorkspaceBudgetDefault) -> Budget:
        """The budget a stored default hands out.

        Always present: ``budget_id`` is NOT NULL and its foreign key is
        ``RESTRICT``, so the row cannot be deleted while a default names it. The
        miss is still raised rather than asserted, because a database restored
        without foreign keys enforced would otherwise materialize a ceiling with
        a null limit, which admits everything.
        """
        budget = await self.db.get(Budget, default.budget_id)
        if budget is None:
            raise WorkspaceBudgetDefaultBudgetNotFoundError(default.budget_id)
        return budget

    async def _require_budget(self, budget_id: str) -> Budget:
        """The budget a caller named, refused as 404 when it does not exist."""
        budget = await self.db.get(Budget, budget_id)
        if budget is None:
            raise WorkspaceBudgetDefaultBudgetNotFoundError(budget_id)
        return budget

    async def _existing_member_budget(self, member_id: uuid.UUID, provider_key_id: str | None) -> ScopedBudget | None:
        stmt = select(ScopedBudget).where(
            ScopedBudget.scope_type == _SCOPE_WORKSPACE_MEMBER,
            ScopedBudget.scope_id == str(member_id),
        )
        stmt = stmt.where(
            ScopedBudget.provider_key_id == provider_key_id
            if provider_key_id is not None
            else ScopedBudget.provider_key_id.is_(None)
        )
        return (await self.db.execute(stmt)).scalars().first()

    # ------------------------------------------------------------------
    # CRUD (commit on success)
    # ------------------------------------------------------------------

    async def _get_or_404(self, workspace: Workspace, default_id: str) -> WorkspaceBudgetDefault:
        default = await self.db.get(WorkspaceBudgetDefault, default_id)
        if default is None or default.workspace_id != workspace.id:
            raise WorkspaceBudgetDefaultNotFoundError(default_id)
        return default

    async def list_defaults(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> WorkspaceMemberBudgetPoliciesPublic:
        """List a page of a workspace's defaults, plus the total. Any member of the workspace may read it."""
        await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        limit = min(limit, _MAX_LIST_LIMIT)

        count = (
            await self.db.execute(
                select(func.count())
                .select_from(WorkspaceBudgetDefault)
                .where(col(WorkspaceBudgetDefault.workspace_id) == workspace_id)
            )
        ).scalar_one()

        defaults = (
            (
                await self.db.execute(
                    select(WorkspaceBudgetDefault)
                    .where(col(WorkspaceBudgetDefault.workspace_id) == workspace_id)
                    .order_by(col(WorkspaceBudgetDefault.created_at), col(WorkspaceBudgetDefault.id))
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        # One query for the page's budgets rather than one per row: a workspace
        # has few defaults, but the N+1 is free to avoid and this is the read the
        # dashboard makes on every workspace switch.
        budgets = {
            budget.budget_id: budget
            for budget in (
                await self.db.execute(
                    select(Budget).where(col(Budget.budget_id).in_([default.budget_id for default in defaults]))
                )
            )
            .scalars()
            .all()
        }
        return WorkspaceMemberBudgetPoliciesPublic(
            data=[
                WorkspaceMemberBudgetPolicyPublic.from_model(default, budgets[default.budget_id])
                for default in defaults
                if default.budget_id in budgets
            ],
            count=count,
        )

    async def create_default(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        request: WorkspaceMemberBudgetPolicyCreate,
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Create a default, materializing it onto every existing active member."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )

        # Serialized against every membership-creation path's own lock on this
        # row (see `WorkspaceRepository.lock`): without it, a concurrent
        # `add_member` can read the pre-creation set of defaults, and this call
        # can read the pre-join set of members, and the new member gets neither
        # ceiling even though both transactions commit successfully.
        await self.workspaces.lock(workspace.id)

        # Before the lock's write, so an unknown budget is a 404 rather than a
        # foreign-key violation reported as "this default already exists".
        budget = await self._require_budget(request.budget_id)
        default = WorkspaceBudgetDefault(
            workspace_id=workspace.id,
            budget_id=budget.budget_id,
            provider_key_id=request.provider_key_id,
        )
        self.db.add(default)
        # Narrowed to the default row's own flush on purpose: this is the only
        # write `WorkspaceBudgetDefaultAlreadyExistsError`'s message describes
        # (the template's own unique index). `materialize_for_default` handles
        # its own, unrelated `IntegrityError`s internally (see
        # `_insert_member_budgets`) precisely so one never reaches here and
        # gets misreported as "this default already exists", rolling back a
        # template that in fact does not conflict with anything.
        try:
            await self.db.flush()
        except IntegrityError:
            await self.db.rollback()
            raise WorkspaceBudgetDefaultAlreadyExistsError(workspace_id, request.provider_key_id) from None

        await self.materialize_for_default(default, budget)
        await self.db.commit()
        await self.db.refresh(default)
        return WorkspaceMemberBudgetPolicyPublic.from_model(default, budget)

    async def update_default(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        default_id: str,
        request: WorkspaceMemberBudgetPolicyUpdate,
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Point a default at a different budget.

        Members already materialized keep the budget they were handed (see
        :class:`WorkspaceMemberBudgetPolicyUpdate`); this changes what someone
        joining afterwards gets. The scope (``provider_key_id``) is not editable,
        matching ``routes/scoped_budgets.py``'s own rule for the concrete
        ceilings this produces: changing it would move the template to a
        different identity, which is a delete and a create, not an update.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        default = await self._get_or_404(workspace, default_id)

        budget = await self._require_budget(request.budget_id)
        default.budget_id = budget.budget_id

        await self.db.commit()
        await self.db.refresh(default)
        return WorkspaceMemberBudgetPolicyPublic.from_model(default, budget)

    async def delete_default(self, *, user: User, workspace_id: uuid.UUID, default_id: str) -> None:
        """Delete a default.

        Only the template row is removed. The ``ScopedBudget`` rows it already
        materialized (and their spend history) are left exactly as they are;
        a member joining afterwards no longer gets one from it. Matches
        ``WorkspaceService.delete_workspace``'s own preference for a stated,
        intentional non-cascade over a silent one.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        default = await self._get_or_404(workspace, default_id)
        await self.db.delete(default)
        await self.db.commit()


__all__ = [
    "WorkspaceBudgetDefaultService",
    "WorkspaceMemberBudgetPoliciesPublic",
    "WorkspaceMemberBudgetPolicyCreate",
    "WorkspaceMemberBudgetPolicyPublic",
    "WorkspaceMemberBudgetPolicyUpdate",
]
