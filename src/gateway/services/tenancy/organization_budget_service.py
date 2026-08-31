"""An organization's own spend budgets and the ceilings that enforce them.

The tenant-scoped half of budgets. ``/v1/budgets`` and ``/v1/scoped-budgets``
stay what they are, the deployment's own surface behind
``require_deployment_operator``; this is what an organization admin gets
instead, and the roles matrix (otari-ai#1943) is what says they should have one:
Spend & budgets is Hidden for a member and Edit for an admin, and before this
both routers refused a tenant outright, so a hosted organization could manage
neither its own caps nor the ceilings holding them.

Two objects, because the schema has two and they answer different questions:

- A :class:`Budget` is *what a cap is*: a USD figure and the period it is spent
  over. ``budgets.organization_id`` says whose it is, and this service only ever
  reads or writes rows carrying the caller's own.
- A :class:`ScopedBudget` is *who is capped*: an organization, a workspace, a
  membership in either, or a single API key, optionally narrowed to one provider
  instance. It names a budget and holds its own counters, so the limit is read
  through the budget and never copied.

Three rules live here rather than in the route.

**The caller's organization is resolved from their identity, never from the
request.** Scoped to ``/me`` for the reason ``routes/organizations.py`` is:
a request cannot name an organization at all, so there is no parameter to
confuse with an authorization decision.

**Only a management role may read or write.** Unlike the pricing overrides,
where a member may see what their own requests are priced at, this is
Hidden for a member in the matrix: a cap is a statement about what colleagues
may spend, and the roster it implies is not a member's to read. So both halves
take ``require_active_organization_management_access``.

**Every scope must resolve into the caller's own organization, and every budget
must be owned by it.** A scope id is a bare uuid on the wire with nothing in it
saying which tenant it belongs to, so an unresolved one is the whole of the
cross-tenant risk on this surface: without the check, an admin could cap another
organization's workspace, or point their own ceiling at another organization's
budget and then edit that budget's figure. Both answer 404 rather than 403, for
the reason :class:`TenancyNotFoundError` gives: another tenant's row must not be
distinguishable from one that was never created.

A **NULL** ``organization_id`` is the deployment's own budget: the rows that
predate ``b7e1c4a9d2f5`` and the ones the otari-ai cutover mints, which are
shared across tenants by shape and so have no single owner. Nothing here lists,
offers or repoints one. An organization's *existing* ceilings may still name one,
because the cutover wrote them that way, so a ceiling reads back with the real
figures it enforces (they are carried on the ceiling, read through the budget)
and ``manageable`` false, rather than being hidden and leaving the page claiming
the organization is uncapped.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import col

from gateway.models.entities import APIKey, Budget, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.entities import User as GatewayUser
from gateway.models.money import MAX_USD_LIMIT, as_float, to_usd_or_none
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.budget_periods import ResetAlignment, period_window
from gateway.services.budget_retiming import cadence_of, retime_ceilings_for_budget
from gateway.services.tenancy.errors import (
    OrganizationBudgetHeldElsewhereError,
    OrganizationBudgetInUseError,
    OrganizationBudgetNotFoundError,
    OrganizationScopedBudgetAlreadyExistsError,
    OrganizationScopedBudgetNotFoundError,
    OrganizationScopeNotFoundError,
    TenancyValidationError,
)
from gateway.services.tenancy.organization_service import OrganizationService

# The scopes this surface understands, spelled out rather than imported from
# `scoped_budget_service`: that module reaches `workspace_scope`, which reaches
# `tenancy.provisioning_service`, which runs `tenancy/__init__`, which imports
# this package. `WorkspaceService._delete_scoped_budgets_for` avoids the same
# cycle the same way, and `tests/unit/test_service_module_imports.py` pins it.
# The values are identical to `ScopeType`, and
# `tests/unit/test_organization_budget_scopes.py` asserts that rather than
# trusting it.
SCOPE_ORGANIZATION = "organization"
SCOPE_WORKSPACE = "workspace"
SCOPE_WORKSPACE_MEMBER = "workspace_member"
SCOPE_ORG_MEMBER = "org_member"
SCOPE_API_TOKEN = "api_token"

# Spelled as a `Literal` and not just as the constants above, because the
# `Literal` is what puts the allowed values in the OpenAPI schema and refuses an
# unknown one at the boundary, exactly as `ScopeType` does for the deployment
# router. The constants stay for the resolution code, where a bare string
# comparison reads worse than a name.
OrganizationScopeType = Literal["organization", "workspace", "workspace_member", "org_member", "api_token"]
ORGANIZATION_SCOPE_TYPES: tuple[str, ...] = get_args(OrganizationScopeType)

_MAX_LIST_LIMIT = 1000

_PERIOD_DESCRIPTION = (
    "Seconds between resets, counted from the last one. Mutually exclusive with reset_alignment"
)
_ALIGNMENT_DESCRIPTION = (
    "Reset on a UTC calendar boundary instead of a fixed number of seconds, which is the only way "
    "to express a calendar month. Mutually exclusive with budget_duration_sec"
)


def _require_single_period_source(duration: int | None, alignment: str | None) -> None:
    """Refuse the state ``ck_budgets_single_period_source`` refuses, with a message.

    The same rule ``routes/budgets.py`` states for the deployment surface, and it
    has to be stated on this one too: a period comes from a duration or from a
    calendar boundary, and both set is one concept encoded twice, storable only
    with an "ignored when" rule to give it a meaning. Checked here so the refusal
    is a 400 naming the pair rather than the CHECK surfacing as a 500.
    """
    if duration is not None and alignment is not None:
        raise TenancyValidationError("A budget resets on budget_duration_sec or on reset_alignment, not both")


class OrganizationBudgetRates(BaseModel):
    """The figure and the period a budget holds, shared by the create and update bodies."""

    name: str | None = Field(default=None, max_length=200, description="Admin-facing label for the budget")
    max_budget: float | None = Field(
        default=None,
        ge=0,
        le=MAX_USD_LIMIT,
        description="Maximum spend in USD over one period; null caps nothing",
    )
    budget_duration_sec: int | None = Field(default=None, gt=0, description=_PERIOD_DESCRIPTION)
    # The `Literal`, not a bare `str`: an unrecognized alignment is stored happily
    # and then raises out of `period_window` the first time a window is derived
    # from it, which is a 500 on creating a ceiling or on retiming one rather than
    # a 422 on the request that introduced it. `_roll_expired_periods` degrades
    # safely (it logs and leaves the exhausted window in place) but the API paths
    # do not, so the value is refused at the boundary and published in the schema,
    # matching what `CreateBudgetRequest` on the deployment route already does.
    #
    # The response models keep `str | None` on purpose: they echo whatever is
    # stored, and narrowing them would turn a row holding an unexpected value into
    # a failed read rather than a readable row someone can go and fix.
    reset_alignment: ResetAlignment | None = Field(default=None, description=_ALIGNMENT_DESCRIPTION)


class OrganizationBudgetCreate(OrganizationBudgetRates):
    """Create one budget owned by the caller's organization."""


class OrganizationBudgetUpdate(OrganizationBudgetRates):
    """Replace a budget's label, figure and period.

    Every field is optional and keyed on ``model_fields_set``, matching
    ``PATCH /v1/budgets/{id}``'s own: an *omitted* field is left alone, and an
    explicit null clears it, so sending ``max_budget: null`` takes a budget back
    to uncapped, which is what the dashboard's dialog does. The period pair is
    still mutually exclusive, and setting one does not clear the other, which is
    why :func:`_require_single_period_source` re-checks the *resulting* pair
    rather than the submitted one.
    """


class OrganizationBudgetPublic(BaseModel):
    """One of the organization's budgets, and how much of its own config names it.

    Carries no spend rollup. ``BudgetResponse`` on the deployment surface sums
    ``users.spend`` over the gateway's ``users`` table, which is deployment-wide
    and has no tenancy column, so the same figure here would be a cross-tenant
    read. What an organization's own spend is, is a question for Usage.

    ``ceiling_count`` is the organization-relevant fact instead: how many of its
    ceilings this budget currently holds, which is what makes a delete refuse.
    """

    budget_id: str
    organization_id: uuid.UUID
    name: str | None
    max_budget: float | None
    budget_duration_sec: int | None
    reset_alignment: str | None
    ceiling_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, budget: Budget, *, ceiling_count: int) -> OrganizationBudgetPublic:
        # `organization_id` is narrowed rather than declared optional: every row
        # this service returns was filtered on the caller's own, so a null here
        # would be a bug in the query and not a state the wire should describe.
        if budget.organization_id is None:  # pragma: no cover - the queries filter it
            raise OrganizationBudgetNotFoundError(budget.budget_id)
        return cls(
            budget_id=budget.budget_id,
            organization_id=budget.organization_id,
            name=budget.name,
            # Narrowed on the way out: the cap is exact in the database, while
            # the wire contract and the dashboard client stay float.
            max_budget=as_float(budget.max_budget),
            budget_duration_sec=budget.budget_duration_sec,
            reset_alignment=budget.reset_alignment,
            ceiling_count=ceiling_count,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


class OrganizationBudgetsPublic(BaseModel):
    data: list[OrganizationBudgetPublic]
    count: int


class OrganizationScopedBudgetCreate(BaseModel):
    """Attach one of the organization's budgets to a scope inside it."""

    scope_type: OrganizationScopeType = Field(description="Which kind of identity this ceiling caps")
    scope_id: str = Field(
        min_length=1,
        max_length=255,
        description=(
            "Id of the capped identity: this organization, one of its workspaces, "
            "a membership in either, or an API key in one"
        ),
    )
    # Absent means every provider; a present value must name a real instance.
    # Resolution matches `provider_key_id == provider_instance OR IS NULL`, and a
    # blank string is neither, so it would store, list, and never bind. Refused
    # rather than folded into null, because null is the *wider* cap and coercing
    # would silently cap more than the caller asked for.
    provider_key_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=r"^\S+$",
        description=(
            "Narrow the cap to one provider instance; omit or null to cap spend across every provider. "
            "Must name a real instance: a blank value would store a ceiling that never binds"
        ),
    )
    budget_id: str = Field(
        min_length=1,
        max_length=255,
        description="The budget this ceiling enforces, which must be one this organization owns",
    )
    name: str | None = Field(default=None, max_length=200, description="Admin-facing label for this ceiling")


class OrganizationScopedBudgetUpdate(BaseModel):
    """Relabel a ceiling, or point it at a different budget of this organization's.

    The scope and the provider narrowing are not editable, for the reason
    ``PATCH /v1/scoped-budgets/{id}`` gives: changing either moves the ceiling to
    a different identity while carrying its spend, which is a delete and a
    create, not an update.
    """

    budget_id: str | None = Field(default=None, min_length=1, max_length=255)
    name: str | None = Field(default=None, max_length=200)


class OrganizationScopedBudgetPublic(BaseModel):
    """One ceiling inside the organization, and the figures it enforces.

    The limit and the period are read through the budget rather than stored here,
    and carried on the wire so a page can render a ceiling without fetching every
    budget to resolve one id. Same reasoning as ``ScopedBudgetResponse``, whose
    shape this deliberately mirrors.
    """

    id: str
    scope_type: str
    scope_id: str
    provider_key_id: str | None
    budget_id: str
    name: str | None
    max_budget: float | None
    current_spend: float
    reserved_spend: float
    budget_duration_sec: int | None
    reset_alignment: str | None
    period_start: str | None
    period_end: str | None
    # Whether the budget behind this ceiling is one this organization owns, and
    # so whether its figure can be changed here at all. False for a ceiling the
    # otari-ai cutover pointed at a deployment budget: the ceiling is real and
    # enforcing, and the amount it holds is set outside this organization.
    manageable: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_model(
        cls,
        ceiling: ScopedBudget,
        budget: Budget,
        *,
        organization_id: uuid.UUID,
    ) -> OrganizationScopedBudgetPublic:
        return cls(
            id=ceiling.id,
            scope_type=ceiling.scope_type,
            scope_id=ceiling.scope_id,
            provider_key_id=ceiling.provider_key_id,
            budget_id=ceiling.budget_id,
            name=ceiling.name,
            max_budget=as_float(budget.max_budget),
            current_spend=float(ceiling.current_spend),
            reserved_spend=float(ceiling.reserved_spend),
            budget_duration_sec=budget.budget_duration_sec,
            reset_alignment=budget.reset_alignment,
            period_start=ceiling.period_start.isoformat() if ceiling.period_start else None,
            period_end=ceiling.period_end.isoformat() if ceiling.period_end else None,
            manageable=budget.organization_id == organization_id,
            created_at=ceiling.created_at.isoformat(),
            updated_at=ceiling.updated_at.isoformat(),
        )


class OrganizationScopedBudgetsPublic(BaseModel):
    data: list[OrganizationScopedBudgetPublic]
    count: int


class OrganizationBudgetService:
    """Read and write the caller's organization's budgets and spend ceilings."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db)

    # ------------------------------------------------------------------
    # Identity and scope resolution
    # ------------------------------------------------------------------

    async def _managed_organization(self, user: User) -> Organization:
        """The caller's organization, having checked they may manage its spend.

        One gate for reads and writes alike. See the module docstring: the matrix
        puts this surface at Hidden for a member, which is a stronger line than
        the pricing overrides take, and a cap names colleagues.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )
        return organization

    async def _scope_organization_id(self, *, scope_type: str, scope_id: str) -> uuid.UUID | None:
        """Which organization a scope belongs to, or None when it resolves to nothing.

        The whole cross-tenant check on this surface, so it resolves every scope
        rather than trusting any of them. ``api_token`` keys on a string id and
        the rest on UUIDs, matching ``_SCOPE_SUBJECTS`` on the deployment router;
        a scope id that is not a UUID where one is required resolves to nothing
        rather than raising, because a typo and another tenant's row have to be
        the same answer.
        """
        if scope_type == SCOPE_API_TOKEN:
            key = (
                await self.db.execute(select(APIKey.workspace_id).where(APIKey.id == scope_id))
            ).scalar_one_or_none()
            if key is None:
                return None
            return await self._workspace_organization_id(key)

        try:
            identifier = uuid.UUID(scope_id)
        except ValueError:
            return None

        if scope_type == SCOPE_ORGANIZATION:
            found = (
                await self.db.execute(select(col(Organization.id)).where(col(Organization.id) == identifier))
            ).scalar_one_or_none()
            return found
        if scope_type == SCOPE_WORKSPACE:
            return await self._workspace_organization_id(identifier)
        if scope_type == SCOPE_ORG_MEMBER:
            return (
                await self.db.execute(
                    select(col(OrganizationMember.organization_id)).where(col(OrganizationMember.id) == identifier)
                )
            ).scalar_one_or_none()
        if scope_type == SCOPE_WORKSPACE_MEMBER:
            workspace_id = (
                await self.db.execute(
                    select(col(WorkspaceMember.workspace_id)).where(col(WorkspaceMember.id) == identifier)
                )
            ).scalar_one_or_none()
            if workspace_id is None:
                return None
            return await self._workspace_organization_id(workspace_id)
        # Not reachable through the routes, which validate `scope_type` against
        # `ORGANIZATION_SCOPE_TYPES` in the schema, so an unknown one here is a
        # caller inside this process and resolving to nothing is the safe answer.
        return None

    async def _workspace_organization_id(self, workspace_id: uuid.UUID) -> uuid.UUID | None:
        return (
            await self.db.execute(
                select(col(Workspace.organization_id)).where(col(Workspace.id) == workspace_id)
            )
        ).scalar_one_or_none()

    async def _require_scope_in_organization(
        self,
        *,
        organization: Organization,
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Refuse a scope that resolves to nothing, or into another organization.

        Both as 404 and with one message, so the response cannot be read as an
        oracle for whether another tenant holds that id.
        """
        if scope_type not in ORGANIZATION_SCOPE_TYPES:
            raise TenancyValidationError(f"Unknown scope type: {scope_type}")
        owner = await self._scope_organization_id(scope_type=scope_type, scope_id=scope_id)
        if owner is None or owner != organization.id:
            raise OrganizationScopeNotFoundError(scope_type, scope_id)

    async def _require_own_budget(self, *, organization: Organization, budget_id: str) -> Budget:
        """A budget this organization owns, or 404.

        Filtered on ``organization_id`` in the query rather than fetched and
        checked, so a deployment budget and another tenant's are one answer and
        neither can be told from a budget that does not exist.
        """
        budget = (
            await self.db.execute(
                select(Budget).where(
                    Budget.budget_id == budget_id,
                    Budget.organization_id == organization.id,
                )
            )
        ).scalar_one_or_none()
        if budget is None:
            raise OrganizationBudgetNotFoundError(budget_id)
        return budget

    # ------------------------------------------------------------------
    # Budgets
    # ------------------------------------------------------------------

    async def list_budgets(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> OrganizationBudgetsPublic:
        """A page of the organization's own budgets, with how many ceilings hold each."""
        organization = await self._managed_organization(user)
        limit = min(limit, _MAX_LIST_LIMIT)

        count = (
            await self.db.execute(
                select(func.count()).select_from(Budget).where(Budget.organization_id == organization.id)
            )
        ).scalar_one()

        budgets = (
            (
                await self.db.execute(
                    select(Budget)
                    .where(Budget.organization_id == organization.id)
                    .order_by(Budget.created_at, Budget.budget_id)
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )

        # One grouped count for the page rather than one query per row: a long
        # lived organization has many budgets and this is the read its Spend page
        # makes first.
        held: dict[str, int] = {
            budget_id: count
            for budget_id, count in (
                await self.db.execute(
                    select(ScopedBudget.budget_id, func.count())
                    .where(ScopedBudget.budget_id.in_([budget.budget_id for budget in budgets]))
                    .group_by(ScopedBudget.budget_id)
                )
            ).all()
        }
        return OrganizationBudgetsPublic(
            data=[
                OrganizationBudgetPublic.from_model(budget, ceiling_count=held.get(budget.budget_id, 0))
                for budget in budgets
            ],
            count=count,
        )

    async def create_budget(self, *, user: User, request: OrganizationBudgetCreate) -> OrganizationBudgetPublic:
        """Create a budget owned by the caller's organization."""
        organization = await self._managed_organization(user)
        _require_single_period_source(request.budget_duration_sec, request.reset_alignment)

        budget = Budget(
            organization_id=organization.id,
            name=request.name,
            max_budget=to_usd_or_none(request.max_budget),
            budget_duration_sec=request.budget_duration_sec,
            reset_alignment=request.reset_alignment,
        )
        self.db.add(budget)
        await self.db.commit()
        await self.db.refresh(budget)
        # Freshly created, so nothing can name it yet.
        return OrganizationBudgetPublic.from_model(budget, ceiling_count=0)

    async def update_budget(
        self,
        *,
        user: User,
        budget_id: str,
        request: OrganizationBudgetUpdate,
    ) -> OrganizationBudgetPublic:
        """Change one of the organization's budgets.

        Every ceiling naming it moves with it, which is the point of naming one:
        a budget is the figure, and the ceilings are where it applies.

        **Changing the cadence retimes every ceiling naming this budget**, in this
        transaction. A ceiling holds its own window and reads the cadence through
        the budget, so leaving the windows alone would leave the two disagreeing,
        and in one direction that is an enforcement bug rather than a cosmetic
        one: ``_roll_expired_periods`` is guarded on ``period_end IS NOT NULL``,
        so a budget moved from "no reset" to a periodic cadence would leave its
        ceilings with NULL windows that never roll, accumulating spend forever
        while this API reported the new cadence.

        The counters are deliberately **not** zeroed, matching what repointing a
        ceiling at a different budget already does: spend already recorded stays,
        and the ceiling is the same allowance held to a different figure from here
        on. ``reserved_spend`` is untouched so a hold taken before the change is
        still released against the right counter.
        """
        organization = await self._managed_organization(user)
        budget = await self._require_own_budget(organization=organization, budget_id=budget_id)
        # Read before the mutation, because that is what says whether the cadence
        # moved at all: retiming a ceiling whose cadence did not change would
        # restart its window for a rename or a figure change, throwing away the
        # part of the period it had already spent.
        cadence_before = cadence_of(budget.budget_duration_sec, budget.reset_alignment)

        fields = request.model_fields_set
        if "name" in fields:
            budget.name = request.name
        if "max_budget" in fields:
            budget.max_budget = to_usd_or_none(request.max_budget)
        if "budget_duration_sec" in fields:
            budget.budget_duration_sec = request.budget_duration_sec
        if "reset_alignment" in fields:
            budget.reset_alignment = request.reset_alignment
        # The *resulting* pair, not the submitted one: setting a duration on a
        # budget that already resets on a calendar boundary is what the CHECK
        # refuses, and neither field alone looks wrong.
        _require_single_period_source(budget.budget_duration_sec, budget.reset_alignment)

        if cadence_of(budget.budget_duration_sec, budget.reset_alignment) != cadence_before:
            # Shared with the deployment-wide surface rather than written twice,
            # so a rule this important cannot hold on one and not the other.
            await retime_ceilings_for_budget(
                self.db,
                budget_id=budget.budget_id,
                duration=budget.budget_duration_sec,
                alignment=budget.reset_alignment,
            )

        await self.db.commit()
        await self.db.refresh(budget)
        return OrganizationBudgetPublic.from_model(
            budget,
            ceiling_count=await self._ceiling_count(budget.budget_id),
        )

    async def delete_budget(self, *, user: User, budget_id: str) -> None:
        """Delete one of the organization's budgets, refusing while anything names it.

        ``scoped_budgets.budget_id`` and ``workspace_budget_defaults.budget_id``
        are both RESTRICT, so the database would refuse anyway, as an
        ``IntegrityError`` with nothing naming what to go and change. Checked here
        so the refusal can say which.

        ``users.budget_id`` is counted but not named. It can hold a tenant's
        budget, because ``GET /v1/budgets`` is unfiltered and ``POST /v1/users``
        accepts any id it lists, and ``Budget.users`` is a plain relationship, so
        deleting the budget would not refuse: the ORM nulls the column out and
        the assignment an operator made disappears with no refusal to either of
        them. The count is what stops that, and it says only that the budget is
        held, because saying "3 users" to an admin who cannot see the users page
        would name a thing they cannot act on.

        ``budget_reset_logs.budget_id`` is the same shape with a NOT NULL column,
        so its null-out fails instead, as an ``IntegrityError`` at the commit.
        Guarded there rather than counted, since a reset log only exists for a
        budget a user already held.
        """
        organization = await self._managed_organization(user)
        budget = await self._require_own_budget(organization=organization, budget_id=budget_id)

        ceilings = await self._ceiling_count(budget.budget_id)
        defaults = (
            await self.db.execute(
                select(func.count())
                .select_from(WorkspaceBudgetDefault)
                .where(WorkspaceBudgetDefault.budget_id == budget.budget_id)
            )
        ).scalar_one()
        if ceilings or defaults:
            raise OrganizationBudgetInUseError(budget.budget_id, ceilings=ceilings, defaults=defaults)

        assigned = (
            await self.db.execute(
                select(func.count()).select_from(GatewayUser).where(GatewayUser.budget_id == budget.budget_id)
            )
        ).scalar_one()
        if assigned:
            raise OrganizationBudgetHeldElsewhereError(budget.budget_id)

        await self.db.delete(budget)
        try:
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise OrganizationBudgetHeldElsewhereError(budget_id) from None

    async def _ceiling_count(self, budget_id: str) -> int:
        return (
            await self.db.execute(
                select(func.count()).select_from(ScopedBudget).where(ScopedBudget.budget_id == budget_id)
            )
        ).scalar_one()

    # ------------------------------------------------------------------
    # Ceilings
    # ------------------------------------------------------------------

    async def _organization_scope_filter(self, organization: Organization) -> ColumnElement[bool]:
        """The predicate matching every ceiling whose scope sits in this organization.

        Built from the organization's own ids rather than by resolving each row,
        because ``scoped_budgets`` holds no tenancy column and no foreign key: the
        table is keyed on ``(scope_type, scope_id)`` strings by design, so the
        only way to ask "which of these are mine" is to name the ids that are.
        Four id sets, one query each, which is bounded by the size of the
        organization rather than by the number of ceilings.
        """
        workspace_ids = (
            (
                await self.db.execute(
                    select(col(Workspace.id)).where(col(Workspace.organization_id) == organization.id)
                )
            )
            .scalars()
            .all()
        )
        org_member_ids = (
            (
                await self.db.execute(
                    select(col(OrganizationMember.id)).where(
                        col(OrganizationMember.organization_id) == organization.id
                    )
                )
            )
            .scalars()
            .all()
        )
        workspace_member_ids = (
            (
                await self.db.execute(
                    select(col(WorkspaceMember.id)).where(col(WorkspaceMember.workspace_id).in_(workspace_ids))
                )
            )
            .scalars()
            .all()
        )
        key_ids = (
            (await self.db.execute(select(APIKey.id).where(APIKey.workspace_id.in_(workspace_ids)))).scalars().all()
        )

        workspace_strings = [str(value) for value in workspace_ids]
        return or_(
            (ScopedBudget.scope_type == SCOPE_ORGANIZATION) & (ScopedBudget.scope_id == str(organization.id)),
            (ScopedBudget.scope_type == SCOPE_WORKSPACE) & ScopedBudget.scope_id.in_(workspace_strings),
            (ScopedBudget.scope_type == SCOPE_ORG_MEMBER)
            & ScopedBudget.scope_id.in_([str(value) for value in org_member_ids]),
            (ScopedBudget.scope_type == SCOPE_WORKSPACE_MEMBER)
            & ScopedBudget.scope_id.in_([str(value) for value in workspace_member_ids]),
            (ScopedBudget.scope_type == SCOPE_API_TOKEN) & ScopedBudget.scope_id.in_(list(key_ids)),
        )

    async def list_ceilings(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> OrganizationScopedBudgetsPublic:
        """A page of the ceilings that cap identities inside this organization.

        Includes the ones naming a deployment budget, reported with ``manageable``
        false: they are enforcing against this organization's spend today (the
        otari-ai cutover wrote them), so leaving them out would let the page read
        as uncapped.
        """
        organization = await self._managed_organization(user)
        limit = min(limit, _MAX_LIST_LIMIT)
        scope_filter = await self._organization_scope_filter(organization)

        count = (
            await self.db.execute(select(func.count()).select_from(ScopedBudget).where(scope_filter))
        ).scalar_one()

        # Joined rather than a lookup per row: the limit and the period live on
        # the budget, so a page of ceilings would otherwise be a page of round
        # trips. The join is inner, which is safe because `budget_id` is NOT NULL
        # with a RESTRICT foreign key, so a ceiling always has its budget.
        rows = (
            await self.db.execute(
                select(ScopedBudget, Budget)
                .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
                .where(scope_filter)
                .order_by(ScopedBudget.created_at, ScopedBudget.id)
                .offset(skip)
                .limit(limit)
            )
        ).all()
        return OrganizationScopedBudgetsPublic(
            data=[
                OrganizationScopedBudgetPublic.from_model(ceiling, budget, organization_id=organization.id)
                for ceiling, budget in rows
            ],
            count=count,
        )

    async def create_ceiling(
        self,
        *,
        user: User,
        request: OrganizationScopedBudgetCreate,
    ) -> OrganizationScopedBudgetPublic:
        """Cap one identity inside the organization at one of its budgets."""
        organization = await self._managed_organization(user)
        await self._require_scope_in_organization(
            organization=organization,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
        )
        budget = await self._require_own_budget(organization=organization, budget_id=request.budget_id)

        # The window opens now rather than on first spend, so a period-limited
        # ceiling has a defined end before any request has arrived. An aligned one
        # opens on the boundary it is already past, so its first period is the
        # remainder of the calendar period it was created in. Same derivation as
        # `POST /v1/scoped-budgets`, through the leaf module both import.
        window = period_window(
            datetime.now(UTC),
            duration=budget.budget_duration_sec,
            alignment=budget.reset_alignment,
        )
        period_start, period_end = window if window is not None else (None, None)

        # Checked before the insert rather than caught as an IntegrityError,
        # because the two partial unique indexes on `scoped_budgets` are what
        # would refuse it and neither is nameable in a message a human can act on.
        # The race that leaves is closed by the index itself, which still refuses.
        await self._require_no_existing_ceiling(request)

        ceiling = ScopedBudget(
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            provider_key_id=request.provider_key_id,
            budget_id=budget.budget_id,
            name=request.name,
            period_start=period_start,
            period_end=period_end,
        )
        self.db.add(ceiling)
        await self.db.commit()
        await self.db.refresh(ceiling)
        return OrganizationScopedBudgetPublic.from_model(ceiling, budget, organization_id=organization.id)

    async def _require_no_existing_ceiling(self, request: OrganizationScopedBudgetCreate) -> None:
        stmt = select(func.count()).select_from(ScopedBudget).where(
            ScopedBudget.scope_type == request.scope_type,
            ScopedBudget.scope_id == request.scope_id,
        )
        if request.provider_key_id is None:
            stmt = stmt.where(ScopedBudget.provider_key_id.is_(None))
        else:
            stmt = stmt.where(ScopedBudget.provider_key_id == request.provider_key_id)
        if (await self.db.execute(stmt)).scalar_one():
            raise OrganizationScopedBudgetAlreadyExistsError(request.scope_type, request.scope_id)

    async def update_ceiling(
        self,
        *,
        user: User,
        ceiling_id: str,
        request: OrganizationScopedBudgetUpdate,
    ) -> OrganizationScopedBudgetPublic:
        """Relabel a ceiling, or point it at a different budget of this organization's.

        Repointing takes a budget the organization owns, which is what stops an
        admin attaching their ceiling to a deployment budget or another tenant's
        and then editing that budget's figure. A ceiling that currently names a
        deployment budget can therefore be moved onto one of the organization's
        own, which is how a cutover-migrated ceiling becomes manageable.
        """
        organization = await self._managed_organization(user)
        ceiling = await self._require_own_ceiling(organization=organization, ceiling_id=ceiling_id)
        budget = (await self.db.execute(select(Budget).where(Budget.budget_id == ceiling.budget_id))).scalar_one()

        # Tri-state on `name`, keyed on `model_fields_set` rather than on the
        # value: omitting it leaves it alone, and an explicit null clears it back
        # to unnamed, which is a state a create can produce.
        if "name" in request.model_fields_set:
            ceiling.name = request.name
        if request.budget_id is not None and request.budget_id != ceiling.budget_id:
            budget = await self._require_own_budget(organization=organization, budget_id=request.budget_id)
            ceiling.budget_id = budget.budget_id
            # Retiming restarts the window from now rather than re-deriving an
            # end from a `period_start` belonging to the old budget's cadence.
            # Spend already recorded stays: the ceiling is the same allowance,
            # held to a different figure from here on.
            window = period_window(
                datetime.now(UTC),
                duration=budget.budget_duration_sec,
                alignment=budget.reset_alignment,
            )
            ceiling.period_start, ceiling.period_end = window if window is not None else (None, None)

        await self.db.commit()
        await self.db.refresh(ceiling)
        return OrganizationScopedBudgetPublic.from_model(ceiling, budget, organization_id=organization.id)

    async def delete_ceiling(self, *, user: User, ceiling_id: str) -> None:
        """Remove a ceiling inside the organization.

        A request holding a reservation against it settles into nothing
        afterwards, which is the right outcome: the ceiling no longer exists to be
        credited. Same note as ``DELETE /v1/scoped-budgets/{id}``.
        """
        organization = await self._managed_organization(user)
        ceiling = await self._require_own_ceiling(organization=organization, ceiling_id=ceiling_id)
        await self.db.delete(ceiling)
        await self.db.commit()

    async def _require_own_ceiling(self, *, organization: Organization, ceiling_id: str) -> ScopedBudget:
        """A ceiling whose scope sits in this organization, or 404.

        Resolved through the scope rather than through the budget it names: a
        cutover-migrated ceiling caps this organization while naming a deployment
        budget, and it is still this organization's ceiling to remove.
        """
        ceiling = (
            await self.db.execute(select(ScopedBudget).where(ScopedBudget.id == ceiling_id))
        ).scalar_one_or_none()
        if ceiling is None:
            raise OrganizationScopedBudgetNotFoundError(ceiling_id)
        owner = await self._scope_organization_id(scope_type=ceiling.scope_type, scope_id=ceiling.scope_id)
        if owner is None or owner != organization.id:
            raise OrganizationScopedBudgetNotFoundError(ceiling_id)
        return ceiling


__all__ = [
    "ORGANIZATION_SCOPE_TYPES",
    "OrganizationScopeType",
    "OrganizationBudgetCreate",
    "OrganizationBudgetPublic",
    "OrganizationBudgetService",
    "OrganizationBudgetUpdate",
    "OrganizationBudgetsPublic",
    "OrganizationScopedBudgetCreate",
    "OrganizationScopedBudgetPublic",
    "OrganizationScopedBudgetUpdate",
    "OrganizationScopedBudgetsPublic",
]
