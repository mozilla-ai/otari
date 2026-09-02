import uuid
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import get_db, require_deployment_operator
from gateway.models.entities import (
    MAX_COUNT_LIMIT,
    Budget,
    BudgetResetLog,
    ScopedBudget,
    User,
    WorkspaceBudgetDefault,
)
from gateway.models.money import MAX_USD_LIMIT, as_float, to_usd, to_usd_or_none
from gateway.models.tenancy import Workspace
from gateway.services.budget_retiming import cadence_of, retime_ceilings_for_budget
from gateway.services.scoped_budget_service import ResetAlignment

router = APIRouter(
    prefix="/v1/budgets",
    tags=["budgets"],
    dependencies=[Depends(require_deployment_operator)],
)

# The rollup below sums exact counters, so its coalesce default is exact too.
_ZERO = Decimal(0)


def _require_single_period_source(duration: int | None, alignment: str | None) -> None:
    """Refuse the state the table's CHECK refuses, with a message instead of a 500.

    A period comes from a duration or from a calendar boundary. Both set is one
    concept encoded twice, so the pair would need an "ignored when" rule to mean
    anything. This moved here with the cadence itself, from the scoped-ceiling
    route that used to own both.
    """
    if duration is not None and alignment is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A budget resets on budget_duration_sec or on reset_alignment, not both",
        )


class CreateBudgetRequest(BaseModel):
    """Request model for creating a new budget."""

    name: str | None = Field(default=None, description="Admin-facing label for the budget")
    max_budget: float | None = Field(default=None, ge=0, le=MAX_USD_LIMIT, description="Maximum spending limit")
    token_limit: int | None = Field(
        default=None,
        ge=0,
        le=MAX_COUNT_LIMIT,
        description="Maximum tokens over the period. Independent of max_budget; null is unlimited",
    )
    request_limit: int | None = Field(
        default=None,
        ge=0,
        le=MAX_COUNT_LIMIT,
        description="Maximum requests over the period. Independent of max_budget; null is unlimited",
    )
    budget_duration_sec: int | None = Field(
        default=None, gt=0, description="Budget duration in seconds (e.g., 86400 for daily, 604800 for weekly)"
    )
    reset_alignment: ResetAlignment | None = Field(
        default=None,
        description=(
            "Reset on a UTC calendar boundary instead of a fixed number of seconds, "
            "which is the only way to express a calendar month. Mutually exclusive with budget_duration_sec"
        ),
    )


class BudgetResponse(BaseModel):
    """Response model for budget information.

    ``max_budget``, ``token_limit`` and ``request_limit`` are the per-user
    ceilings, each independent and each unlimited when null, and multiple users
    can share one budget, so the usage rollup is an aggregate over the users
    assigned to this budget: how many there are and their combined ``spend`` /
    ``reserved``.
    Assigning users to a budget is done through the users API (dashboard support
    lands with user management), so a fresh gateway reports zeros here.
    """

    budget_id: str
    # Null for the deployment's own, and a tenant's organization when set. Carried
    # so a caller can tell the two apart before offering one: an organization's is
    # listed here for the operator to see, and `POST /v1/users` refuses to cap a
    # gateway user at it (mozilla-ai/otari#881).
    organization_id: uuid.UUID | None
    name: str | None
    max_budget: float | None
    token_limit: int | None
    request_limit: int | None
    budget_duration_sec: int | None
    reset_alignment: str | None
    created_at: str
    updated_at: str
    user_count: int = 0
    total_spend: float = 0.0
    total_reserved: float = 0.0

    @classmethod
    def from_model(
        cls,
        budget: "Budget",
        *,
        user_count: int = 0,
        total_spend: float = 0.0,
        total_reserved: float = 0.0,
    ) -> "BudgetResponse":
        """Create a BudgetResponse from a Budget ORM model and its usage rollup."""
        return cls(
            budget_id=budget.budget_id,
            organization_id=budget.organization_id,
            name=budget.name,
            # Narrowed on the way out: the cap is exact in the database, while
            # the wire contract and the dashboard client stay float.
            max_budget=as_float(budget.max_budget),
            token_limit=budget.token_limit,
            request_limit=budget.request_limit,
            budget_duration_sec=budget.budget_duration_sec,
            reset_alignment=budget.reset_alignment,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
            user_count=user_count,
            total_spend=total_spend,
            total_reserved=total_reserved,
        )


class UpdateBudgetRequest(BaseModel):
    """Request model for updating a budget."""

    name: str | None = Field(default=None)
    max_budget: float | None = Field(default=None, ge=0, le=MAX_USD_LIMIT)
    token_limit: int | None = Field(
        default=None,
        ge=0,
        le=MAX_COUNT_LIMIT,
        description="Maximum tokens over the period. Independent of max_budget; null is unlimited",
    )
    request_limit: int | None = Field(
        default=None,
        ge=0,
        le=MAX_COUNT_LIMIT,
        description="Maximum requests over the period. Independent of max_budget; null is unlimited",
    )
    budget_duration_sec: int | None = Field(default=None, gt=0)
    reset_alignment: ResetAlignment | None = Field(default=None)


class BudgetResetLogResponse(BaseModel):
    """Response model for one budget reset event (per user)."""

    id: int
    user_id: str | None
    budget_id: str
    previous_spend: float
    reset_at: str
    next_reset_at: str | None

    @classmethod
    def from_model(cls, log: BudgetResetLog) -> "BudgetResetLogResponse":
        return cls(
            id=log.id,
            user_id=log.user_id,
            budget_id=log.budget_id,
            previous_spend=float(log.previous_spend),
            reset_at=log.reset_at.isoformat(),
            next_reset_at=log.next_reset_at.isoformat() if log.next_reset_at else None,
        )


async def _budget_usage(db: AsyncSession, budget_id: str) -> tuple[int, float, float]:
    """Aggregate active-user spend for one budget: (user_count, total_spend, total_reserved)."""
    row = (
        await db.execute(
            select(
                func.count(),
                # ``Decimal`` defaults, not ``0.0``: in PostgreSQL
                # ``coalesce(numeric, double precision)`` resolves the whole sum
                # as double precision, which would roll exact counters up through
                # a binary float on the way to a page that reports them.
                func.coalesce(func.sum(User.spend), _ZERO),
                func.coalesce(func.sum(User.reserved), _ZERO),
            ).where(User.budget_id == budget_id, User.deleted_at.is_(None))
        )
    ).one()
    return int(row[0]), float(row[1]), float(row[2])


@router.post("")
async def create_budget(
    request: CreateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Create a new budget."""
    _require_single_period_source(request.budget_duration_sec, request.reset_alignment)
    budget = Budget(
        name=request.name,
        max_budget=to_usd_or_none(request.max_budget),
        token_limit=request.token_limit,
        request_limit=request.request_limit,
        budget_duration_sec=request.budget_duration_sec,
        reset_alignment=request.reset_alignment,
    )

    db.add(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)

    # A newly created budget has no users assigned yet, so the rollup is zero.
    return BudgetResponse.from_model(budget)


@router.get("")
async def list_budgets(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetResponse]:
    """List all budgets with pagination."""
    result = await db.execute(select(Budget).offset(skip).limit(limit))
    budgets = result.scalars().all()

    # One grouped query for the whole page instead of a per-budget aggregate, so
    # listing N budgets stays a fixed two queries rather than N+1. Scoped to the
    # page's ids: grouping over every budgeted user and then discarding all but
    # this page would make each call pay for the whole users table.
    page_ids = [budget.budget_id for budget in budgets]
    usage: dict[str, tuple[int, float, float]] = {}
    if page_ids:
        usage_rows = await db.execute(
            select(
                User.budget_id,
                func.count(),
                func.coalesce(func.sum(User.spend), _ZERO),
                func.coalesce(func.sum(User.reserved), _ZERO),
            )
            .where(User.budget_id.in_(page_ids), User.deleted_at.is_(None))
            .group_by(User.budget_id)
        )
        usage = {row[0]: (int(row[1]), float(row[2]), float(row[3])) for row in usage_rows}

    return [
        BudgetResponse.from_model(
            budget,
            user_count=usage.get(budget.budget_id, (0, 0.0, 0.0))[0],
            total_spend=usage.get(budget.budget_id, (0, 0.0, 0.0))[1],
            total_reserved=usage.get(budget.budget_id, (0, 0.0, 0.0))[2],
        )
        for budget in budgets
    ]


@router.get("/{budget_id}")
async def get_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Get details of a specific budget."""
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    user_count, total_spend, total_reserved = await _budget_usage(db, budget_id)
    return BudgetResponse.from_model(
        budget, user_count=user_count, total_spend=total_spend, total_reserved=total_reserved
    )


@router.patch("/{budget_id}")
async def update_budget(
    budget_id: str,
    request: UpdateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Update a budget."""
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    # Read before the mutation below, because it is what decides whether the
    # ceilings naming this budget have to be retimed.
    cadence_before = cadence_of(budget.budget_duration_sec, budget.reset_alignment)

    # Name is tri-state: omit leaves it unchanged, while an explicit null clears
    # it back to unnamed (unlike the numeric fields, where null is not meaningful).
    if "name" in request.model_fields_set:
        budget.name = request.name
    if request.max_budget is not None:
        budget.max_budget = to_usd(request.max_budget)
    if request.token_limit is not None:
        budget.token_limit = request.token_limit
    if request.request_limit is not None:
        budget.request_limit = request.request_limit
    # The two cadence fields settle together, because each is only legal in terms
    # of the other: the pair that has to hold is the one the row ends up with, so
    # an omitted field contributes what is stored. Switching a rolling budget to a
    # calendar one is one request that nulls the duration and names the alignment.
    if {"budget_duration_sec", "reset_alignment"} & request.model_fields_set:
        duration = (
            request.budget_duration_sec
            if "budget_duration_sec" in request.model_fields_set
            else budget.budget_duration_sec
        )
        alignment = (
            request.reset_alignment if "reset_alignment" in request.model_fields_set else budget.reset_alignment
        )
        _require_single_period_source(duration, alignment)
        budget.budget_duration_sec = duration
        budget.reset_alignment = alignment

    # A ceiling holds its own window and reads the cadence through this budget, so
    # changing the cadence without rewriting the windows leaves the two
    # disagreeing. In one direction that is an enforcement bug rather than a
    # cosmetic one: `_roll_expired_periods` only updates a row whose `period_end`
    # is not null, so a budget moved from "no reset" to a periodic cadence would
    # leave its ceilings with NULL windows that never roll, accumulating spend
    # forever. Since `b7e1c4a9d2f5` a budget can also belong to an organization
    # while this route still sees every one of them, so the ceilings stranded that
    # way may be a tenant's. Shared with the tenant-scoped surface rather than
    # written twice.
    if cadence_of(budget.budget_duration_sec, budget.reset_alignment) != cadence_before:
        await retime_ceilings_for_budget(
            db,
            budget_id=budget.budget_id,
            duration=budget.budget_duration_sec,
            alignment=budget.reset_alignment,
        )

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)

    user_count, total_spend, total_reserved = await _budget_usage(db, budget_id)
    return BudgetResponse.from_model(
        budget, user_count=user_count, total_spend=total_spend, total_reserved=total_reserved
    )


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a budget.

    Refused with 409 while anything still names this budget: a workspace handing
    it to its members, or a scoped ceiling enforcing it. Both foreign keys are
    ``RESTRICT``, so the database would refuse either anyway, but as an
    ``IntegrityError`` reported as "Database error" with nothing naming what to
    go and change. Checked here so the refusal can say which, and where.
    """
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    holders = (
        (
            await db.execute(
                select(col(Workspace.name))
                .join(WorkspaceBudgetDefault, WorkspaceBudgetDefault.workspace_id == col(Workspace.id))
                .where(WorkspaceBudgetDefault.budget_id == budget_id)
                .order_by(Workspace.name)
            )
        )
        .scalars()
        .all()
    )
    if holders:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This budget is the member default for "
                f"{', '.join(holders)}. Change or remove that default on the workspace "
                "(Organization > Workspaces > Edit) before deleting it."
            ),
        )

    # The same refusal for the ceilings themselves, which name a budget directly
    # and whose foreign key is RESTRICT too. Counted rather than named: a scope id
    # is a bare uuid, so listing them would say less than the number does.
    enforcing = (
        await db.execute(select(func.count()).select_from(ScopedBudget).where(ScopedBudget.budget_id == budget_id))
    ).scalar_one()
    if enforcing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"This budget is enforced by {enforcing} spend "
                f"{'ceiling' if enforcing == 1 else 'ceilings'}. A member's ceiling is "
                "changed on Members & roles (Edit > Workspace access); others are managed "
                "through /v1/scoped-budgets."
            ),
        )

    await db.delete(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None


@router.get("/{budget_id}/reset-logs")
async def list_budget_reset_logs(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetResetLogResponse]:
    """List per-user reset events for a budget, newest first."""
    budget = (
        await db.execute(select(Budget.budget_id).where(Budget.budget_id == budget_id))
    ).scalar_one_or_none()
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    result = await db.execute(
        select(BudgetResetLog)
        .where(BudgetResetLog.budget_id == budget_id)
        .order_by(BudgetResetLog.reset_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return [BudgetResetLogResponse.from_model(log) for log in result.scalars().all()]
