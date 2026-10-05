from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import get_db, require_deployment_operator
from gateway.core.surface import Surface
from gateway.models.budgets import Budget, BudgetResetLog, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.money import to_usd, to_usd_or_none
from gateway.models.tenancy import Workspace
from gateway.models.users import User
from gateway.schemas.budgets import (
    BudgetResetLogResponse,
    BudgetResponse,
    CreateBudgetRequest,
    UpdateBudgetRequest,
)
from gateway.services.budgets import (
    CYCLE_FIELD_ORDER,
    CYCLE_FIELDS,
    CycleSettings,
    cadence_of,
    retime_ceilings_for_budget,
    validate_cycle_settings,
)

router = APIRouter(
    prefix="/budgets",
    tags=["budgets"],
    dependencies=[Depends(require_deployment_operator)],
)

SURFACE = Surface("budgets")

# The rollup below sums exact counters, so its coalesce default is exact too.
_ZERO = Decimal(0)


def _requested_cycle(request: object) -> CycleSettings:
    """The cadence a create request names, as the tuple the validator reads."""
    return CycleSettings(*(getattr(request, name) for name in CYCLE_FIELD_ORDER))


def _require_valid_cycle(settings: CycleSettings) -> None:
    """Refuse the state the table's CHECKs refuse, with a message instead of a 500.

    Each cycle carries exactly its own settings. Validating here is what names the
    offending field; letting the write reach the constraints answers an
    IntegrityError the caller cannot act on.
    """
    try:
        validate_cycle_settings(settings)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


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
    _require_valid_cycle(_requested_cycle(request))
    budget = Budget(
        name=request.name,
        max_budget=to_usd_or_none(request.max_budget),
        token_limit=request.token_limit,
        request_limit=request.request_limit,
        reset_cycle=request.reset_cycle,
        reset_every_n=request.reset_every_n,
        reset_anchor_at=request.reset_anchor_at,
        reset_weekdays=request.reset_weekdays,
        reset_month_day=request.reset_month_day,
        reset_month=request.reset_month,
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
    cadence_before = cadence_of(budget)

    # Name is tri-state: omit leaves it unchanged, while an explicit null clears
    # it back to unnamed (unlike the numeric fields, where null is not meaningful).
    if "name" in request.model_fields_set:
        budget.name = request.name
    if request.max_budget is not None:
        budget.max_budget = to_usd(request.max_budget)
    # Tri-state like the name, and unlike ``max_budget`` above: null is the value
    # that means "caps nothing on this axis", so a cap that could be raised and
    # lowered but never taken back off would be a cap with no way out. The
    # organization-scoped service settles the same two fields the same way.
    if "token_limit" in request.model_fields_set:
        budget.token_limit = request.token_limit
    if "request_limit" in request.model_fields_set:
        budget.request_limit = request.request_limit
    # The cadence fields settle together, because each is only legal in terms of
    # the others: the set that has to hold is the one the row ends up with, so an
    # omitted field contributes what is stored. Switching a weekly budget to a
    # monthly one is one request that names the cycle and the day, and the
    # weekday mask it leaves behind is what `validate_cycle_settings` refuses, so
    # a caller has to clear it rather than strand it.
    if CYCLE_FIELDS & request.model_fields_set:
        settled = CycleSettings(
            *(
                getattr(request, name) if name in request.model_fields_set else getattr(budget, name)
                for name in CYCLE_FIELD_ORDER
            )
        )
        _require_valid_cycle(settled)
        for name, value in zip(CYCLE_FIELD_ORDER, settled, strict=True):
            setattr(budget, name, value)

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
    if cadence_of(budget) != cadence_before:
        await retime_ceilings_for_budget(db, budget, budget_id=budget.budget_id)

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
                "through /api/v1/scoped-budgets."
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
    budget = (await db.execute(select(Budget.budget_id).where(Budget.budget_id == budget_id))).scalar_one_or_none()
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
