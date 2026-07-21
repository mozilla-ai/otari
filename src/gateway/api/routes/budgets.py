from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import Budget, BudgetResetLog, User

router = APIRouter(prefix="/v1/budgets", tags=["budgets"])


class CreateBudgetRequest(BaseModel):
    """Request model for creating a new budget."""

    name: str | None = Field(default=None, description="Admin-facing label for the budget")
    max_budget: float | None = Field(default=None, ge=0, description="Maximum spending limit")
    budget_duration_sec: int | None = Field(
        default=None, gt=0, description="Budget duration in seconds (e.g., 86400 for daily, 604800 for weekly)"
    )


class BudgetResponse(BaseModel):
    """Response model for budget information.

    ``max_budget`` is the per-user spending limit, and multiple users can share
    one budget, so the usage rollup is an aggregate over the users assigned to
    this budget: how many there are and their combined ``spend`` / ``reserved``.
    Assigning users to a budget is done through the users API (dashboard support
    lands with user management), so a fresh gateway reports zeros here.
    """

    budget_id: str
    name: str | None
    max_budget: float | None
    budget_duration_sec: int | None
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
            name=budget.name,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
            user_count=user_count,
            total_spend=total_spend,
            total_reserved=total_reserved,
        )


class UpdateBudgetRequest(BaseModel):
    """Request model for updating a budget."""

    name: str | None = Field(default=None)
    max_budget: float | None = Field(default=None, ge=0)
    budget_duration_sec: int | None = Field(default=None, gt=0)


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
                func.coalesce(func.sum(User.spend), 0.0),
                func.coalesce(func.sum(User.reserved), 0.0),
            ).where(User.budget_id == budget_id, User.deleted_at.is_(None))
        )
    ).one()
    return int(row[0]), float(row[1]), float(row[2])


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_budget(
    request: CreateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Create a new budget."""
    budget = Budget(
        name=request.name,
        max_budget=request.max_budget,
        budget_duration_sec=request.budget_duration_sec,
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


@router.get("", dependencies=[Depends(verify_master_key)])
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
                func.coalesce(func.sum(User.spend), 0.0),
                func.coalesce(func.sum(User.reserved), 0.0),
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


@router.get("/{budget_id}", dependencies=[Depends(verify_master_key)])
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


@router.patch("/{budget_id}", dependencies=[Depends(verify_master_key)])
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

    # Name is tri-state: omit leaves it unchanged, while an explicit null clears
    # it back to unnamed (unlike the numeric fields, where null is not meaningful).
    if "name" in request.model_fields_set:
        budget.name = request.name
    if request.max_budget is not None:
        budget.max_budget = request.max_budget
    if request.budget_duration_sec is not None:
        budget.budget_duration_sec = request.budget_duration_sec

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


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a budget."""
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
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


@router.get("/{budget_id}/reset-logs", dependencies=[Depends(verify_master_key)])
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
