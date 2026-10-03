"""Manage the tenancy-scoped USD ceilings in ``scoped_budgets``.

The router is mounted in standalone and hosted modes, and not in hybrid mode.
It is operator-gated.
These routes create, list, retime and remove a ceiling, and they enforce none.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import BudgetServiceDep, get_db, require_deployment_operator
from gateway.core.database import DATABASE_ERRORS
from gateway.models.budgets import Budget, ScopedBudget, ScopeType
from gateway.schemas.budgets import CreateScopedBudgetRequest, ScopedBudgetResponse, UpdateScopedBudgetRequest
from gateway.services.budgets import period_window

# Auth is declared on the router, not repeated on each handler, following
# `routes/organizations.py`: every handler here needs the master key, and a
# future one that forgot the decorator would be unauthenticated with nothing
# to notice.
router = APIRouter(
    prefix="/scoped-budgets",
    tags=["scoped-budgets"],
    dependencies=[Depends(require_deployment_operator)],
)


async def _get_or_404(db: AsyncSession, budget_id: str) -> ScopedBudget:
    budget = (await db.execute(select(ScopedBudget).where(ScopedBudget.id == budget_id))).scalar_one_or_none()
    if budget is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scoped budget with id '{budget_id}' not found",
        )
    return budget


async def _require_budget(db: AsyncSession, budget_id: str) -> Budget:
    """The budget a ceiling names, refused as 404 when it does not exist.

    A ceiling naming nothing would cap nothing, in the permissive direction, so
    this refuses it before the write.
    """
    limit = await db.get(Budget, budget_id)
    if limit is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget '{budget_id}' not found",
        )
    return limit


@router.post("")
async def create_scoped_budget(
    request: CreateScopedBudgetRequest,
    budgets: BudgetServiceDep,
) -> ScopedBudgetResponse:
    """Create a scoped budget.

    Answers 404 when the scope names nothing, rather than creating a ceiling
    that can never bind.
    """
    try:
        return await budgets.create_deployment_ceiling(request=request)
    except DATABASE_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None


@router.get("")
async def list_scoped_budgets(
    db: Annotated[AsyncSession, Depends(get_db)],
    scope_type: Annotated[ScopeType | None, Query()] = None,
    scope_id: Annotated[str | None, Query(max_length=255)] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[ScopedBudgetResponse]:
    """List scoped budgets, optionally filtered to one scope."""
    stmt = select(ScopedBudget)
    if scope_type is not None:
        stmt = stmt.where(ScopedBudget.scope_type == scope_type)
    if scope_id is not None:
        stmt = stmt.where(ScopedBudget.scope_id == scope_id)
    # Joined rather than one lookup per row: the limit and period live on the
    # budget now, and a page of ceilings would otherwise be a page of round trips.
    result = await db.execute(
        stmt.join(Budget, Budget.budget_id == ScopedBudget.budget_id)
        .add_columns(Budget)
        .order_by(ScopedBudget.created_at)
        .offset(skip)
        .limit(limit)
    )
    return [ScopedBudgetResponse.from_model(ceiling, limit_row) for ceiling, limit_row in result.all()]


@router.get("/{budget_id}")
async def get_scoped_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScopedBudgetResponse:
    """Get one scoped budget."""
    ceiling = await _get_or_404(db, budget_id)
    limit = await db.get(Budget, ceiling.budget_id)
    if limit is None:  # pragma: no cover - RESTRICT keeps the budget alive
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget '{ceiling.budget_id}' not found",
        )
    return ScopedBudgetResponse.from_model(ceiling, limit)


@router.patch("/{budget_id}")
async def update_scoped_budget(
    budget_id: str,
    request: UpdateScopedBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScopedBudgetResponse:
    """Relabel a ceiling, or point it at a different budget.

    The scope and the provider narrowing are not editable: changing either would
    move the ceiling to a different identity while carrying its spend, which is
    a delete and a create, not an update.

    There is no limit or period to set here any more. Both are properties of the
    budget, so changing what a ceiling allows is either editing that budget,
    which moves every ceiling naming it, or naming a different one.
    """
    budget = await _get_or_404(db, budget_id)
    limit = await db.get(Budget, budget.budget_id)
    if limit is None:  # pragma: no cover - RESTRICT keeps the budget alive
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget '{budget.budget_id}' not found",
        )

    # Tri-state on ``name`` only, keyed on ``model_fields_set`` rather than on the
    # value: omitting it leaves it alone, and an explicit null clears it back to
    # unnamed, which is a state ``POST`` can create.
    if "name" in request.model_fields_set:
        budget.name = request.name
    if request.budget_id is not None and request.budget_id != budget.budget_id:
        limit = await _require_budget(db, request.budget_id)
        budget.budget_id = limit.budget_id
        # Retiming restarts the window from now rather than re-deriving an end
        # from a ``period_start`` belonging to the old budget's cadence. Spend
        # already recorded stays: the ceiling is the same allowance, held to a
        # different figure from here on.
        window = period_window(
            datetime.now(UTC),
            duration=limit.budget_duration_sec,
            alignment=limit.reset_alignment,
        )
        budget.period_start, budget.period_end = window if window is not None else (None, None)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)
    return ScopedBudgetResponse.from_model(budget, limit)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scoped_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a scoped budget.

    A request holding a reservation against it settles into nothing afterwards,
    which is the right outcome: the ceiling no longer exists to be credited.
    """
    budget = await _get_or_404(db, budget_id)
    await db.delete(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
