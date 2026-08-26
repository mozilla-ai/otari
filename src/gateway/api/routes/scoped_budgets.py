"""Manage the tenancy-scoped USD ceilings in ``scoped_budgets``.

Standalone-mode only, and master-key authed on the router itself.
Deliberately minimal: a scope's ceiling is created, listed, retimed and removed
here, and everything about how one is enforced lives in
``services/scoped_budget_service.py``.
"""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, require_deployment_operator
from gateway.models.entities import APIKey, Budget, ScopedBudget
from gateway.models.money import as_float
from gateway.models.tenancy import Organization, OrganizationMember, Workspace, WorkspaceMember

# ``ScopeType`` comes from the service that resolves a scope, not from a copy
# here: the ``Literal`` is what puts the allowed values in the OpenAPI schema,
# and a second roster would eventually let a client create a scope enforcement
# does not know.
from gateway.services.scoped_budget_service import ScopeType, period_window

# Auth is declared on the router, not repeated on each handler, following
# `routes/organizations.py`: every handler here needs the master key, and a
# future one that forgot the decorator would be unauthenticated with nothing
# to notice.
router = APIRouter(
    prefix="/v1/scoped-budgets",
    tags=["scoped-budgets"],
    dependencies=[Depends(require_deployment_operator)],
)


class CreateScopedBudgetRequest(BaseModel):
    """Request model for creating a scoped budget."""

    scope_type: ScopeType = Field(description="Which kind of identity this ceiling caps")
    scope_id: str = Field(
        min_length=1,
        max_length=255,
        description="Id of the capped identity: an organization, workspace, membership row, or API key",
    )
    provider_key_id: str | None = Field(
        default=None,
        max_length=255,
        description="Narrow the cap to one provider instance; null caps spend across every provider",
    )
    budget_id: str = Field(
        min_length=1,
        max_length=255,
        description="The budget this ceiling enforces; its limit and period are read through it",
    )
    name: str | None = Field(default=None, max_length=200, description="Admin-facing label for this ceiling")


class UpdateScopedBudgetRequest(BaseModel):
    """Request model for updating a scoped budget."""

    budget_id: str | None = Field(default=None, min_length=1, max_length=255)
    name: str | None = Field(default=None, max_length=200)


class ScopedBudgetResponse(BaseModel):
    """One scoped ceiling and its live counters.

    Unlike ``/v1/budgets``, the counters are the row's own: a scoped ceiling is
    enforced against ``current_spend + reserved_spend``, so there is no rollup
    over users to compute.

    ``max_budget``, ``budget_duration_sec`` and ``reset_alignment`` are read off
    the budget rather than stored here, and are carried on the wire so a caller
    can render a ceiling without fetching every budget to resolve one id.
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
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, budget: ScopedBudget, limit: Budget) -> "ScopedBudgetResponse":
        """Create a ScopedBudgetResponse from a ceiling and the budget it names."""
        return cls(
            id=budget.id,
            scope_type=budget.scope_type,
            scope_id=budget.scope_id,
            provider_key_id=budget.provider_key_id,
            budget_id=budget.budget_id,
            name=budget.name,
            # Narrowed on the way out: the cap and the counters are exact in the
            # database, while the wire contract and the dashboard client stay float.
            max_budget=as_float(limit.max_budget),
            current_spend=float(budget.current_spend),
            reserved_spend=float(budget.reserved_spend),
            budget_duration_sec=limit.budget_duration_sec,
            reset_alignment=limit.reset_alignment,
            period_start=budget.period_start.isoformat() if budget.period_start else None,
            period_end=budget.period_end.isoformat() if budget.period_end else None,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


async def _get_or_404(db: AsyncSession, budget_id: str) -> ScopedBudget:
    budget = (await db.execute(select(ScopedBudget).where(ScopedBudget.id == budget_id))).scalar_one_or_none()
    if budget is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scoped budget with id '{budget_id}' not found",
        )
    return budget


# What a scope's id has to name for the ceiling to ever bind, by scope type.
# `api_token` keys on a string id, the rest on UUIDs.
_ScopeSubject = type[Organization | Workspace | OrganizationMember | WorkspaceMember | APIKey]
_SCOPE_SUBJECTS: dict[str, tuple[_ScopeSubject, str]] = {
    "organization": (Organization, "Organization"),
    "workspace": (Workspace, "Workspace"),
    "workspace_member": (WorkspaceMember, "Workspace membership"),
    "org_member": (OrganizationMember, "Organization membership"),
    "api_token": (APIKey, "API key"),
}


async def _require_scope_exists(db: AsyncSession, scope_type: str, scope_id: str) -> None:
    """Refuse a ceiling on a scope that does not exist.

    Without this a typo answers 200 and then never binds: resolution matches on
    the id, so a ceiling naming nothing is created, listed, and silently
    unenforced, with nothing anywhere to surface it. That is the shape a bulk
    import produces from a mis-mapped id, and it fails in the permissive
    direction. `POST /v1/keys` already refuses an unknown workspace for the same
    reason; this matches it.
    """
    model, subject = _SCOPE_SUBJECTS[scope_type]
    identifier: object = scope_id
    if model is not APIKey:
        try:
            identifier = uuid.UUID(scope_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{subject} '{scope_id}' not found",
            ) from None

    found = await db.get(model, identifier)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{subject} '{scope_id}' not found",
        )


async def _require_budget(db: AsyncSession, budget_id: str) -> Budget:
    """The budget a ceiling names, refused as 404 when it does not exist.

    A ceiling naming nothing would cap nothing, in the permissive direction, so
    this refuses for the same reason `_require_scope_exists` does.
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
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScopedBudgetResponse:
    """Create a scoped budget.

    Answers 404 when the scope names nothing, rather than creating a ceiling
    that can never bind.
    """
    await _require_scope_exists(db, request.scope_type, request.scope_id)
    limit = await _require_budget(db, request.budget_id)
    # The window opens now rather than on first spend, so a period-limited ceiling
    # has a defined end before any request has arrived. An aligned one opens on the
    # boundary it is already past, so its first period is the remainder of the
    # calendar period it was created in rather than a full one starting now. The
    # cadence is the budget's, which is the whole point of naming one.
    window = period_window(
        datetime.now(UTC),
        duration=limit.budget_duration_sec,
        alignment=limit.reset_alignment,
    )
    period_start, period_end = window if window is not None else (None, None)
    budget = ScopedBudget(
        scope_type=request.scope_type,
        scope_id=request.scope_id,
        provider_key_id=request.provider_key_id,
        budget_id=limit.budget_id,
        name=request.name,
        period_start=period_start,
        period_end=period_end,
    )
    db.add(budget)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A budget already exists for this scope and provider",
        ) from None
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)
    return ScopedBudgetResponse.from_model(budget, limit)


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
