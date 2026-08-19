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

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import APIKey, ScopedBudget
from gateway.models.tenancy import Organization, OrganizationMember, Workspace, WorkspaceMember

# ``ScopeType`` comes from the service that resolves a scope, not from a copy
# here: the ``Literal`` is what puts the allowed values in the OpenAPI schema,
# and a second roster would eventually let a client create a scope enforcement
# does not know.
from gateway.services.scoped_budget_service import ResetAlignment, ScopeType, period_window

# Auth is declared on the router, not repeated on each handler, following
# `routes/organizations.py`: every handler here needs the master key, and a
# future one that forgot the decorator would be unauthenticated with nothing
# to notice.
router = APIRouter(
    prefix="/v1/scoped-budgets",
    tags=["scoped-budgets"],
    dependencies=[Depends(verify_master_key)],
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
    name: str | None = Field(default=None, max_length=200, description="Admin-facing label for the budget")
    max_budget: float | None = Field(default=None, ge=0, description="Maximum USD spend in the period")
    budget_duration_sec: int | None = Field(
        default=None, gt=0, description="Period length in seconds (e.g. 86400 for daily); null never resets"
    )
    reset_alignment: ResetAlignment | None = Field(
        default=None,
        description=(
            "Reset on a UTC calendar boundary instead of a fixed number of seconds, "
            "which is the only way to express a calendar month. Mutually exclusive with budget_duration_sec"
        ),
    )


class UpdateScopedBudgetRequest(BaseModel):
    """Request model for updating a scoped budget."""

    name: str | None = Field(default=None, max_length=200)
    max_budget: float | None = Field(default=None, ge=0)
    budget_duration_sec: int | None = Field(default=None, gt=0)
    reset_alignment: ResetAlignment | None = Field(default=None)


class ScopedBudgetResponse(BaseModel):
    """One scoped ceiling and its live counters.

    Unlike ``/v1/budgets``, the counters are the row's own: a scoped ceiling is
    enforced against ``current_spend + reserved_spend``, so there is no rollup
    over users to compute.
    """

    id: str
    scope_type: str
    scope_id: str
    provider_key_id: str | None
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
    def from_model(cls, budget: ScopedBudget) -> "ScopedBudgetResponse":
        """Create a ScopedBudgetResponse from a ScopedBudget ORM model."""
        return cls(
            id=budget.id,
            scope_type=budget.scope_type,
            scope_id=budget.scope_id,
            provider_key_id=budget.provider_key_id,
            name=budget.name,
            max_budget=budget.max_budget,
            current_spend=budget.current_spend,
            reserved_spend=budget.reserved_spend,
            budget_duration_sec=budget.budget_duration_sec,
            reset_alignment=budget.reset_alignment,
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


def _require_single_period_source(duration: int | None, alignment: str | None) -> None:
    """Refuse the state the table's CHECK refuses, with a message instead of a 500.

    A period comes from a duration or from a calendar boundary. Both set is one
    concept encoded twice, so the pair would need an "ignored when" rule to mean
    anything.
    """
    if duration is not None and alignment is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A budget resets on budget_duration_sec or on reset_alignment, not both",
        )


@router.post("")
async def create_scoped_budget(
    request: CreateScopedBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScopedBudgetResponse:
    """Create a scoped budget.

    Answers 404 when the scope names nothing, rather than creating a ceiling
    that can never bind.
    """
    _require_single_period_source(request.budget_duration_sec, request.reset_alignment)
    await _require_scope_exists(db, request.scope_type, request.scope_id)
    # The window opens now rather than on first spend, so a period-limited ceiling
    # has a defined end before any request has arrived. An aligned one opens on the
    # boundary it is already past, so its first period is the remainder of the
    # calendar period it was created in rather than a full one starting now.
    window = period_window(
        datetime.now(UTC),
        duration=request.budget_duration_sec,
        alignment=request.reset_alignment,
    )
    period_start, period_end = window if window is not None else (None, None)
    budget = ScopedBudget(
        scope_type=request.scope_type,
        scope_id=request.scope_id,
        provider_key_id=request.provider_key_id,
        name=request.name,
        max_budget=request.max_budget,
        budget_duration_sec=request.budget_duration_sec,
        reset_alignment=request.reset_alignment,
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
    return ScopedBudgetResponse.from_model(budget)


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
    result = await db.execute(stmt.order_by(ScopedBudget.created_at).offset(skip).limit(limit))
    return [ScopedBudgetResponse.from_model(budget) for budget in result.scalars().all()]


@router.get("/{budget_id}")
async def get_scoped_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScopedBudgetResponse:
    """Get one scoped budget."""
    return ScopedBudgetResponse.from_model(await _get_or_404(db, budget_id))


@router.patch("/{budget_id}")
async def update_scoped_budget(
    budget_id: str,
    request: UpdateScopedBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScopedBudgetResponse:
    """Update a scoped budget's label, limit, or period.

    The scope and the provider narrowing are not editable: changing either would
    move the ceiling to a different identity while carrying its spend, which is
    a delete and a create, not an update.
    """
    budget = await _get_or_404(db, budget_id)

    # Every field is tri-state, keyed on ``model_fields_set`` rather than on the
    # value: omitting one leaves it alone, and an explicit null clears it. Null
    # is meaningful on every one of them, because it is a state ``POST`` can
    # create. A null ``max_budget`` is a ceiling that admits everything and only
    # meters, and a cap with neither ``budget_duration_sec`` nor
    # ``reset_alignment`` never resets; testing ``is not None`` would have made
    # both reachable at creation and permanent afterwards, so a limit could be
    # tightened but never relaxed.
    if "name" in request.model_fields_set:
        budget.name = request.name
    if "max_budget" in request.model_fields_set:
        budget.max_budget = request.max_budget
    # The two cadence fields are settled together, because each is only legal in
    # terms of the other: the pair that has to hold is the one the row ends up
    # with, so an omitted field contributes what is stored. Switching a rolling
    # ceiling to a calendar one is therefore one request that nulls the duration
    # and names the alignment, and naming only the alignment is refused rather
    # than silently clearing a duration the caller did not mention.
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
        # Retiming restarts the window from now rather than re-deriving an end
        # from a period_start that belongs to the old cadence. Clearing the
        # cadence drops the window with it, matching what ``POST`` writes for a
        # budget created without one. An alignment lands on its boundary, so
        # retiming to one is not a way to shift where the boundary falls.
        window = period_window(datetime.now(UTC), duration=duration, alignment=alignment)
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
    return ScopedBudgetResponse.from_model(budget)


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
