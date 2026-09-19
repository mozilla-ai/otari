"""Request and response models of the budgets domain.

Requests type the closed vocabularies.
Responses echo the stored string, so a row that holds an unknown value still reads.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from gateway.models.budgets import MAX_COUNT_LIMIT, Budget, BudgetResetLog, ResetAlignment, ScopedBudget, ScopeType
from gateway.models.money import MAX_USD_LIMIT, as_float


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
    # None is the deployment's own budget, and a value is the organization that owns it.
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
        budget: Budget,
        *,
        user_count: int = 0,
        total_spend: float = 0.0,
        total_reserved: float = 0.0,
    ) -> BudgetResponse:
        """Create a BudgetResponse from a Budget ORM model and its usage rollup."""
        return cls(
            budget_id=budget.budget_id,
            organization_id=budget.organization_id,
            name=budget.name,
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
    def from_model(cls, log: BudgetResetLog) -> BudgetResetLogResponse:
        return cls(
            id=log.id,
            user_id=log.user_id,
            budget_id=log.budget_id,
            previous_spend=float(log.previous_spend),
            reset_at=log.reset_at.isoformat(),
            next_reset_at=log.next_reset_at.isoformat() if log.next_reset_at else None,
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
        min_length=1,
        max_length=255,
        pattern=r"^\S+$",
        description=(
            "Narrow the cap to one provider instance; omit or null to cap spend across every provider. "
            "A blank value would store a ceiling that never binds, so it is refused; this does not check "
            "that the value names a configured provider instance"
        ),
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

    Unlike ``/api/v1/budgets``, the counters are the row's own: a scoped ceiling is
    enforced against ``current_spend + reserved_spend``, so there is no rollup
    over users to compute.

    Every limit, along with ``budget_duration_sec`` and ``reset_alignment``, is
    read off the budget rather than stored here, and carried on the wire so a
    caller can render a ceiling without fetching every budget to resolve one id.
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
    token_limit: int | None
    current_tokens: int
    reserved_tokens: int
    request_limit: int | None
    current_requests: int
    reserved_requests: int
    budget_duration_sec: int | None
    reset_alignment: str | None
    period_start: str | None
    period_end: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, budget: ScopedBudget, limit: Budget) -> ScopedBudgetResponse:
        """Create a ScopedBudgetResponse from a ceiling and the budget it names."""
        return cls(
            id=budget.id,
            scope_type=budget.scope_type,
            scope_id=budget.scope_id,
            provider_key_id=budget.provider_key_id,
            budget_id=budget.budget_id,
            name=budget.name,
            max_budget=as_float(limit.max_budget),
            current_spend=float(budget.current_spend),
            reserved_spend=float(budget.reserved_spend),
            token_limit=limit.token_limit,
            current_tokens=budget.current_tokens,
            reserved_tokens=budget.reserved_tokens,
            request_limit=limit.request_limit,
            current_requests=budget.current_requests,
            reserved_requests=budget.reserved_requests,
            budget_duration_sec=limit.budget_duration_sec,
            reset_alignment=limit.reset_alignment,
            period_start=budget.period_start.isoformat() if budget.period_start else None,
            period_end=budget.period_end.isoformat() if budget.period_end else None,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


__all__ = [
    "BudgetResetLogResponse",
    "BudgetResponse",
    "CreateBudgetRequest",
    "CreateScopedBudgetRequest",
    "ScopedBudgetResponse",
    "UpdateBudgetRequest",
    "UpdateScopedBudgetRequest",
]
