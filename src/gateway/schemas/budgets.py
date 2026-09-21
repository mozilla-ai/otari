"""Request and response models of the budgets domain.

Requests type the closed vocabularies.
Responses echo the stored string, so a row that holds an unknown value still reads.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field

from gateway.models.budgets import (
    MAX_COUNT_LIMIT,
    Budget,
    BudgetResetLog,
    ResetAlignment,
    ScopedBudget,
    ScopeType,
    WorkspaceBudgetDefault,
)
from gateway.models.money import MAX_USD_LIMIT, as_float

_PERIOD_DESCRIPTION = (
    "Seconds between resets, counted from the last one. Mutually exclusive with reset_alignment"
)
_ALIGNMENT_DESCRIPTION = (
    "Reset on a UTC calendar boundary instead of a fixed number of seconds, which is the only way "
    "to express a calendar month. Mutually exclusive with budget_duration_sec"
)

# A blank value matches no provider instance, so a ceiling that stored one would never bind.
_ProviderKeyId = Annotated[str, Field(min_length=1, max_length=255, pattern=r"^\S+$")]


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
        description=_ALIGNMENT_DESCRIPTION,
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
    provider_key_id: _ProviderKeyId | None = Field(
        default=None,
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


class OrganizationBudgetRates(BaseModel):
    """The figures and the period a budget holds, shared by the create and update bodies."""

    name: str | None = Field(default=None, max_length=200, description="Admin-facing label for the budget")
    max_budget: float | None = Field(
        default=None,
        ge=0,
        le=MAX_USD_LIMIT,
        description="Maximum spend in USD over one period; null caps nothing",
    )
    token_limit: int | None = Field(
        default=None,
        ge=0,
        le=MAX_COUNT_LIMIT,
        description="Maximum tokens over one period; null caps nothing. Independent of max_budget",
    )
    request_limit: int | None = Field(
        default=None,
        ge=0,
        le=MAX_COUNT_LIMIT,
        description="Maximum requests over one period; null caps nothing. Independent of max_budget",
    )
    budget_duration_sec: int | None = Field(default=None, gt=0, description=_PERIOD_DESCRIPTION)
    reset_alignment: ResetAlignment | None = Field(default=None, description=_ALIGNMENT_DESCRIPTION)


class OrganizationBudgetCreate(OrganizationBudgetRates):
    """Create one budget owned by the caller's organization."""


class OrganizationBudgetUpdate(OrganizationBudgetRates):
    """Replace a budget's label, figure and period.

    Every field is optional and keyed on ``model_fields_set``, matching
    the deployment-wide budget update's own: an *omitted* field is left alone, and an
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
    token_limit: int | None
    request_limit: int | None
    budget_duration_sec: int | None
    reset_alignment: str | None
    ceiling_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_model(
        cls,
        budget: Budget,
        *,
        organization_id: uuid.UUID,
        ceiling_count: int,
    ) -> OrganizationBudgetPublic:
        """Build the public form of a budget that the given organization owns.

        Raises:
            ValueError: The budget belongs to another owner, or to the deployment.
        """
        if budget.organization_id != organization_id:
            raise ValueError("The budget does not belong to this organization")
        return cls(
            budget_id=budget.budget_id,
            organization_id=organization_id,
            name=budget.name,
            max_budget=as_float(budget.max_budget),
            token_limit=budget.token_limit,
            request_limit=budget.request_limit,
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

    scope_type: ScopeType = Field(description="Which kind of identity this ceiling caps")
    scope_id: str = Field(
        min_length=1,
        max_length=255,
        description=(
            "Id of the capped identity: this organization, one of its workspaces, "
            "a membership in either, or an API key in one"
        ),
    )
    provider_key_id: _ProviderKeyId | None = Field(
        default=None,
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
    the deployment-wide ceiling update gives: changing either moves the ceiling to
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
    # This is False when the ceiling's budget belongs to another owner, so its figure cannot change here.
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
            token_limit=budget.token_limit,
            current_tokens=ceiling.current_tokens,
            reserved_tokens=ceiling.reserved_tokens,
            request_limit=budget.request_limit,
            current_requests=ceiling.current_requests,
            reserved_requests=ceiling.reserved_requests,
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


class WorkspaceMemberBudgetPolicyCreate(BaseModel):
    """Request body for creating a default."""

    budget_id: str = Field(
        min_length=1,
        max_length=255,
        description="The budget this workspace hands to every member",
    )
    provider_key_id: _ProviderKeyId | None = Field(
        default=None,
        description=(
            "Narrow the default to one provider instance; omit or null to apply to every provider. "
            "Must name a real instance: a blank value would materialize ceilings that never bind"
        ),
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
    name: str | None
    max_budget: float | None
    token_limit: int | None
    request_limit: int | None
    budget_duration_sec: int | None
    reset_alignment: str | None
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
            max_budget=as_float(budget.max_budget),
            token_limit=budget.token_limit,
            request_limit=budget.request_limit,
            budget_duration_sec=budget.budget_duration_sec,
            reset_alignment=budget.reset_alignment,
            created_at=default.created_at.isoformat(),
            updated_at=default.updated_at.isoformat(),
        )


class WorkspaceMemberBudgetPoliciesPublic(BaseModel):
    data: list[WorkspaceMemberBudgetPolicyPublic]
    count: int


__all__ = [
    "BudgetResetLogResponse",
    "BudgetResponse",
    "CreateBudgetRequest",
    "CreateScopedBudgetRequest",
    "OrganizationBudgetCreate",
    "OrganizationBudgetPublic",
    "OrganizationBudgetRates",
    "OrganizationBudgetUpdate",
    "OrganizationBudgetsPublic",
    "OrganizationScopedBudgetCreate",
    "OrganizationScopedBudgetPublic",
    "OrganizationScopedBudgetUpdate",
    "OrganizationScopedBudgetsPublic",
    "ScopedBudgetResponse",
    "UpdateBudgetRequest",
    "UpdateScopedBudgetRequest",
    "WorkspaceMemberBudgetPoliciesPublic",
    "WorkspaceMemberBudgetPolicyCreate",
    "WorkspaceMemberBudgetPolicyPublic",
    "WorkspaceMemberBudgetPolicyUpdate",
]
