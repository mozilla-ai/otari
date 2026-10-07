"""Request and response models of the budgets domain.

Requests type the closed vocabularies.
Responses echo the stored string, so a row that holds an unknown value still reads.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field, model_validator

from gateway.models.budgets import (
    MAX_COUNT_LIMIT,
    MAX_MONTH_DAY,
    WEEKDAY_MASK_MAX,
    WEEKDAY_MASK_MIN,
    Budget,
    BudgetResetLog,
    ResetCycle,
    ScopedBudget,
    ScopeType,
    WorkspaceBudgetDefault,
)
from gateway.models.money import MAX_USD_LIMIT, as_float

_CYCLE_DESCRIPTION = (
    "How often the budget resets, at 00:00 UTC. Null never resets. Each cycle carries its own "
    "settings and no others: every_n_hours and every_n_days take reset_every_n and "
    "reset_anchor_at, weekly takes reset_weekdays, monthly takes reset_month_day, and yearly "
    "takes reset_month and reset_month_day"
)
_EVERY_N_DESCRIPTION = "How many hours or days between resets, in the unit reset_cycle names"
_ANCHOR_DESCRIPTION = (
    "Where an interval cycle's first period opened. Later periods are counted from it, so the "
    "cadence keeps its phase rather than restarting at the next request"
)
_WEEKDAYS_DESCRIPTION = (
    "The weekdays a weekly cycle resets on, as a bitmask with bit 0 Monday through bit 6 Sunday. "
    "Several are allowed and the limit applies to each period between them, so Monday and Friday "
    "give a four-day period and a three-day one"
)
_MONTH_DAY_DESCRIPTION = (
    f"The day of the month a monthly or yearly cycle resets on, 1 to {MAX_MONTH_DAY}. Capped so every month has the day"
)
_MONTH_DESCRIPTION = "The month a yearly cycle resets in, 1 to 12"


# A blank value matches no provider instance, so a ceiling that stored one would never bind.
_ProviderKeyId = Annotated[str, Field(min_length=1, max_length=255, pattern=r"^\S+$")]


class _ModelNarrowing(BaseModel):
    """The resource axes of a ceiling's create body: a provider, and optionally one of its models."""

    provider_key_id: _ProviderKeyId | None = None
    model: _ProviderKeyId | None = Field(
        default=None,
        description=(
            "Narrow the cap to one model of the provider, by the id the provider gives it; omit or null to cap "
            "every model. Requires provider_key_id, because a model id is only unique within its provider"
        ),
    )

    @model_validator(mode="after")
    def _model_needs_provider(self) -> Self:
        if self.model is not None and self.provider_key_id is None:
            raise ValueError("model requires provider_key_id")
        return self


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
    reset_cycle: ResetCycle | None = Field(default=None, description=_CYCLE_DESCRIPTION)
    reset_every_n: int | None = Field(default=None, gt=0, description=_EVERY_N_DESCRIPTION)
    reset_anchor_at: datetime | None = Field(default=None, description=_ANCHOR_DESCRIPTION)
    reset_weekdays: int | None = Field(
        default=None, ge=WEEKDAY_MASK_MIN, le=WEEKDAY_MASK_MAX, description=_WEEKDAYS_DESCRIPTION
    )
    reset_month_day: int | None = Field(default=None, ge=1, le=MAX_MONTH_DAY, description=_MONTH_DAY_DESCRIPTION)
    reset_month: int | None = Field(default=None, ge=1, le=12, description=_MONTH_DESCRIPTION)


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
    reset_cycle: str | None
    reset_every_n: int | None
    reset_anchor_at: datetime | None
    reset_weekdays: int | None
    reset_month_day: int | None
    reset_month: int | None
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
            reset_cycle=budget.reset_cycle,
            reset_every_n=budget.reset_every_n,
            reset_anchor_at=budget.reset_anchor_at,
            reset_weekdays=budget.reset_weekdays,
            reset_month_day=budget.reset_month_day,
            reset_month=budget.reset_month,
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
    reset_cycle: ResetCycle | None = Field(default=None, description=_CYCLE_DESCRIPTION)
    reset_every_n: int | None = Field(default=None, gt=0, description=_EVERY_N_DESCRIPTION)
    reset_anchor_at: datetime | None = Field(default=None, description=_ANCHOR_DESCRIPTION)
    reset_weekdays: int | None = Field(
        default=None, ge=WEEKDAY_MASK_MIN, le=WEEKDAY_MASK_MAX, description=_WEEKDAYS_DESCRIPTION
    )
    reset_month_day: int | None = Field(default=None, ge=1, le=MAX_MONTH_DAY, description=_MONTH_DAY_DESCRIPTION)
    reset_month: int | None = Field(default=None, ge=1, le=12, description=_MONTH_DESCRIPTION)


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


class CreateScopedBudgetRequest(_ModelNarrowing):
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


class ScopedBudgetChanges(BaseModel):
    """The two editable fields of a scoped ceiling: the budget it enforces and its label."""

    budget_id: str | None = Field(default=None, min_length=1, max_length=255)
    name: str | None = Field(default=None, max_length=200)


class UpdateScopedBudgetRequest(ScopedBudgetChanges):
    """Request model for updating a scoped budget."""


class ScopedBudgetFigures(BaseModel):
    """One scoped ceiling: its identity, its live counters, and the limits and period of its budget."""

    id: str
    scope_type: str
    scope_id: str
    provider_key_id: str | None
    model: str | None
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
    reset_cycle: str | None
    reset_every_n: int | None
    reset_anchor_at: datetime | None
    reset_weekdays: int | None
    reset_month_day: int | None
    reset_month: int | None
    period_start: str | None
    period_end: str | None
    created_at: str
    updated_at: str

    @staticmethod
    def _figures_of(ceiling: ScopedBudget, budget: Budget) -> dict[str, Any]:
        """Return the value of every field here, read from a ceiling and the budget it names."""
        ended = ceiling.period_has_ended
        return {
            "id": ceiling.id,
            "scope_type": ceiling.scope_type,
            "scope_id": ceiling.scope_id,
            "provider_key_id": ceiling.provider_key_id,
            "model": ceiling.model,
            "budget_id": ceiling.budget_id,
            "name": ceiling.name,
            "max_budget": as_float(budget.max_budget),
            # Last period's figures until the next request rolls the window.
            "current_spend": 0.0 if ended else float(ceiling.current_spend),
            "reserved_spend": float(ceiling.reserved_spend),
            "token_limit": budget.token_limit,
            "current_tokens": 0 if ended else ceiling.current_tokens,
            "reserved_tokens": ceiling.reserved_tokens,
            "request_limit": budget.request_limit,
            "current_requests": 0 if ended else ceiling.current_requests,
            "reserved_requests": ceiling.reserved_requests,
            "reset_cycle": budget.reset_cycle,
            "reset_every_n": budget.reset_every_n,
            "reset_anchor_at": budget.reset_anchor_at,
            "reset_weekdays": budget.reset_weekdays,
            "reset_month_day": budget.reset_month_day,
            "reset_month": budget.reset_month,
            "period_start": ceiling.period_start.isoformat() if ceiling.period_start else None,
            "period_end": ceiling.period_end.isoformat() if ceiling.period_end else None,
            "created_at": ceiling.created_at.isoformat(),
            "updated_at": ceiling.updated_at.isoformat(),
        }


class ScopedBudgetResponse(ScopedBudgetFigures):
    """One scoped ceiling and its live counters.

    Unlike ``/api/v1/budgets``, the counters are the row's own: a scoped ceiling is
    enforced against ``current_spend + reserved_spend``, so there is no rollup
    over users to compute.

    Every limit, along with the reset cycle and its settings, is
    read off the budget rather than stored here, and carried on the wire so a
    caller can render a ceiling without fetching every budget to resolve one id.
    """

    @classmethod
    def from_model(cls, ceiling: ScopedBudget, budget: Budget) -> ScopedBudgetResponse:
        """Create a ScopedBudgetResponse from a ceiling and the budget it names."""
        return cls(**cls._figures_of(ceiling, budget))


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
    reset_cycle: ResetCycle | None = Field(default=None, description=_CYCLE_DESCRIPTION)
    reset_every_n: int | None = Field(default=None, gt=0, description=_EVERY_N_DESCRIPTION)
    reset_anchor_at: datetime | None = Field(default=None, description=_ANCHOR_DESCRIPTION)
    reset_weekdays: int | None = Field(
        default=None, ge=WEEKDAY_MASK_MIN, le=WEEKDAY_MASK_MAX, description=_WEEKDAYS_DESCRIPTION
    )
    reset_month_day: int | None = Field(default=None, ge=1, le=MAX_MONTH_DAY, description=_MONTH_DAY_DESCRIPTION)
    reset_month: int | None = Field(default=None, ge=1, le=12, description=_MONTH_DESCRIPTION)


# Bounded so one save cannot stage an unbounded number of rows; matches the list routes' page ceiling.
MAX_APPLIED_ENTITIES = 1000


class AppliedEntity(_ModelNarrowing):
    """One entity a budget applies to: a scope inside the organization, optionally narrowed to a provider or model."""

    scope_type: ScopeType = Field(description="Which kind of identity the budget caps")
    scope_id: str = Field(min_length=1, max_length=255, description="Id of the capped identity")

    def key(self) -> tuple[str, str, str | None, str | None]:
        """The entity's identity, which the unique index allows one budget per."""
        return (self.scope_type, self.scope_id, self.provider_key_id, self.model)


class _AppliedTo(BaseModel):
    """The entity list a budget save carries, so the budget and where it applies are written in one step."""

    applied_to: list[AppliedEntity] | None = Field(
        default=None,
        max_length=MAX_APPLIED_ENTITIES,
        description=(
            "Every entity the budget applies to, as a whole set: entities missing from it stop carrying the "
            "budget, and entities already in it keep their spend. Omit to leave the entities as they are"
        ),
    )

    @model_validator(mode="after")
    def _no_repeated_entity(self) -> Self:
        keys = [entity.key() for entity in self.applied_to or ()]
        if len(keys) != len(set(keys)):
            raise ValueError("applied_to names the same entity twice")
        return self


class OrganizationBudgetCreate(_AppliedTo, OrganizationBudgetRates):
    """Create one budget owned by the caller's organization, and apply it to its entities in the same step."""


class OrganizationBudgetUpdate(_AppliedTo, OrganizationBudgetRates):
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
    reset_cycle: str | None
    reset_every_n: int | None
    reset_anchor_at: datetime | None
    reset_weekdays: int | None
    reset_month_day: int | None
    reset_month: int | None
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
            reset_cycle=budget.reset_cycle,
            reset_every_n=budget.reset_every_n,
            reset_anchor_at=budget.reset_anchor_at,
            reset_weekdays=budget.reset_weekdays,
            reset_month_day=budget.reset_month_day,
            reset_month=budget.reset_month,
            ceiling_count=ceiling_count,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


class OrganizationBudgetsPublic(BaseModel):
    data: list[OrganizationBudgetPublic]
    count: int


class OrganizationScopedBudgetCreate(_ModelNarrowing):
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


class OrganizationScopedBudgetUpdate(ScopedBudgetChanges):
    """Relabel a ceiling, or point it at a different budget of this organization's.

    The scope and the provider narrowing are not editable, for the reason
    the deployment-wide ceiling update gives: changing either moves the ceiling to
    a different identity while carrying its spend, which is a delete and a
    create, not an update.
    """


class OrganizationScopedBudgetPublic(ScopedBudgetFigures):
    """One ceiling inside the organization, and the figures it enforces.

    The limit and the period are read through the budget rather than stored here,
    and carried on the wire so a page can render a ceiling without fetching every
    budget to resolve one id. Same reasoning as ``ScopedBudgetResponse``, whose
    shape this deliberately mirrors.
    """

    # This is False when the ceiling's budget belongs to another owner, so its figure cannot change here.
    manageable: bool

    @classmethod
    def from_model(
        cls,
        ceiling: ScopedBudget,
        budget: Budget,
        *,
        organization_id: uuid.UUID,
    ) -> OrganizationScopedBudgetPublic:
        return cls(**cls._figures_of(ceiling, budget), manageable=budget.organization_id == organization_id)


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
    reset_cycle: str | None
    reset_every_n: int | None
    reset_anchor_at: datetime | None
    reset_weekdays: int | None
    reset_month_day: int | None
    reset_month: int | None
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
            reset_cycle=budget.reset_cycle,
            reset_every_n=budget.reset_every_n,
            reset_anchor_at=budget.reset_anchor_at,
            reset_weekdays=budget.reset_weekdays,
            reset_month_day=budget.reset_month_day,
            reset_month=budget.reset_month,
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
    "ScopedBudgetChanges",
    "ScopedBudgetFigures",
    "ScopedBudgetResponse",
    "UpdateBudgetRequest",
    "UpdateScopedBudgetRequest",
    "WorkspaceMemberBudgetPoliciesPublic",
    "WorkspaceMemberBudgetPolicyCreate",
    "WorkspaceMemberBudgetPolicyPublic",
    "WorkspaceMemberBudgetPolicyUpdate",
]
