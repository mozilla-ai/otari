"""Response models of the dashboard overview."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorstAllocationResponse(BaseModel):
    """The row furthest through its allowance."""

    budget_id: str
    name: str | None = Field(description="The row's own name, or null where nobody gave it one.")
    spent: float
    allocated: float
    scope_type: str | None = Field(
        description="What a spend ceiling caps (workspace, org_member, api_token, ...); null for a budget."
    )
    scope_id: str | None = Field(description="The scope's id, so an unnamed ceiling can be named after it.")


class AllocationHealthResponse(BaseModel):
    """One set of capped rows, reduced to what a strip renders."""

    over_count: int = Field(description="Rows at or past their allowance.")
    near_count: int = Field(description="Rows at 80% of their allowance or more, but not past it.")
    capped_count: int = Field(description="Rows with a finite cap, which are the ones that can be judged.")
    total_count: int = Field(
        description="Rows of any kind, so a caller can tell 'none configured' from 'none caps spend'."
    )
    worst: WorstAllocationResponse | None


class OverviewSummaryResponse(BaseModel):
    """What the dashboard overview renders beside its usage chart.

    ``budgets`` and ``ceilings`` are null where the caller may not see them,
    which is not the same as a strip with nothing in it: deployment budgets are
    the operator's, and spend ceilings are an organization owner's or admin's.
    """

    active_keys: int
    active_members: int = Field(description="Active members of the named workspace; 0 when none is named.")
    budgets: AllocationHealthResponse | None
    ceilings: AllocationHealthResponse | None
