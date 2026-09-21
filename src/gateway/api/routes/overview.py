"""The dashboard overview's summary.

Thin composition over ``gateway.services.overview.overview_service``, which
carries the reasoning: which caller sees which strip, and why the judgment
happens here rather than in the browser.

One endpoint for one page, which is unusual on this surface and deliberate. The
overview renders three integers and one worst-case row per strip, and answering
it from the list routes meant reading four whole tables to compute them
(otari#1425, under otari#1376).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from gateway.api.deps import CurrentIdentity, OverviewServiceDep, verify_master_key
from gateway.services.overview.overview_service import AllocationHealth, OverviewSummary

# Authentication on the router for the reason ``admin.py`` declares it there:
# the one handler happens to take an identity today, and one that did not would
# be unauthenticated with nothing to notice. Authorization is not here, because
# this answers every signed-in caller and withholds per caller instead: the
# service decides which strip each one may see.
router = APIRouter(
    prefix="/overview",
    tags=["overview"],
    dependencies=[Depends(verify_master_key)],
)


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


def _health(health: AllocationHealth | None) -> AllocationHealthResponse | None:
    if health is None:
        return None
    return AllocationHealthResponse(
        over_count=health.over_count,
        near_count=health.near_count,
        capped_count=health.capped_count,
        total_count=health.total_count,
        worst=(
            WorstAllocationResponse(
                budget_id=health.worst.budget_id,
                name=health.worst.name,
                spent=health.worst.spent,
                allocated=health.worst.allocated,
                scope_type=health.worst.scope_type,
                scope_id=health.worst.scope_id,
            )
            if health.worst is not None
            else None
        ),
    )


@router.get("")
async def get_overview(
    service: OverviewServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Narrow the counts to one workspace of the caller's organization."),
    ] = None,
) -> OverviewSummaryResponse:
    """Summarize what the dashboard overview shows beside its usage chart.

    The counts and the budget judgment in one answer, so the page does not read
    four collections to compute them. A workspace outside the caller's
    organization is treated as none given rather than refused, because the id
    comes from a switcher whose contents can go stale.
    """

    summary: OverviewSummary = await service.summary(identity=current_identity, workspace_id=workspace_id)
    return OverviewSummaryResponse(
        active_keys=summary.active_keys,
        active_members=summary.active_members,
        budgets=_health(summary.budgets),
        ceilings=_health(summary.ceilings),
    )
