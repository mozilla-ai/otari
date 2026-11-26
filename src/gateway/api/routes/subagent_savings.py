"""What the subagent router saved the caller's organization.

A second reading of the recorded subagent decisions, shaped like
``organization_usage.py``: the scope is derived from the caller's own
membership and never accepted, ``workspace_id`` narrows it and cannot widen
it, and a workspace outside it answers 404. Every decision is recorded against
a workspace, so there is no deployment-wide twin.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from gateway.api.deps import CurrentIdentity, WorkspaceAccessDep, verify_master_key
from gateway.api.deps.routing import SubagentDecisionServiceDep
from gateway.api.routes.usage import _END_DESC, _START_DESC, _USER_MULTI_DESC, _WORKSPACE_DESC
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.core.surface import Surface
from gateway.schemas.routing import SubagentSavingsPublic
from gateway.schemas.usage_and_telemetry import Bucket

router = APIRouter(
    prefix="/organizations/me/subagent-savings",
    tags=["organization-usage"],
    dependencies=[Depends(verify_master_key)],
)

SURFACE = Surface("subagent_savings")


@router.get("")
async def read_subagent_savings(
    identity: CurrentIdentity,
    access: WorkspaceAccessDep,
    decisions: SubagentDecisionServiceDep,
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    bucket: Annotated[Bucket, Query(description="The grain of `series`.")] = "day",
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)] = None,
    workspace_id: Annotated[uuid.UUID | None, Query(description=_WORKSPACE_DESC)] = None,
) -> SubagentSavingsPublic:
    """What the subagent router saved, within what the caller's membership lets them see.

    Each recommendation records the model the subagent would have started on
    without it: the model the caller asked for, else the parent's. When the
    harness reports what the subagent consumed, those tokens are priced at the
    models it ran on and at that baseline. The saving is the difference, less
    what the recommendations themselves cost, so it can be negative.

    The costs count only the outcomes priced on both sides. `router_cost`
    counts every recommendation in the window, also one whose subagent was
    never reported or could not be priced. The figures are estimates unless
    counted under `measured`. With no dates, the last 30 days.
    """
    workspaces = await access.readable_workspaces(user=identity, workspace_id=workspace_id)
    return await decisions.savings(
        workspaces=workspaces, start_date=start_date, end_date=end_date, user_ids=user_id, bucket=bucket
    )
