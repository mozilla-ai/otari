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

from gateway.api.deps import AgentModelRecommenderPortDep, CurrentIdentity, UsageServiceDep, verify_master_key
from gateway.api.deps._base import PortSessionDep
from gateway.api.deps.routing import SubagentDecisionServiceDep
from gateway.api.routes._passthrough import DECISIONS_ENDPOINT
from gateway.api.routes.organization_usage import _scope
from gateway.api.routes.usage import _END_DESC, _START_DESC, _USER_MULTI_DESC, _WORKSPACE_DESC
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.core.surface import Surface
from gateway.exceptions.routing_exceptions import RecommenderNotConfiguredError
from gateway.schemas.routing import SubagentSavingsPublic
from gateway.schemas.usage_and_telemetry import Bucket
from gateway.services.routing import ROUTER_LABEL_MODEL, ROUTER_LABEL_PROVIDER
from gateway.services.usage_and_telemetry import UsageFilters, resolve_window

router = APIRouter(
    prefix="/organizations/me/subagent-savings",
    tags=["organization-usage"],
    dependencies=[Depends(verify_master_key)],
)

SURFACE = Surface("subagent_savings")


@router.get("")
async def read_subagent_savings(
    identity: CurrentIdentity,
    db: PortSessionDep,
    decisions: SubagentDecisionServiceDep,
    usage: UsageServiceDep,
    recommender: AgentModelRecommenderPortDep,
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
    scope = await _scope(db, user=identity, workspace_id=workspace_id)
    start, end = resolve_window(start_date, end_date)
    try:
        recommender.quote()
        recommender_available = True
    except RecommenderNotConfiguredError:
        recommender_available = False
    unanswered = await usage.count(
        UsageFilters(
            start_date=start,
            end_date=end,
            user_id=user_id,
            status="error",
            model=[ROUTER_LABEL_MODEL],
            provider=ROUTER_LABEL_PROVIDER,
            endpoint=DECISIONS_ENDPOINT,
        ),
        scope,
    )
    return await decisions.savings(
        workspaces=scope.workspaces if scope.workspaces is not None else [],
        start=start,
        end=end,
        user_ids=user_id,
        bucket=bucket,
        recommender_available=recommender_available,
        router_unanswered=unanswered,
    )
