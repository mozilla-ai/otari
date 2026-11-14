"""What the API key a request authenticates with has spent, and what it may still spend.

An application that gives each of its users a key of their own needs to show
that user where they stand without holding the master key, which a hosted
tenant does not have. These reads answer for the calling key alone: the key is
the credential, so no parameter names a user, a key or a workspace, and none
can widen what a key sees.

The aggregation is the usage service's and the caps are the budget gate's, so a
number read here is the number the operator's views and the gate work from.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from gateway.api.deps import BudgetServiceDep, CallingKey, UsageServiceDep, get_config, verify_api_key
from gateway.api.key_scopes import key_scope
from gateway.api.routes.usage import (
    _COUNTS_DESC,
    _DIMENSIONS_DESC,
    _END_DESC,
    _ENDPOINT_DESC,
    _MODEL_MULTI_DESC,
    _PRICED_DESC,
    _PROVIDER_DESC,
    _SOURCE_DESC,
    _SOURCE_LABEL_DESC,
    _START_DESC,
    _STATUS_CODE_DESC,
    _STATUS_DESC,
    _TOOL_DESC,
    GroupByTag,
    TagFilter,
)
from gateway.core.config import GatewayConfig
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.models.api_keys import APIKey
from gateway.schemas.budgets import KeyBudgetPublic
from gateway.schemas.usage_and_telemetry import (
    Bucket,
    SeriesGroupBy,
    SummaryDimension,
    ToolFilter,
    UsageGroupedSeries,
    UsageSummary,
)
from gateway.services.usage_and_telemetry import UsageFilters, UsageScope, parse_tag_filters
from gateway.types.key_scope import KeyScope

# An API key and nothing else, declared on the router so a route added here inherits it.
router = APIRouter(prefix="/key", tags=["key"], dependencies=[Depends(verify_api_key)])


def _own_rows(api_key: APIKey) -> UsageScope:
    """The rows this key wrote, in the workspace it bills to."""
    return UsageScope(workspaces=[api_key.workspace_id], api_key_id=api_key.id)


@router.get("/budget", openapi_extra=key_scope(KeyScope.USAGE_READ))
async def get_key_budget(
    api_key: CallingKey,
    budgets: BudgetServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> KeyBudgetPublic:
    """What the calling key may still spend, and every cap that decides it.

    Lists the budget the key's user is on and each spend ceiling on a scope the
    key bills to (the key itself, its user's memberships, its workspace and its
    organization), as the budget gate counts them: recorded use plus the holds of
    requests in flight. A period that has ended reads as the fresh one the next
    request starts. ``remaining_usd`` is the least any cap on every provider
    leaves, which is what a request can still spend before a 403. Caps shared with
    the rest of the workspace or organization report only what they leave.
    """
    return await budgets.key_budget(api_key, strategy=config.budget_strategy)


@router.get("/usage/summary", openapi_extra=key_scope(KeyScope.USAGE_READ))
async def get_key_usage_summary(
    api_key: CallingKey,
    usage: UsageServiceDep,
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    tag: TagFilter = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
    dimensions: list[SummaryDimension] | None = Query(default=None, description=_DIMENSIONS_DESC),
    group_by_tag: GroupByTag = None,
) -> UsageSummary:
    """Spend, tokens and request volume of the calling key's own requests.

    The same summary as ``GET /api/v1/usage/summary``, over the rows this key
    wrote alone: the same bounded window (default last 30 days), breakdowns, tag
    filters and ``group_by_tag``. Tag a request with ``metadata`` and pass
    ``group_by_tag`` to read what each kind of request cost.
    """
    filters = UsageFilters(
        start_date=start_date,
        end_date=end_date,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        tags=parse_tag_filters(tag),
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
    )
    return await usage.summary(
        filters, _own_rows(api_key), bucket=bucket, dimensions=dimensions, group_by_tag=group_by_tag
    )


@router.get("/usage/series", openapi_extra=key_scope(KeyScope.USAGE_READ))
async def get_key_usage_series(
    api_key: CallingKey,
    usage: UsageServiceDep,
    group_by: SeriesGroupBy = Query(description="Dimension to split the series by"),
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    tag: TagFilter = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
) -> UsageGroupedSeries:
    """The calling key's own requests over time, split by one dimension.

    The same series as ``GET /api/v1/usage/series``, over the rows this key wrote
    alone, kept in lockstep with the summary above.
    """
    filters = UsageFilters(
        start_date=start_date,
        end_date=end_date,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        tags=parse_tag_filters(tag),
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
    )
    return await usage.grouped_series(filters, _own_rows(api_key), bucket=bucket, group_by=group_by)
