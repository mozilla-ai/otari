"""Bulk usage log endpoint.

Provides a single query interface over the usage logs of the caller's
organization with optional time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

import uuid
from datetime import datetime
from time import monotonic
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    CallerUsageScope,
    CallerWorkspaceQuery,
    UsageServiceDep,
    get_config,
    get_db,
    require_deployment_operator,
    verify_api_key_or_master_key,
)
from gateway.api.routes._helpers import retryable_on_database_outage
from gateway.api.routes._usage_query import (
    API_KEY_MULTI_DESC,
    COUNTS_DESC,
    DIMENSIONS_DESC,
    END_DESC,
    ENDPOINT_DESC,
    MAX_REQUEST_GROUPS,
    MODEL_MULTI_DESC,
    PRICED_DESC,
    PROVIDER_DESC,
    REQUEST_GROUP_DESC,
    SOURCE_DESC,
    SOURCE_LABEL_DESC,
    START_DESC,
    STATUS_CODE_DESC,
    STATUS_DESC,
    TOOL_DESC,
    USER_MULTI_DESC,
    WORKSPACE_DESC,
    GroupByTag,
    TagFilter,
)
from gateway.core.config import GatewayConfig
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.core.surface import Surface
from gateway.inflight import get_registry
from gateway.models.api_keys import APIKey
from gateway.schemas.usage_and_telemetry import (
    Bucket,
    RequestSettlement,
    SeriesGroupBy,
    SummaryDimension,
    ToolFilter,
    UsageCount,
    UsageEntry,
    UsageGroupedSeries,
    UsageSummary,
)
from gateway.services.external_usage_service import (
    ExternalEventsRequest,
    ExternalIngestResult,
    ingest_external_events,
)
from gateway.services.usage_admin_service import (
    UsageDeleteRequest,
    UsageDeleteResult,
    UsageSetPriceRequest,
    UsageSetPriceResult,
    delete_usage,
    set_usage_price,
)
from gateway.services.usage_and_telemetry import UsageFilters, parse_tag_filters

# Two routers under one prefix, because their callers authenticate
# differently. Reading or amending usage rows across an organization is an
# operator's job, so that gate is declared on the router and a route added later
# inherits it; ``POST /external-events`` files rows on behalf of the API key that
# holds them, and ``GET /requests/{request_id}`` reads a key's own request back,
# so both sit on a router of their own rather than a route-level override, and
# admitting a non-operator is spelled here. Each
# router names its own rule, so adding a route to either one inherits a gate
# rather than none.
operator_router = APIRouter(
    prefix="/usage",
    tags=["usage"],
    dependencies=[Depends(require_deployment_operator)],
)
key_router = APIRouter(
    prefix="/usage",
    tags=["usage"],
    dependencies=[Depends(verify_api_key_or_master_key)],
)

SURFACE = Surface("usage")

# How many in-flight requests are serialized. A live panel is read at a glance, so
# a fixed cap beats a pagination knob; the response reports the true count next to
# the capped list, and the longest-running are the ones kept.
_MAX_IN_FLIGHT_ROWS = 50


class InFlightEntry(BaseModel):
    """One request the gateway is serving right now.

    Field names match their ``UsageEntry`` counterparts so a request reads the
    same way in flight as it does once it has settled. ``id`` is the exception: it
    is an ephemeral tracking id, not the id of the usage row this will become.
    """

    id: str
    endpoint: str
    model: str
    provider: str | None
    user_id: str | None
    api_key_id: str | None
    policy_name: str | None
    started_at: datetime
    elapsed_ms: int


class InFlightResponse(BaseModel):
    """The requests in flight on the answering worker."""

    requests: list[InFlightEntry]
    total: int


# The count endpoint alone narrows the false case further, so it cannot publish the
# description COUNTS_DESC gives: there it would promise the rows this count is what excludes.
_COUNT_COUNTS_DESC = (
    "Filter by budget participation: true = only enforced gateway rows, false = only "
    "imported rows, narrowed past the filter of the same name on GET /api/v1/usage so the "
    "total matches what bulk delete and set-price can reach"
)


@operator_router.get("")
async def list_usage(
    usage: UsageServiceDep,
    usage_scope: CallerUsageScope,
    start_date: datetime | None = Query(default=None, description=START_DESC),
    end_date: datetime | None = Query(default=None, description=END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=STATUS_DESC),
    status_code: int | None = Query(default=None, description=STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=PROVIDER_DESC),
    source: str | None = Query(default=None, description=SOURCE_DESC),
    source_label: str | None = Query(default=None, description=SOURCE_LABEL_DESC),
    tag: TagFilter = None,
    api_key_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=API_KEY_MULTI_DESC)] = None,
    priced: bool | None = Query(default=None, description=PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=COUNTS_DESC),
    request_group_id: Annotated[
        list[str] | None, Query(max_length=MAX_REQUEST_GROUPS, description=REQUEST_GROUP_DESC)
    ] = None,
    workspace_id: Annotated[uuid.UUID | None, Query(description=WORKSPACE_DESC)] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, status, failure status code,
    model, endpoint, provider, source, session (``source_label``), and request
    group (``request_group_id``, repeatable, which returns a routed request's
    whole attempt plan). Paginated via skip/limit. The return shape is a bare JSON array; external
    billing/analytics consumers depend on this, so the total row count for a
    paginated UI is served separately by ``GET /api/v1/usage/count`` rather than
    wrapped in an envelope here. Timestamps accept either ISO 8601 strings or
    Unix epoch seconds (numeric).
    """
    filters = UsageFilters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        tags=parse_tag_filters(tag),
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
        workspace_id=workspace_id,
    )
    return await usage.page(filters, usage_scope, skip=skip, limit=limit)


@key_router.post("/external-events")
async def ingest_external_usage(
    request: ExternalEventsRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ExternalIngestResult:
    """Ingest a batch of externally-observed usage events.

    Authenticated with either an API key or the master key. Usage binds to the
    authenticated principal: an API key attributes to its own user (and stamps its
    id on the rows); the master key may name any user via ``user_id``. Records
    subscription-backed usage (e.g. Claude Code) as usage-log rows tagged with their
    ``source``, priced at the effective API rate for each event's timestamp.
    Imported usage is real cost, but never counts toward budgets or mutates
    ``users.spend`` (it is retrospective, so it cannot be reserved). Idempotent by
    ``(source, source_event_id)``. The payload is content-free; any
    prompt/completion/tool field is rejected (422), not stored.
    """
    api_key, is_master_key = auth_result
    with retryable_on_database_outage("external usage ingest"):
        return await ingest_external_events(
            db,
            request,
            api_key=api_key,
            is_master_key=is_master_key,
            reject_user_mismatch=config.reject_user_mismatch,
        )


@key_router.get("/requests/{request_id}")
async def get_request_settlement(
    request_id: Annotated[str, Path(max_length=255)],
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    usage: UsageServiceDep,
) -> RequestSettlement:
    """Look up a request's settled cost by the ``Otari-Request-ID`` it was sent.

    This is how a caller recovers the cost of a request whose response never
    carried one: a stream that failed or was disconnected mid-response, or a
    request that errored. An API key sees only the requests it made; the master
    key sees any. Returns 404 until the request has settled (its rows are written
    by a background writer, so a lookup made the instant a stream closes can
    precede them), and for an id that is unknown or belongs to another key, with
    no way to tell those apart. A stream that ended before the provider reported
    usage, and ran no gateway tools, writes no row, so its id stays 404.
    """
    api_key, is_master_key = auth_result
    api_key_id: str | None = None
    if not is_master_key:
        if api_key is None:
            raise HTTPException(status_code=404, detail="Request not found")
        api_key_id = api_key.id
    settlement = await usage.request_settlement(request_id, api_key_id=api_key_id)
    if settlement is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return settlement


@operator_router.get("/count")
async def count_usage(
    usage: UsageServiceDep,
    usage_scope: CallerUsageScope,
    start_date: datetime | None = Query(default=None, description=START_DESC),
    end_date: datetime | None = Query(default=None, description=END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=STATUS_DESC),
    status_code: int | None = Query(default=None, description=STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=PROVIDER_DESC),
    source: str | None = Query(default=None, description=SOURCE_DESC),
    source_label: str | None = Query(default=None, description=SOURCE_LABEL_DESC),
    tag: TagFilter = None,
    api_key_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=API_KEY_MULTI_DESC)] = None,
    priced: bool | None = Query(default=None, description=PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNT_COUNTS_DESC),
    request_group_id: Annotated[
        list[str] | None, Query(max_length=MAX_REQUEST_GROUPS, description=REQUEST_GROUP_DESC)
    ] = None,
    workspace_id: Annotated[uuid.UUID | None, Query(description=WORKSPACE_DESC)] = None,
) -> UsageCount:
    """Total number of usage logs matching the given filters.

    Serves the dashboard paginator's "N of M" total without changing the bare
    array contract of ``GET /api/v1/usage``. Runs only when the client asks (a
    separate request), so the ``COUNT(*)`` is not paid on every page load. With
    ``counts_toward_budget=false`` it also backs the "select all N matching this
    filter" affordance for bulk delete / set-price, which touch imported rows only.

    That value is the one place this count is narrower than ``GET /api/v1/usage``: it
    also excludes rows this deployment served itself, so the number an operator
    confirms is the number the mutation can reach. The list still pages the
    budget-exempt gateway rows it omits.
    """
    filters = UsageFilters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        tags=parse_tag_filters(tag),
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
        workspace_id=workspace_id,
    )
    # counts_toward_budget alone does not say "imported": gateway traffic on an
    # exclude_from_budget key is also False, so without this the count would
    # promise rows _selection_conditions then refuses to touch.
    return UsageCount(total=await usage.count(filters, usage_scope, imported_only=counts_toward_budget is False))


@operator_router.get("/in-flight")
async def list_in_flight(raw_request: Request) -> InFlightResponse:
    """Requests the gateway is currently serving, longest-running first.

    A usage row is written when a request settles, so the log alone cannot answer
    "is anything happening right now": on a slow backend, a 30-second local model
    call is invisible until it finishes. This reports what is in progress.

    Read from an in-memory registry, so it describes the process that answers this
    call and not the deployment: behind a load balancer, consecutive polls reach
    different otari processes, and there is no deployment-wide total to ask for.
    ``total`` is the true in-flight count for the answering process even when
    ``requests`` is capped.
    """
    registry = get_registry(raw_request)
    if registry is None:
        return InFlightResponse(requests=[], total=0)
    entries = registry.snapshot()
    # One clock reading for the whole response, so two rows started together
    # report the same elapsed time.
    now = monotonic()
    return InFlightResponse(
        requests=[
            InFlightEntry(
                id=entry.id,
                endpoint=entry.endpoint,
                model=entry.model,
                provider=entry.provider,
                user_id=entry.user_id,
                api_key_id=entry.api_key_id,
                policy_name=entry.policy_name,
                started_at=entry.started_at,
                elapsed_ms=entry.elapsed_ms(now),
            )
            for entry in entries[:_MAX_IN_FLIGHT_ROWS]
        ],
        total=len(entries),
    )


@operator_router.delete("")
async def delete_usage_rows(
    request: UsageDeleteRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_ids: CallerWorkspaceQuery,
) -> UsageDeleteResult:
    """Delete imported usage rows by explicit ids or by filter.

    Target either the current selection (``ids``) or everything matching a filter
    (``by_filter: true`` plus optional ``source`` / ``model`` / ``user_id`` /
    ``status`` / date range / ``priced``). Only imported rows
    (``counts_toward_budget = false``) are ever removed: enforced gateway rows and
    the spend ledger (``users.spend``) are untouched, so a delete can never desync a
    budget. Master-key only.
    """
    return await delete_usage(db, request, workspace_ids=workspace_ids)


@operator_router.post("/set-price")
async def set_usage_price_rows(
    request: UsageSetPriceRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_ids: CallerWorkspaceQuery,
) -> UsageSetPriceResult:
    """Set the cost of imported usage rows from manual per-1M rates.

    Target either the current selection (``ids``) or everything matching a filter
    (``by_filter: true``). Cost / billing meters / pricing breakdown are recomputed
    from each row's own token counts at the supplied ``input`` / ``output`` /
    ``cache_read`` / ``cache_write`` per-1M rates (manual rates, not a recompute from
    configured pricing). Only imported rows (``counts_toward_budget = false``) are
    touched, so ``users.spend`` is never affected. Master-key only.
    """
    return await set_usage_price(db, request, workspace_ids=workspace_ids)


# ---------------------------------------------------------------------------
# Aggregated analytics (dashboard Usage page). Separate from the bare-array
# list above, which stays a stable external-consumer contract.
# ---------------------------------------------------------------------------


@operator_router.get("/summary")
async def usage_summary(
    usage: UsageServiceDep,
    usage_scope: CallerUsageScope,
    start_date: datetime | None = Query(default=None, description=START_DESC),
    end_date: datetime | None = Query(default=None, description=END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=STATUS_DESC),
    status_code: int | None = Query(default=None, description=STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=PROVIDER_DESC),
    source: str | None = Query(default=None, description=SOURCE_DESC),
    source_label: str | None = Query(default=None, description=SOURCE_LABEL_DESC),
    tag: TagFilter = None,
    api_key_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=API_KEY_MULTI_DESC)] = None,
    priced: bool | None = Query(default=None, description=PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=COUNTS_DESC),
    workspace_id: Annotated[uuid.UUID | None, Query(description=WORKSPACE_DESC)] = None,
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
    dimensions: list[SummaryDimension] | None = Query(default=None, description=DIMENSIONS_DESC),
    group_by_tag: GroupByTag = None,
) -> UsageSummary:
    """Aggregate spend, tokens, and request volume for the dashboard Usage page.

    Range-bounded (default last 30 days, hard-capped): unlike the raw ``/api/v1/usage``
    list, every aggregate is scoped to a bounded window so it stays served by the
    timestamp index. Returns grand totals, breakdowns by model / user / API key /
    source / session (``source_label``) / endpoint / provider (top rows plus a
    reconciling ``other`` fold, billed token counts), the error taxonomy grouped
    by failure status code, and a UTC-bucketed time series carrying each bucket's
    error count and billed token composition (input incl. cache, cache read/write,
    output).

    Each breakdown is its own ``GROUP BY`` pass, so a caller that reads only the
    totals or the series should narrow ``dimensions`` rather than pay for all eight
    (the dashboard's tiles, timeline context, and model typeahead all do). Omitting
    the parameter keeps the full set.

    ``model``, ``user_id``, and ``api_key_id`` are repeatable: several values match
    any of them, so one chart can compare a handful of models, users, or keys.
    """
    filters = UsageFilters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        tags=parse_tag_filters(tag),
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        workspace_id=workspace_id,
    )
    return await usage.summary(filters, usage_scope, bucket=bucket, dimensions=dimensions, group_by_tag=group_by_tag)


@operator_router.get("/series")
async def usage_series(
    usage: UsageServiceDep,
    usage_scope: CallerUsageScope,
    group_by: SeriesGroupBy = Query(description="Dimension to split the series by"),
    start_date: datetime | None = Query(default=None, description=START_DESC),
    end_date: datetime | None = Query(default=None, description=END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=STATUS_DESC),
    status_code: int | None = Query(default=None, description=STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=PROVIDER_DESC),
    source: str | None = Query(default=None, description=SOURCE_DESC),
    source_label: str | None = Query(default=None, description=SOURCE_LABEL_DESC),
    tag: TagFilter = None,
    api_key_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=API_KEY_MULTI_DESC)] = None,
    priced: bool | None = Query(default=None, description=PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=COUNTS_DESC),
    workspace_id: Annotated[uuid.UUID | None, Query(description=WORKSPACE_DESC)] = None,
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
) -> UsageGroupedSeries:
    """Time series split by one dimension, for the dashboard's stacked charts.

    Same filters and window bounds as ``/summary`` (kept in lockstep: the
    dashboard serializes one filter object for both, and a filter this endpoint
    silently ignored would make the stacked chart disagree with the tiles beside
    it). The window's top groups by spend are returned as their own series;
    everything past the top eight folds into a single ``other`` series per
    bucket, so the stack always reconciles with the summary totals. Points are
    sparse (populated cells only); the bucket grid is bounded like ``/summary``'s
    series, so an hourly bucket over a too-wide window is rejected rather than
    ballooning the payload.
    """
    filters = UsageFilters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        tags=parse_tag_filters(tag),
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        workspace_id=workspace_id,
    )
    return await usage.grouped_series(filters, usage_scope, bucket=bucket, group_by=group_by)
