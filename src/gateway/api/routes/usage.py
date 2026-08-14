"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

import csv
import io
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Annotated, Any, Literal, NamedTuple, TypeVar, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import ColumnElement, and_, case, func, null, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key, verify_master_key
from gateway.api.routes._billing_schemas import ChargeLine, MeterMap
from gateway.core.config import GatewayConfig
from gateway.core.sql import MAX_FILTER_VALUES, match_any, utc_bound
from gateway.inflight import get_registry
from gateway.models.entities import APIKey, UsageLog, User
from gateway.services.external_usage_service import (
    ExternalEventsRequest,
    ExternalIngestResult,
    ingest_external_events,
)
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.tool_usage import TOOL_METER_NAMESPACE
from gateway.services.usage_admin_service import (
    UsageDeleteRequest,
    UsageDeleteResult,
    UsageSetPriceRequest,
    UsageSetPriceResult,
    delete_usage,
    set_usage_price,
)
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME

router = APIRouter(prefix="/v1/usage", tags=["usage"])

# The analytics summary is range-bounded, unlike the raw list. Absent a start_date
# it looks back this far; a wider explicit window is clamped to the hard cap so a
# single request can never turn into an unbounded full-table scan on a growing log.
_DEFAULT_SUMMARY_LOOKBACK = timedelta(days=30)
_MAX_SUMMARY_SPAN = timedelta(days=366)

# How many rows each breakdown returns before the remainder is folded into a
# single synthesized "other" row (so the tables still reconcile with the totals).
_BREAKDOWN_TOP_N = 100

# How many groups a grouped time series carries before the remainder folds into
# "other". Eight is the ceiling a stacked chart can keep legible (and the size of
# the dashboard's fixed categorical palette); the breakdown tables, not the chart,
# are the place to read a longer tail.
_SERIES_TOP_N = 8

# How many in-flight requests are serialized. A live panel is read at a glance, so
# a fixed cap beats a pagination knob; the response reports the true count next to
# the capped list, and the longest-running are the ones kept.
_MAX_IN_FLIGHT_ROWS = 50

# Sessions (``source_label``) are an order of magnitude higher-cardinality than
# models or users: one agent workload can open hundreds of them in a month, and
# the interesting signal is a long-ish head ("which tasks burned the budget"),
# not just the top few. Give that dimension a deeper cap so the head is not
# swallowed by the "other" fold.
_SESSION_BREAKDOWN_TOP_N = 250

# How many request groups one call may ask for. The dashboard batches the groups
# visible on a page of the activity log into a single lookup, so the bound tracks
# the largest page size (1000) rather than a plan's candidate count; it exists to
# keep a caller from posting an unbounded IN list.
_MAX_REQUEST_GROUPS = 1000

Bucket = Literal["hour", "day"]
SeriesGroupBy = Literal["model", "user_id", "api_key_id", "source"]

# Coarse display buckets for a failure's status code. A closed Literal rather than
# a bare str so the set lands in the OpenAPI schema as an enum and a consumer can
# switch on it exhaustively instead of string-matching whatever the server sent.
ErrorClass = Literal["pricing", "rate_limit", "auth", "provider_error", "client_error", "unknown"]

# Every breakdown ``/summary`` can compute, mapped to the column it groups by and
# its top-N cap. A dimension name is the ``by_<name>`` response field it fills, so
# a caller reads the selector and the payload with one vocabulary.
class _LabelJoin(NamedTuple):
    """How to resolve a breakdown key's display name in the same GROUP BY.

    Only the two dimensions whose key is an opaque id need this: a model, source
    or endpoint already reads as its own name. Resolving it here is what lets a
    client offer a user or key filter without holding those whole tables.
    """

    entity: Any
    on: Any
    label: Any


_USER_LABEL = _LabelJoin(entity=User, on=User.user_id == UsageLog.user_id, label=User.alias)
_API_KEY_LABEL = _LabelJoin(entity=APIKey, on=APIKey.id == UsageLog.api_key_id, label=APIKey.key_name)

_SUMMARY_DIMENSIONS: dict[str, tuple[Any, int, "_LabelJoin | None"]] = {
    "model": (UsageLog.model, _BREAKDOWN_TOP_N, None),
    "user": (UsageLog.user_id, _BREAKDOWN_TOP_N, _USER_LABEL),
    "api_key": (UsageLog.api_key_id, _BREAKDOWN_TOP_N, _API_KEY_LABEL),
    "source": (UsageLog.source, _BREAKDOWN_TOP_N, None),
    "source_label": (UsageLog.source_label, _SESSION_BREAKDOWN_TOP_N, None),
    "endpoint": (UsageLog.endpoint, _BREAKDOWN_TOP_N, None),
    "provider": (UsageLog.provider, _BREAKDOWN_TOP_N, None),
}

# The failure taxonomy (``errors_by_status_code``) is a GROUP BY pass like the
# breakdowns above, but it groups failures by status code rather than spend by a
# dimension, so it is selectable by name without living in _SUMMARY_DIMENSIONS.
# It is the one dimension whose response field is not ``by_<name>``.
_ERROR_TAXONOMY_DIMENSION = "status_code"

# The gateway-run tools that can be enumerated for a filter or a breakdown. MCP
# tool names come from a caller-supplied server, so they are unbounded and appear
# only in a row's own detail, never as a dimension of their own. The ``any``
# selector still matches them, because it tests the meter namespace itself.
GATEWAY_TOOL_NAMES: tuple[str, ...] = (WEB_SEARCH_TOOL_NAME, CODE_EXECUTION_TOOL_NAME)
_ANY_TOOL = "any"
ToolFilter = Literal["any", "web_search", "code_execution"]

# Keep in step with _SUMMARY_DIMENSIONS; the extra ``none`` is the explicit empty
# selection (a repeated query param cannot express an empty list on the wire).
SummaryDimension = Literal[
    "model", "user", "api_key", "source", "source_label", "endpoint", "provider", "status_code", "tool", "none"
]

_TOOL_DIMENSION = "tool"
_ALL_SUMMARY_DIMENSIONS: set[str] = set(_SUMMARY_DIMENSIONS) | {_ERROR_TAXONOMY_DIMENSION, _TOOL_DIMENSION}


def _utc_iso(value: datetime) -> str:
    """Serialize a stored timestamp as unambiguous UTC ISO-8601.

    ``usage_logs.timestamp`` is timezone-aware, but SQLite returns it naive (it does
    not persist the offset). A naive ``isoformat()`` has no ``+00:00``, so a browser
    reads it in its own local zone and a recent UTC event can land in the future,
    showing as "0s ago". Treat a naive value as the UTC it was stored as.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


class UsageEntry(BaseModel):
    """A single usage log entry."""

    id: str
    user_id: str | None
    # Display labels resolved server-side, so a client rendering a page of rows
    # does not have to hold the whole users/api_keys tables to name them. Null
    # when the row has no owner, when the referenced row is gone (both foreign
    # keys are ON DELETE SET NULL), or when the entity simply has no label set;
    # a client falls back to the id in every one of those cases.
    user_alias: str | None = None
    api_key_id: str | None
    api_key_name: str | None = None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cache_write_1h_tokens: int | None
    # Precise shapes with a permissive fallback arm; see _billing_schemas for why
    # the fallback is what keeps a row written by an older gateway renderable.
    billing_meters: MeterMap | None
    pricing_breakdown: Sequence[ChargeLine] | None
    cost: float | None
    status: str
    error_message: str | None
    status_code: int | None
    latency_ms: int | None
    source: str
    source_label: str | None
    counts_toward_budget: bool
    # Routing attribution. All null for a request that named a plain model.
    # `status == "absorbed"` marks an attempt a policy recovered from; those rows
    # are excluded from `error_count` and from `request_count`, since the request
    # they belong to is counted once by the attempt that served it.
    policy_name: str | None = None
    selection_reason: str | None = None
    attempt_position: int | None = None
    attempt_count: int | None = None
    request_group_id: str | None = None

    @classmethod
    def from_model(
        cls,
        log: UsageLog,
        *,
        user_alias: str | None = None,
        api_key_name: str | None = None,
    ) -> "UsageEntry":
        return cls(
            id=log.id,
            user_id=log.user_id,
            user_alias=user_alias,
            api_key_id=log.api_key_id,
            api_key_name=api_key_name,
            timestamp=_utc_iso(log.timestamp),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            source=log.source,
            source_label=log.source_label,
            counts_toward_budget=log.counts_toward_budget,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cache_read_tokens=log.cache_read_tokens,
            cache_write_tokens=log.cache_write_tokens,
            cache_write_1h_tokens=log.cache_write_1h_tokens,
            billing_meters=log.billing_meters,
            pricing_breakdown=log.pricing_breakdown,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
            status_code=log.status_code,
            latency_ms=log.latency_ms,
            policy_name=log.policy_name,
            selection_reason=log.selection_reason,
            attempt_position=log.attempt_position,
            attempt_count=log.attempt_count,
            request_group_id=log.request_group_id,
        )


class UsageCount(BaseModel):
    """Total number of usage logs matching a set of filters."""

    total: int


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


# Shared query descriptions so the list and count endpoints stay in lockstep.
_START_DESC = "Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)"
_END_DESC = "Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)"
_STATUS_DESC = (
    "Filter to a single status: 'success', 'error', or 'absorbed' (an attempt a routing policy "
    "recovered from, excluded from error_count and request_count)"
)
_STATUS_CODE_DESC = (
    "Filter to a single failure status code (e.g. 429 for provider rate limits, "
    "402 for missing-pricing rejections). Only error rows carry one, so this "
    "filter also restricts to status='error' unless 'status' is given explicitly"
)
_ENDPOINT_DESC = "Filter to a single endpoint (e.g. '/v1/chat/completions')"
_PROVIDER_DESC = "Filter to a single provider (e.g. 'openai')"
_SOURCE_DESC = "Filter to a single provenance source (e.g. 'gateway' or 'claude_code')"
_SOURCE_LABEL_DESC = "Filter to a single session/project label (the source_label carried by imported usage)"
_REQUEST_GROUP_DESC = (
    "Filter to the rows of one or more request groups; repeatable "
    "(request_group_id=a&request_group_id=b). A routed request writes one row per "
    "attempt, all sharing a request_group_id, so this returns a request's whole plan: "
    "its absorbed attempts and the attempt that served it. Ignore ordering by "
    "timestamp and read attempt_position to reconstruct the plan. At most "
    f"{_MAX_REQUEST_GROUPS} ids per call."
)
_PRICED_DESC = (
    "Filter by token-pricing state: true = only rows whose model tokens were priced, "
    "false = only rows that still need pricing (no cost at all, or tokens that were "
    "never metered because the model had no rate). A row charged only for gateway-run "
    "tool calls still counts as needing pricing."
)
_TOOL_DESC = (
    "Filter to requests that ran a gateway-run tool. 'any' matches any tool; a tool "
    f"name ({', '.join(GATEWAY_TOOL_NAMES)}) matches that tool specifically."
)
_COUNTS_DESC = (
    "Filter by budget participation: true = only enforced gateway rows, "
    "false = only imported rows that never touch a budget"
)
# The three entity filters are repeatable on every usage endpoint, so a chart or a
# log view can compare a handful of models / users / keys instead of one at a time.
# The bulk delete / set-price selection body takes the same form (see
# UsageSelection): "all N matching" is counted over these filters and re-derived
# from that body, so a filter one side could not express would target a different
# set of rows than the operator was shown.
_USER_MULTI_DESC = (
    "Filter to one or more users; repeatable (user_id=a&user_id=b). Several values match any of "
    f"them. At most {MAX_FILTER_VALUES} per call."
)
_MODEL_MULTI_DESC = (
    "Filter to one or more models; repeatable (model=a&model=b). Several values match any of them. "
    f"At most {MAX_FILTER_VALUES} per call."
)
_API_KEY_MULTI_DESC = (
    "Filter to one or more API key ids; repeatable (api_key_id=a&api_key_id=b). Several values "
    f"match any of them. At most {MAX_FILTER_VALUES} per call."
)
_DIMENSIONS_DESC = (
    "Which breakdowns to compute; repeatable (dimensions=model&dimensions=user). Each value names the "
    "'by_<value>' response field it fills, except 'status_code', which fills the failure taxonomy in "
    "'errors_by_status_code'. Omit for every breakdown (the default); pass 'none' for a "
    "totals-and-series-only response. Each dimension left out skips one GROUP BY scan, so a caller that "
    "reads only the tiles or the time series should say so. Fields that were not requested come back empty."
)


def _usage_filters(
    *,
    start_date: datetime | None,
    end_date: datetime | None,
    user_id: str | list[str] | None,
    status: str | None,
    model: str | list[str] | None,
    endpoint: str | None,
    provider: str | None = None,
    source: str | None = None,
    source_label: str | None = None,
    api_key_id: str | list[str] | None = None,
    priced: bool | None = None,
    tool: str | None = None,
    counts_toward_budget: bool | None = None,
    status_code: int | None = None,
    request_group_id: list[str] | None = None,
) -> list[ColumnElement[bool]]:
    """Build the shared WHERE conditions for the list and count endpoints.

    Keeping this in one place guarantees the paginator's total (``/count``)
    always matches the rows ``list_usage`` returns for the same filters.

    Bounds are pinned to UTC here rather than only in ``_resolve_window``, which
    the summary endpoints route through but the list and count endpoints do not:
    an offset-less bound would otherwise resolve against the process's local
    timezone, so the same query would size a different set of rows per deployment.
    """
    conditions: list[ColumnElement[bool]] = []
    if start_date is not None:
        conditions.append(UsageLog.timestamp >= utc_bound(start_date))
    if end_date is not None:
        conditions.append(UsageLog.timestamp < utc_bound(end_date))
    if user_id is not None and user_id != []:
        conditions.append(match_any(UsageLog.user_id, user_id))
    if status is not None:
        conditions.append(UsageLog.status == status)
    if status_code is not None:
        conditions.append(UsageLog.status_code == status_code)
        if status is None:
            # Only a failure carries a status code (see ``UsageLog.status_code``),
            # so a bare code filter means "these failures" rather than "whatever
            # rows happen to hold this code": it stays error-scoped even if a
            # future write path starts stamping a code on a non-error row, and it
            # is served by the (status, timestamp) index instead of scanning the
            # window. An explicit ``status`` wins, so the combination stays a
            # literal query rather than a silently contradictory one.
            conditions.append(UsageLog.status == "error")
    if model is not None and model != []:
        conditions.append(match_any(UsageLog.model, model))
    if endpoint is not None:
        conditions.append(UsageLog.endpoint == endpoint)
    if provider is not None:
        conditions.append(UsageLog.provider == provider)
    if source is not None:
        conditions.append(UsageLog.source == source)
    if source_label is not None:
        conditions.append(UsageLog.source_label == source_label)
    if api_key_id is not None and api_key_id != []:
        conditions.append(match_any(UsageLog.api_key_id, api_key_id))
    if request_group_id:
        # A one-id lookup stays an equality test so it uses the index the same way
        # a single-row fetch would; the IN form is for the dashboard's batched
        # page lookup.
        conditions.append(match_any(UsageLog.request_group_id, request_group_id))
    if priced is True:
        conditions.append(~_needs_pricing_expr())
    elif priced is False:
        conditions.append(_needs_pricing_expr())
    if tool is not None:
        conditions.append(_tool_used_expr(tool))
    if counts_toward_budget is not None:
        conditions.append(UsageLog.counts_toward_budget.is_(counts_toward_budget))
    return conditions


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)
    ] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    request_group_id: Annotated[
        list[str] | None, Query(max_length=_MAX_REQUEST_GROUPS, description=_REQUEST_GROUP_DESC)
    ] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, status, failure status code,
    model, endpoint, provider, source, session (``source_label``), and request
    group (``request_group_id``, repeatable, which returns a routed request's
    whole attempt plan). Paginated via skip/limit. The return shape is a bare JSON array; external
    billing/analytics consumers depend on this, so the total row count for a
    paginated UI is served separately by ``GET /v1/usage/count`` rather than
    wrapped in an envelope here. Timestamps accept either ISO 8601 strings or
    Unix epoch seconds (numeric).
    """
    conditions = _usage_filters(
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
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
    )
    # Outer-joined rather than looked up per row, and rather than left to the
    # client: naming a page of rows must not cost a round trip each, nor oblige a
    # dashboard to hold every user and every key in memory to label 100 rows.
    # Outer so a row whose owner was deleted still comes back, with a null label.
    stmt = (
        select(UsageLog, User.alias, APIKey.key_name)
        .outerjoin(User, User.user_id == UsageLog.user_id)
        .outerjoin(APIKey, APIKey.id == UsageLog.api_key_id)
        .where(*conditions)
        .order_by(UsageLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [
        UsageEntry.from_model(log, user_alias=alias, api_key_name=key_name)
        for log, alias, key_name in result.all()
    ]


@router.post("/external-events")
async def ingest_external_usage(
    request: ExternalEventsRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ExternalIngestResult:
    """Ingest a batch of externally-observed usage events (standalone).

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
    return await ingest_external_events(
        db,
        request,
        api_key=api_key,
        is_master_key=is_master_key,
        reject_user_mismatch=config.reject_user_mismatch,
    )


@router.get("/count", dependencies=[Depends(verify_master_key)])
async def count_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)
    ] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    request_group_id: Annotated[
        list[str] | None, Query(max_length=_MAX_REQUEST_GROUPS, description=_REQUEST_GROUP_DESC)
    ] = None,
) -> UsageCount:
    """Total number of usage logs matching the given filters.

    Serves the dashboard paginator's "N of M" total without changing the bare
    array contract of ``GET /v1/usage``. Runs only when the client asks (a
    separate request), so the ``COUNT(*)`` is not paid on every page load. With
    ``counts_toward_budget=false`` it also backs the "select all N matching this
    filter" affordance for bulk delete / set-price, which touch imported rows only.
    """
    conditions = _usage_filters(
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
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
    )
    stmt: Any = select(func.count()).select_from(UsageLog).where(*conditions)
    total = (await db.execute(stmt)).scalar_one()
    return UsageCount(total=total)


@router.get("/in-flight", dependencies=[Depends(verify_master_key)])
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


@router.delete("", dependencies=[Depends(verify_master_key)])
async def delete_usage_rows(
    request: UsageDeleteRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UsageDeleteResult:
    """Delete imported usage rows by explicit ids or by filter (standalone).

    Target either the current selection (``ids``) or everything matching a filter
    (``by_filter: true`` plus optional ``source`` / ``model`` / ``user_id`` /
    ``status`` / date range / ``priced``). Only imported rows
    (``counts_toward_budget = false``) are ever removed: enforced gateway rows and
    the spend ledger (``users.spend``) are untouched, so a delete can never desync a
    budget. Master-key only.
    """
    return await delete_usage(db, request)


@router.post("/set-price", dependencies=[Depends(verify_master_key)])
async def set_usage_price_rows(
    request: UsageSetPriceRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UsageSetPriceResult:
    """Set the cost of imported usage rows from manual per-1M rates (standalone).

    Target either the current selection (``ids``) or everything matching a filter
    (``by_filter: true``). Cost / billing meters / pricing breakdown are recomputed
    from each row's own token counts at the supplied ``input`` / ``output`` /
    ``cache_read`` / ``cache_write`` per-1M rates (manual rates, not a recompute from
    configured pricing). Only imported rows (``counts_toward_budget = false``) are
    touched, so ``users.spend`` is never affected. Master-key only.
    """
    return await set_usage_price(db, request)


# ---------------------------------------------------------------------------
# Aggregated analytics (dashboard Usage page). Separate from the bare-array
# list above, which stays a stable external-consumer contract.
# ---------------------------------------------------------------------------


class UsageTotals(BaseModel):
    """Grand totals over the filtered window."""

    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int
    request_count: int
    error_count: int
    avg_latency_ms: float | None
    # Served rows with no configured price (cost is NULL), e.g. imported usage for
    # an unpriced model. Surfaced so a $0 cost is not mistaken for free usage.
    # Scoped to status="success": a gateway-side rejection also carries cost=NULL
    # (nothing was spent), so counting error rows here would make a budget or
    # allow-list incident read as a pricing misconfiguration.
    unpriced_requests: int = 0
    # Billed input tokens (fresh input plus both cache buckets), normalized via
    # each row's billing meters where present (see ``_billed_expr``). Unlike
    # ``prompt_tokens`` this is convention-independent, so cache hit rate
    # (cache_read_tokens / billed_input_tokens) is meaningful across providers.
    billed_input_tokens: int = 0
    # Billed output tokens, normalized the same way, so the breakdown fold row
    # reconciles against the same quantity the per-group rows sum (a raw
    # ``completion_tokens`` residual would drift, even negative, whenever a
    # row's meter and column disagree).
    billed_output_tokens: int = 0


class UsageGroupRow(BaseModel):
    """One breakdown row (a model, a user, an API key, a session, ...).

    ``key`` is None both for the synthesized fold row (``is_other=True``) and for a
    real group whose column was NULL (e.g. usage from a since-deleted user, with
    ``is_other=False``). ``is_other`` disambiguates the two so the UI does not
    mislabel deleted-user usage as the fold.
    """

    key: str | None
    # Display name for an opaque key (a user's alias, an API key's name), resolved
    # in the same GROUP BY. Only ever set for the ``user`` and ``api_key``
    # dimensions, and null there too when the entity has no label or is gone; a
    # client falls back to ``key``. Its purpose is to let a client build a user or
    # key filter from this breakdown alone, rather than reading both whole tables.
    label: str | None = None
    cost: float
    tokens: int
    requests: int
    is_other: bool = False


class UsageErrorCodeRow(BaseModel):
    """One error-taxonomy row: the failures in the window sharing a status code.

    ``status_code`` is None for failures recorded without one (rows written
    before the column existed, and failures no HTTP status describes, e.g. a
    stream that finished without usage data under the ``fail`` policy).
    ``error_class`` is the coarse display bucket derived from the code, so a UI
    can group "provider fault" against "my own misconfiguration" without
    re-deriving a status ladder; the raw code stays alongside it for precision.
    """

    status_code: int | None
    error_class: ErrorClass
    requests: int


class UsageSeriesPoint(BaseModel):
    """One time bucket. ``bucket_start`` is canonical ISO-8601 UTC (``...Z``),
    identical across SQLite and PostgreSQL for the same underlying instant.

    ``tokens`` stays the raw provider-reported total (the field predates the
    composition split and external consumers may read it). The billed
    composition fields are normalized via billing meters (see ``_billed_expr``):
    ``input_tokens`` includes both cache buckets, so a chart derives fresh input
    as ``max(0, input_tokens - cache_read_tokens - cache_write_tokens)`` and the
    billed total as ``fresh + cache_read + cache_write + output``.
    """

    bucket_start: str
    cost: float
    tokens: int
    requests: int
    errors: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


class UsageToolRow(BaseModel):
    """Spend and volume for one gateway-run tool inside the window.

    ``calls`` counts billable calls, not requests: one request can run a tool
    several times, which is the whole reason a per-tool view exists. ``errors``
    counts calls that failed and were therefore never billed. ``requests`` is how
    many requests ran the tool at least once, so the number reconciles with the
    Activity list when the same filter is applied there.
    """

    tool: str
    calls: int
    errors: int
    requests: int
    cost: float


class UsageSummary(BaseModel):
    """Aggregate spend/volume for the Usage & analytics page.

    Every breakdown field is always present. One the caller excluded through
    ``dimensions`` comes back as an empty list, the same shape a window with no
    matching rows produces, so narrowing the selector never changes the schema.
    """

    start_date: str
    end_date: str
    bucket: Bucket
    totals: UsageTotals
    by_model: list[UsageGroupRow]
    by_user: list[UsageGroupRow]
    by_api_key: list[UsageGroupRow]
    by_source: list[UsageGroupRow]
    # Session/project attribution for agent traffic: a handful of long-running
    # sessions routinely account for most of a workload's tokens, so this is the
    # dimension that turns "spend went up" into "this task went wrong". Gateway
    # rows carry no label, so they group under a single null key.
    by_source_label: list[UsageGroupRow]
    # API surface (/v1/chat/completions vs /v1/messages vs /v1/responses) and
    # upstream provider: the two splits a gateway operator needs and that no
    # other endpoint reports.
    by_endpoint: list[UsageGroupRow]
    by_provider: list[UsageGroupRow]
    # Gateway-run tool spend. Empty when the window has none, and MCP tools are
    # excluded by design (their names are unbounded, see GATEWAY_TOOL_NAMES).
    by_tool: list[UsageToolRow] = []
    # Failures only, so the taxonomy is not swamped by the successes that carry
    # no status code. Counts sum to ``totals.error_count``, unless a window
    # somehow held more than ``_BREAKDOWN_TOP_N`` distinct codes, in which case
    # the tail is omitted rather than folded (there is no synthesized "other"
    # row: a null key would collide with the real "no code recorded" group).
    errors_by_status_code: list[UsageErrorCodeRow]
    series: list[UsageSeriesPoint]


class UsageGroupedSeriesPoint(BaseModel):
    """One (time bucket, group) cell of a grouped series.

    ``key``/``is_other`` follow the ``UsageGroupRow`` convention: ``key=None``
    with ``is_other=True`` is the fold of groups outside the top N, ``key=None``
    with ``is_other=False`` is a real NULL group (e.g. a deleted user).
    ``tokens`` is the *billed* total (input including cache, plus output), the
    same quantity the ungrouped series' composition fields sum to.
    """

    bucket_start: str
    key: str | None
    is_other: bool = False
    cost: float
    tokens: int
    requests: int


class UsageGroupedSeries(BaseModel):
    """A per-group time series for the dashboard's stacked charts.

    ``groups`` ranks the window's top groups by spend (plus the reconciling
    ``other`` fold), in the order a chart should stack and color them; ``points``
    is sparse (only populated cells), keyed by canonical UTC ``bucket_start``.
    """

    start_date: str
    end_date: str
    bucket: Bucket
    group_by: SeriesGroupBy
    groups: list[UsageGroupRow]
    points: list[UsageGroupedSeriesPoint]


def _resolve_window(start_date: datetime | None, end_date: datetime | None) -> tuple[datetime, datetime]:
    """Clamp the requested window to a bounded, forward-ordered range.

    A summary must never scan an unbounded log: absent a start we look back
    ``_DEFAULT_SUMMARY_LOOKBACK``; a span wider than ``_MAX_SUMMARY_SPAN`` has its
    start pulled forward so the aggregates stay bounded by the timestamp index.

    An offset-less ISO datetime (which the query params advertise as valid) parses
    to a naive value; ``now(UTC)`` is aware. Comparing or subtracting the two would
    raise, so naive bounds are assumed UTC and made aware first.
    """
    if start_date is not None and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date is not None and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)
    end = end_date or datetime.now(UTC)
    start = start_date if start_date is not None else end - _DEFAULT_SUMMARY_LOOKBACK
    if start > end:
        start = end
    if end - start > _MAX_SUMMARY_SPAN:
        start = end - _MAX_SUMMARY_SPAN
    return start, end


def _bucket_expr(dialect_name: str, bucket: Bucket, column: Any = None) -> Any:
    """A SQL expression that truncates ``column`` to the bucket start, in UTC.

    Defaults to ``usage_logs.timestamp``; ``column`` lets another timestamped
    table (``agent_telemetry``) bucket on the same grid, so a chart built from
    both lines up. PostgreSQL ``date_trunc`` honors the session ``TimeZone``, so we
    pin UTC with ``AT TIME ZONE 'UTC'`` (``func.timezone``) rather than trusting
    engine config, otherwise buckets would silently shift per deployment and break
    across DST. SQLite ``strftime`` already normalizes any stored offset to UTC.
    ``bucket`` is a validated ``Literal`` (never raw client text), so there is no
    injection surface.
    """
    timestamp = UsageLog.timestamp if column is None else column
    if dialect_name == "sqlite":
        fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
        return func.strftime(fmt, timestamp)
    # PostgreSQL (and anything else that speaks date_trunc).
    return func.date_trunc(bucket, func.timezone("UTC", timestamp))


def _canonical_bucket(value: Any, bucket: Bucket) -> str:
    """Normalize a bucket key to canonical ISO-8601 UTC (``YYYY-MM-DDTHH:00:00Z``).

    SQLite already returns that string; PostgreSQL returns a (naive, UTC) datetime.
    """
    if isinstance(value, str):
        return value
    dt: datetime = value
    fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
    return dt.strftime(fmt)


def _dialect_name(db: AsyncSession) -> str:
    bind = db.get_bind()
    return bind.dialect.name


def _request_count_expr(status_filter: str | None = None) -> Any:
    """Count requests, not rows.

    Used by every "request count" in this module (totals, dimension breakdowns, and
    the grouped series) so the number means one thing everywhere and the breakdowns
    still sum to the total. The row-count endpoint (`/v1/usage/count`) deliberately
    does not use it: that one paginates the activity list, where an absorbed attempt
    is a row the operator can see and page through.

    A request served through a routing policy can write more than one row: the
    attempt that served it, plus one ``status="absorbed"`` row per failure the
    policy recovered from. Those extra rows describe attempts within a request that
    is already counted, so a plain ``count()`` would inflate request volume and
    deflate every rate computed against it (error rate, cost per request).

    ``status_filter`` is the caller's ``status`` filter, and ``"absorbed"`` is the
    one value that inverts the rule: every row in scope is then an excluded one, so
    the sum would report 0 requests beside non-zero cost and tokens, which reads as
    a bug rather than as a definition. Filtering *to* the attempts makes them the
    unit being asked about, so count rows.
    """
    if status_filter == "absorbed":
        return func.count()
    return func.coalesce(func.sum(case((UsageLog.status != "absorbed", 1), else_=0)), 0)


def _billed_expr(meter: str, fallback: Any) -> Any:
    """A per-row billed token meter, as a summable SQL expression.

    Providers disagree on whether cache tokens are counted inside
    ``prompt_tokens`` (see ``services/metered_pricing.billable_usage``), so raw
    column sums cannot be split into a billed composition. The pricing writers
    resolve that into ``billing_meters`` when a row is priced; prefer that, and
    fall back to the raw column under the subset convention for meterless rows,
    the same fallback the dashboard's per-row token bar applies. JSON extraction
    of a missing key and a NULL column both yield NULL, so the fallback chain
    covers both, ending at 0. ``as_integer`` compiles to the dialect's JSON
    number cast on SQLite and PostgreSQL alike.

    Cost note: ``billing_meters`` is a plain ``json`` column, so on PostgreSQL
    every extraction re-parses the row's text, and the summary runs several
    aggregate passes per render. If profiling shows this biting on large
    windows, the escapes are a ``jsonb`` migration or persisting the billed
    totals as real columns at write time; the fallback semantics here would be
    unchanged by either.
    """
    return func.coalesce(UsageLog.billing_meters[meter].as_integer(), fallback, 0)


def _needs_pricing_expr() -> ColumnElement[bool]:
    """Predicate for "this row still needs pricing".

    Two ways to qualify. Either nothing was charged at all (``cost IS NULL``, how it
    has always been expressed), or the row carries a cost that came *only* from
    gateway-run tool calls while its tokens were never metered. The second case
    exists because a request against an unpriced model can still owe for the searches
    it ran, and charging it a tool cost would otherwise hide it from the exact view an
    operator uses to find what needs a rate.

    The tool-namespace test keeps this narrow on purpose: a row with a cost and no
    meters at all is a row priced before the meter columns existed, and it must keep
    reading as priced.
    """
    token_metered = UsageLog.billing_meters["total_input_tokens"].as_integer().is_not(None)
    tool_charged = UsageLog.billing_meters[TOOL_METER_NAMESPACE].as_string().is_not(None)
    return or_(UsageLog.cost.is_(None), and_(tool_charged, ~token_metered))


def _tool_calls_expr(tool: str) -> Any:
    """Billable call count for one gateway-run tool on a row, or NULL."""
    return UsageLog.billing_meters[(TOOL_METER_NAMESPACE, tool, "billed")].as_integer()


def _tool_cost_expr(tool: str) -> Any:
    """USD charged for one tool on a row, read off its own charge line.

    The row's ``cost`` mixes tokens and tools, so per-tool spend has to come from
    the line ``price_tool_calls`` wrote. ``pricing_breakdown`` is a JSON *array*,
    and the tool lines are appended after the token ones, so the position is not
    fixed; the units and the rate are both on the line, so the product is
    reconstructed from the meter count instead, which needs no array search.
    """
    return _tool_calls_expr(tool) * _tool_unit_rate_expr(tool)


def _tool_unit_rate_expr(tool: str) -> Any:
    """Per-call USD rate recorded on the row for one tool.

    Stored per row rather than looked up live, so a historical row keeps the rate
    it was actually billed at even after the operator changes the price.
    """
    return func.coalesce(
        UsageLog.billing_meters[(TOOL_METER_NAMESPACE, tool, "unit_rate")].as_float(),
        0.0,
    )


def _tool_used_expr(tool: str) -> ColumnElement[bool]:
    """Predicate for "this request ran a gateway tool".

    ``any`` tests the namespace itself so an MCP tool (whose name is supplied by the
    caller's server and cannot be enumerated here) still matches. A specific name
    indexes into the namespace. Both compile to the dialect's JSON extraction on
    SQLite and PostgreSQL; neither is indexable, which is acceptable because every
    activity query is already bounded by the indexed timestamp window.
    """
    if tool == _ANY_TOOL:
        # ``.as_string()`` is load-bearing: an uncoerced JSON index compares with JSON
        # semantics, where SQL NULL and JSON null are not the same thing, and
        # ``IS NOT NULL`` then matches every row. Coercing to text makes a missing key
        # read as SQL NULL on both SQLite and PostgreSQL.
        namespace = UsageLog.billing_meters[TOOL_METER_NAMESPACE].as_string()
        return cast("ColumnElement[bool]", namespace.is_not(None))
    return cast("ColumnElement[bool]", _tool_calls_expr(tool).is_not(None))


# The billed composition aggregates shared by the summary series, the grand
# totals, and the grouped series, so all three reconcile by construction.
def _billed_input_sum() -> Any:
    return func.coalesce(func.sum(_billed_expr("total_input_tokens", UsageLog.prompt_tokens)), 0)


def _billed_output_sum() -> Any:
    return func.coalesce(func.sum(_billed_expr("completion_tokens", UsageLog.completion_tokens)), 0)


async def _totals(
    db: AsyncSession, conditions: list[ColumnElement[bool]], status_filter: str | None = None
) -> UsageTotals:
    row = (
        await db.execute(
            select(
                func.coalesce(func.sum(UsageLog.cost), 0.0),
                func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
                func.coalesce(func.sum(UsageLog.completion_tokens), 0),
                func.coalesce(func.sum(UsageLog.total_tokens), 0),
                func.coalesce(func.sum(UsageLog.cache_read_tokens), 0),
                func.coalesce(func.sum(UsageLog.cache_write_tokens), 0),
                func.coalesce(func.sum(UsageLog.cache_write_1h_tokens), 0),
                _request_count_expr(status_filter),
                func.coalesce(func.sum(case((UsageLog.status == "error", 1), else_=0)), 0),
                # Averaged over requests, not attempts: an absorbed row carries the
                # time spent on a candidate that did not serve, and folding it in
                # would make a policy that recovers quickly look slower than one
                # that never fails.
                func.avg(case((UsageLog.status != "absorbed", UsageLog.latency_ms))),
                # Unpriced *served* rows only; see UsageTotals.unpriced_requests. The
                # predicate is shared with the list filter so the tile and the rows it
                # sends an operator to agree.
                func.coalesce(
                    func.sum(case(((UsageLog.status == "success") & _needs_pricing_expr(), 1), else_=0)),
                    0,
                ),
                _billed_input_sum(),
                _billed_output_sum(),
            ).where(*conditions)
        )
    ).one()
    return UsageTotals(
        cost=float(row[0]),
        prompt_tokens=int(row[1]),
        completion_tokens=int(row[2]),
        total_tokens=int(row[3]),
        cache_read_tokens=int(row[4]),
        cache_write_tokens=int(row[5]),
        cache_write_1h_tokens=int(row[6]),
        request_count=int(row[7]),
        error_count=int(row[8]),
        avg_latency_ms=float(row[9]) if row[9] is not None else None,
        unpriced_requests=int(row[10]),
        billed_input_tokens=int(row[11]),
        billed_output_tokens=int(row[12]),
    )


async def _breakdown(
    db: AsyncSession,
    column: Any,
    conditions: list[ColumnElement[bool]],
    totals: UsageTotals,
    *,
    limit: int | None,
    status_filter: str | None = None,
    label_join: "_LabelJoin | None" = None,
) -> list[UsageGroupRow]:
    """Spend/tokens/requests grouped by ``column``, biggest spend first.

    ``tokens`` is the *billed* total (input including both cache buckets, plus
    output, via ``_billed_expr``), the same quantity the series composition and
    the grouped series report, so every analytics surface agrees on what a
    token count means. When ``limit`` is set, only the top rows are returned and
    the remainder is folded into a synthesized ``other`` row derived from the
    grand totals, so the breakdown always reconciles with the tiles.
    ``limit=None`` returns every group (used by the CSV export, which must not
    truncate).
    """
    cost_sum = func.coalesce(func.sum(UsageLog.cost), 0.0)
    # The label rides along in the same pass rather than costing a second query or
    # a client-side table dump. Grouped by as well as selected: it is functionally
    # dependent on the joined row's primary key, but only PostgreSQL infers that,
    # and this has to run on SQLite too. Outer-joined so a group whose entity was
    # deleted keeps its row (with a null label) instead of vanishing from a
    # breakdown that must still reconcile against the totals.
    label_column = label_join.label if label_join is not None else null()
    stmt = select(
        column,
        label_column,
        cost_sum,
        _billed_input_sum() + _billed_output_sum(),
        _request_count_expr(status_filter),
    )
    if label_join is not None:
        stmt = stmt.outerjoin(label_join.entity, label_join.on)
    group_by = (column,) if label_join is None else (column, label_column)
    stmt = stmt.where(*conditions).group_by(*group_by).order_by(cost_sum.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    rows = (await db.execute(stmt)).all()
    result = [
        UsageGroupRow(key=row[0], label=row[1], cost=float(row[2]), tokens=int(row[3]), requests=int(row[4]))
        for row in rows
    ]
    if limit is not None:
        seen_requests = sum(r.requests for r in result)
        # request_count is an exact integer, so a positive residual is the reliable
        # signal that groups were folded; cost/tokens residuals follow from totals.
        residual_requests = totals.request_count - seen_requests
        if residual_requests > 0:
            billed_total = totals.billed_input_tokens + totals.billed_output_tokens
            result.append(
                UsageGroupRow(
                    key=None,
                    cost=totals.cost - sum(r.cost for r in result),
                    tokens=billed_total - sum(r.tokens for r in result),
                    requests=residual_requests,
                    is_other=True,
                )
            )
    return result


def error_class_for(status_code: int | None) -> ErrorClass:
    """Coarse display bucket for a failure's HTTP status code.

    Two kinds of code reach this column: the status a provider returned, and the
    status the gateway itself refused with, since #465 records those rejections
    too and each row carries the code it returned (403 for a blocked or
    over-budget user, a user/key mismatch, or a model outside a key's allow-list,
    402 for missing pricing, 400 for a selector that no longer resolves).

    So ``auth`` currently covers both a provider rejecting the gateway's
    credentials and the gateway rejecting the caller, and a **budget denial files
    as ``auth``**, because ``reserve_budget`` refuses with 403 and the code is all
    this function sees. Splitting budget and permission refusals into their own
    class needs a discriminator the row does not reliably carry (``provider`` is
    NULL only on the gates that refuse before the selector resolves, not on the
    budget gate), and these names are dashboard-visible, so that stays a
    deliberate follow-up rather than something guessed at here.
    """
    if status_code is None:
        return "unknown"
    if status_code == 402:
        return "pricing"
    if status_code == 429:
        return "rate_limit"
    if status_code in (401, 403, 407):
        return "auth"
    if 500 <= status_code <= 599:
        return "provider_error"
    if 400 <= status_code <= 499:
        return "client_error"
    return "unknown"


async def _tool_breakdown(
    db: AsyncSession,
    conditions: list[ColumnElement[bool]],
) -> list[UsageToolRow]:
    """Per-tool calls, failures, spend, and requests inside the window.

    One aggregate per known tool rather than a ``GROUP BY`` over the meter map:
    a JSON map's keys cannot be grouped portably across SQLite and PostgreSQL
    (``json_each`` versus ``jsonb_each``, on a plain ``json`` column), and the set
    of gateway-run tools is small and fixed. Cost comes from each row's charge
    line rather than the row total, because a row's ``cost`` also carries tokens.

    Rows with no calls for a tool contribute nothing, so a deployment that never
    ran a tool gets an empty list and the UI can hide the section entirely.

    ``calls`` and ``errors`` count every call a request made, including calls made by
    an attempt a routing policy later abandoned: the tally is shared across a
    request's attempts and settled onto the row that served, so those calls are on
    that one row rather than spread across the absorbed ones. ``requests`` counts
    requests, so the two answer different questions on purpose: "how much tool work
    did we do" and "how many requests used a tool".
    """
    out: list[UsageToolRow] = []
    for tool in GATEWAY_TOOL_NAMES:
        calls = _tool_calls_expr(tool)
        errors = UsageLog.billing_meters[(TOOL_METER_NAMESPACE, tool, "errors")].as_integer()
        row = (
            await db.execute(
                select(
                    func.coalesce(func.sum(calls), 0),
                    func.coalesce(func.sum(errors), 0),
                    # Requests, not rows: a request that failed over through a
                    # routing policy writes an absorbed row per recovered attempt,
                    # and counting those would report more requests using a tool
                    # than the Activity list shows for the same filter.
                    _request_count_expr(),
                    func.coalesce(func.sum(_tool_cost_expr(tool)), 0.0),
                ).where(*conditions, calls.is_not(None))
            )
        ).one()
        if not row[0] and not row[1]:
            continue
        out.append(
            UsageToolRow(
                tool=tool,
                calls=int(row[0]),
                errors=int(row[1]),
                requests=int(row[2]),
                cost=float(row[3]),
            )
        )
    return sorted(out, key=lambda r: r.cost, reverse=True)


async def _errors_by_status_code(
    db: AsyncSession,
    conditions: list[ColumnElement[bool]],
) -> list[UsageErrorCodeRow]:
    """Failures in the window grouped by status code, most frequent first.

    The whole point of the column: this is a GROUP BY rather than substring
    matching over provider-specific error prose. Capped at ``_BREAKDOWN_TOP_N``
    distinct codes, which no real window reaches (HTTP has far fewer), so unlike
    the cost breakdowns there is no synthesized fold row.
    """
    request_count = func.count()
    rows = (
        await db.execute(
            select(UsageLog.status_code, request_count)
            .where(*conditions, UsageLog.status == "error")
            .group_by(UsageLog.status_code)
            .order_by(request_count.desc())
            .limit(_BREAKDOWN_TOP_N)
        )
    ).all()
    return [
        UsageErrorCodeRow(
            status_code=row[0],
            error_class=error_class_for(row[0]),
            requests=int(row[1]),
        )
        for row in rows
    ]


async def _summary_context(
    db: AsyncSession,
    *,
    start_date: datetime | None,
    end_date: datetime | None,
    user_id: list[str] | None,
    status: str | None,
    model: list[str] | None,
    endpoint: str | None,
    provider: str | None = None,
    source: str | None = None,
    source_label: str | None = None,
    api_key_id: list[str] | None = None,
    priced: bool | None = None,
    tool: str | None = None,
    counts_toward_budget: bool | None = None,
    status_code: int | None = None,
) -> tuple[datetime, datetime, list[ColumnElement[bool]], UsageTotals]:
    """Resolve the bounded window, the shared WHERE conditions, and the grand
    totals: the common preamble both summary endpoints run, kept in one place so a
    fix (like the naive-datetime handling in ``_resolve_window``) lands once.
    """
    start, end = _resolve_window(start_date, end_date)
    conditions = _usage_filters(
        start_date=start,
        end_date=end,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
    )
    totals = await _totals(db, conditions, status)
    return start, end, conditions, totals


# Upper bound on zero-filled series points, so a pathological range/bucket combo
# (e.g. hourly over a year) cannot balloon the payload; beyond it the endpoint
# returns the sparse populated buckets instead.
_MAX_SERIES_POINTS = 1000

_PointT = TypeVar("_PointT")


def _empty_usage_point(bucket_start: str) -> UsageSeriesPoint:
    return UsageSeriesPoint(bucket_start=bucket_start, cost=0.0, tokens=0, requests=0)


def _dense_series(
    start: datetime,
    end: datetime,
    bucket: Bucket,
    populated: dict[str, _PointT],
    empty: Callable[[str], _PointT] | None = None,
) -> list[_PointT]:
    """Fill every bucket in ``[floor(start), end)`` so the chart's x-axis is linear
    in time. ``GROUP BY`` omits empty buckets, so without this a sparse range (say
    usage on day 1 and day 20 of a month) would render as two adjacent bars and
    misread the trend. Falls back to the sparse buckets past ``_MAX_SERIES_POINTS``.
    An empty window (no rows at all) returns an empty series, not a wall of zeros.

    ``empty`` builds the zero point for a gap; it is what lets another series type
    (the agent-telemetry summary's) share this fill rather than restate it.
    """
    if not populated:
        return []
    make_empty = cast("Callable[[str], _PointT]", empty or _empty_usage_point)
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
    if bucket == "hour":
        cursor = start.replace(minute=0, second=0, microsecond=0)
        fmt = "%Y-%m-%dT%H:00:00Z"
    else:
        cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
        fmt = "%Y-%m-%dT00:00:00Z"
    points: list[_PointT] = []
    while cursor < end:
        if len(points) >= _MAX_SERIES_POINTS:
            return [populated[key] for key in sorted(populated)]
        key = cursor.strftime(fmt)
        points.append(populated.get(key) or make_empty(key))
        cursor += step
    return points


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def usage_summary(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)
    ] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
    dimensions: list[SummaryDimension] | None = Query(default=None, description=_DIMENSIONS_DESC),
) -> UsageSummary:
    """Aggregate spend, tokens, and request volume for the dashboard Usage page.

    Range-bounded (default last 30 days, hard-capped): unlike the raw ``/v1/usage``
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
    start, end, conditions, totals = await _summary_context(
        db,
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
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
    )
    # ``none`` is dropped rather than rejected: it exists only so a caller can send
    # an empty selection, and it never contributes a dimension of its own.
    requested: set[str] = _ALL_SUMMARY_DIMENSIONS if dimensions is None else {d for d in dimensions if d != "none"}
    breakdowns = {
        name: await _breakdown(db, column, conditions, totals, limit=cap, status_filter=status, label_join=label)
        for name, (column, cap, label) in _SUMMARY_DIMENSIONS.items()
        if name in requested
    }
    # The failure taxonomy is a GROUP BY pass like the others, so it answers to the
    # same selector rather than being charged to every caller: the tiles and the
    # timeline ask for no dimensions at all.
    errors_by_status_code = (
        await _errors_by_status_code(db, conditions) if _ERROR_TAXONOMY_DIMENSION in requested else []
    )
    by_tool = await _tool_breakdown(db, conditions) if _TOOL_DIMENSION in requested else []

    expr = _bucket_expr(_dialect_name(db), bucket)
    series_rows = (
        await db.execute(
            select(
                expr,
                func.coalesce(func.sum(UsageLog.cost), 0.0),
                func.coalesce(func.sum(UsageLog.total_tokens), 0),
                _request_count_expr(status),
                func.coalesce(func.sum(case((UsageLog.status == "error", 1), else_=0)), 0),
                _billed_input_sum(),
                func.coalesce(func.sum(_billed_expr("cache_read_tokens", UsageLog.cache_read_tokens)), 0),
                func.coalesce(func.sum(_billed_expr("cache_write_tokens", UsageLog.cache_write_tokens)), 0),
                _billed_output_sum(),
            )
            .where(*conditions)
            .group_by(expr)
        )
    ).all()
    # Zero-fill empty buckets so the chart is time-linear (GROUP BY drops gaps).
    populated = {
        _canonical_bucket(row[0], bucket): UsageSeriesPoint(
            bucket_start=_canonical_bucket(row[0], bucket),
            cost=float(row[1]),
            tokens=int(row[2]),
            requests=int(row[3]),
            errors=int(row[4]),
            input_tokens=int(row[5]),
            cache_read_tokens=int(row[6]),
            cache_write_tokens=int(row[7]),
            output_tokens=int(row[8]),
        )
        for row in series_rows
    }
    series = _dense_series(start, end, bucket, populated)

    return UsageSummary(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        bucket=bucket,
        totals=totals,
        by_model=breakdowns.get("model", []),
        by_user=breakdowns.get("user", []),
        by_api_key=breakdowns.get("api_key", []),
        by_source=breakdowns.get("source", []),
        by_source_label=breakdowns.get("source_label", []),
        by_endpoint=breakdowns.get("endpoint", []),
        by_provider=breakdowns.get("provider", []),
        by_tool=by_tool,
        errors_by_status_code=errors_by_status_code,
        series=series,
    )


_GROUP_COLUMNS: dict[str, tuple[Any, "_LabelJoin | None"]] = {
    "model": (UsageLog.model, None),
    "user_id": (UsageLog.user_id, _USER_LABEL),
    "api_key_id": (UsageLog.api_key_id, _API_KEY_LABEL),
    "source": (UsageLog.source, None),
}


@router.get("/series", dependencies=[Depends(verify_master_key)])
async def usage_series(
    db: Annotated[AsyncSession, Depends(get_db)],
    group_by: SeriesGroupBy = Query(description="Dimension to split the series by"),
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)
    ] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
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
    start, end, conditions, totals = await _summary_context(
        db,
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
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
    )
    # Finding-5 guard: /summary densifies then caps at _MAX_SERIES_POINTS; this
    # endpoint is sparse, so cap the bucket *grid* instead (hourly over the
    # 366-day max window would otherwise be ~8.8k buckets x 10 groups per call).
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
    if (end - start) / step > _MAX_SERIES_POINTS:
        raise HTTPException(
            status_code=422,
            detail=f"window spans more than {_MAX_SERIES_POINTS} {bucket} buckets; use bucket=day or narrow the range",
        )
    column, label_join = _GROUP_COLUMNS[group_by]
    groups = await _breakdown(
        db, column, conditions, totals, limit=_SERIES_TOP_N, status_filter=status, label_join=label_join
    )

    # One grouped query for the whole grid: groups outside the top N collapse
    # into the fold in SQL rather than being fetched and folded here, so the row
    # count stays bounded by buckets × (top N + 2) regardless of cardinality.
    # The synthesized groups are encoded as (key NULL, fold flag) rather than a
    # sentinel key string: GROUP BY treats NULLs as equal on both dialects, and
    # no sentinel can be trusted never to collide with a real key. A NULL column
    # value never matches ``IN``, so it lands in the CASE's ``else`` arm; the
    # fold flag then separates a NULL group that ranked in the top N (a real
    # ``key=None`` series, e.g. a deleted user) from the past-top-N remainder.
    named = {g.key for g in groups if g.key is not None}
    keeps_null = any(g.key is None and not g.is_other for g in groups)
    key_expr = case((column.in_(named), column), else_=null())
    if keeps_null:
        fold_expr = case((column.is_(None), 0), (column.in_(named), 0), else_=1)
    else:
        fold_expr = case((column.in_(named), 0), else_=1)
    bucket_expr = _bucket_expr(_dialect_name(db), bucket)
    rows = (
        await db.execute(
            select(
                bucket_expr,
                key_expr,
                fold_expr,
                func.coalesce(func.sum(UsageLog.cost), 0.0),
                _billed_input_sum() + _billed_output_sum(),
                _request_count_expr(status),
            )
            .where(*conditions)
            .group_by(bucket_expr, key_expr, fold_expr)
        )
    ).all()

    points = [
        UsageGroupedSeriesPoint(
            bucket_start=_canonical_bucket(row[0], bucket),
            key=row[1],
            is_other=bool(row[2]),
            cost=float(row[3]),
            tokens=int(row[4]),
            requests=int(row[5]),
        )
        for row in rows
    ]
    points.sort(key=lambda p: p.bucket_start)

    return UsageGroupedSeries(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        bucket=bucket,
        group_by=group_by,
        groups=groups,
        points=points,
    )


# Leading characters a spreadsheet may interpret as a formula. Any cell starting
# with one is prefixed with a single quote so opening the CSV in Excel/Sheets can
# never execute attacker-influenced text (model / user ids are caller-supplied).
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# The export names a dimension the way an operator reads it where that differs from
# the column name the API selector uses.
_CSV_DIMENSION_LABELS = {"source_label": "session"}


def _csv_safe(value: str) -> str:
    if value and value[0] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


@router.get("/summary.csv", dependencies=[Depends(verify_master_key)])
async def usage_summary_csv(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)
    ] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
) -> Response:
    """Download every breakdown the summary reports, as one CSV.

    One row per (dimension, key): model, user, API key, source, session
    (``source_label``), endpoint, and provider. A dedicated route rather than a
    ``format=csv`` flag on ``/summary`` so that endpoint keeps a single JSON
    response model and a clean OpenAPI schema. The export is **uncapped** (no
    top-N fold): finance wants every row. ``tokens`` is the billed total (fresh
    input, both cache buckets, and output), matching the dashboard's analytics.
    Kept separate from the bare-array ``/v1/usage`` contract, which is untouched.
    """
    _start, _end, conditions, totals = await _summary_context(
        db,
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
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
    )
    # Driven off the same dimension table as ``/summary`` so a new breakdown lands
    # in the export without a second edit here.
    dimensions = [
        (
            _CSV_DIMENSION_LABELS.get(name, name),
            await _breakdown(db, column, conditions, totals, limit=None, status_filter=status),
        )
        for name, (column, _cap, _label) in _SUMMARY_DIMENSIONS.items()
    ]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["dimension", "key", "cost", "tokens", "requests"])
    for dimension, rows in dimensions:
        for row in rows:
            writer.writerow([dimension, _csv_safe(row.key or ""), f"{row.cost:.6f}", row.tokens, row.requests])

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="usage-summary.csv"'},
    )
