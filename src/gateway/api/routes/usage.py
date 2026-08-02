"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

import csv
import io
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel
from sqlalchemy import ColumnElement, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, UsageLog
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

router = APIRouter(prefix="/v1/usage", tags=["usage"])

# The analytics summary is range-bounded, unlike the raw list. Absent a start_date
# it looks back this far; a wider explicit window is clamped to the hard cap so a
# single request can never turn into an unbounded full-table scan on a growing log.
_DEFAULT_SUMMARY_LOOKBACK = timedelta(days=30)
_MAX_SUMMARY_SPAN = timedelta(days=366)

# How many rows each breakdown returns before the remainder is folded into a
# single synthesized "other" row (so the tables still reconcile with the totals).
_BREAKDOWN_TOP_N = 100

# Sessions (``source_label``) are an order of magnitude higher-cardinality than
# models or users: one agent workload can open hundreds of them in a month, and
# the interesting signal is a long-ish head ("which tasks burned the budget"),
# not just the top few. Give that dimension a deeper cap so the head is not
# swallowed by the "other" fold.
_SESSION_BREAKDOWN_TOP_N = 250

Bucket = Literal["hour", "day"]

# Coarse display buckets for a failure's status code. A closed Literal rather than
# a bare str so the set lands in the OpenAPI schema as an enum and a consumer can
# switch on it exhaustively instead of string-matching whatever the server sent.
ErrorClass = Literal["pricing", "rate_limit", "auth", "provider_error", "client_error", "unknown"]

# Every breakdown ``/summary`` can compute, mapped to the column it groups by and
# its top-N cap. A dimension name is the ``by_<name>`` response field it fills, so
# a caller reads the selector and the payload with one vocabulary.
_SUMMARY_DIMENSIONS: dict[str, tuple[Any, int]] = {
    "model": (UsageLog.model, _BREAKDOWN_TOP_N),
    "user": (UsageLog.user_id, _BREAKDOWN_TOP_N),
    "api_key": (UsageLog.api_key_id, _BREAKDOWN_TOP_N),
    "source": (UsageLog.source, _BREAKDOWN_TOP_N),
    "source_label": (UsageLog.source_label, _SESSION_BREAKDOWN_TOP_N),
    "endpoint": (UsageLog.endpoint, _BREAKDOWN_TOP_N),
    "provider": (UsageLog.provider, _BREAKDOWN_TOP_N),
}

# The failure taxonomy (``errors_by_status_code``) is a GROUP BY pass like the
# breakdowns above, but it groups failures by status code rather than spend by a
# dimension, so it is selectable by name without living in _SUMMARY_DIMENSIONS.
# It is the one dimension whose response field is not ``by_<name>``.
_ERROR_TAXONOMY_DIMENSION = "status_code"

# Keep in step with _SUMMARY_DIMENSIONS; the extra ``none`` is the explicit empty
# selection (a repeated query param cannot express an empty list on the wire).
SummaryDimension = Literal[
    "model", "user", "api_key", "source", "source_label", "endpoint", "provider", "status_code", "none"
]

_ALL_SUMMARY_DIMENSIONS: set[str] = set(_SUMMARY_DIMENSIONS) | {_ERROR_TAXONOMY_DIMENSION}


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
    api_key_id: str | None
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
    billing_meters: dict[str, Any] | None
    pricing_breakdown: list[dict[str, float | int | str]] | None
    cost: float | None
    status: str
    error_message: str | None
    status_code: int | None
    latency_ms: int | None
    source: str
    source_label: str | None
    counts_toward_budget: bool

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageEntry":
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
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
        )


class UsageCount(BaseModel):
    """Total number of usage logs matching a set of filters."""

    total: int


# Shared query descriptions so the list and count endpoints stay in lockstep.
_START_DESC = "Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)"
_END_DESC = "Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)"
_USER_DESC = "Filter to a single user"
_STATUS_DESC = "Filter to a single status (e.g. 'success' or 'error')"
_STATUS_CODE_DESC = (
    "Filter to a single failure status code (e.g. 429 for provider rate limits, "
    "402 for missing-pricing rejections). Only error rows carry one"
)
_MODEL_DESC = "Filter to a single model"
_ENDPOINT_DESC = "Filter to a single endpoint (e.g. '/v1/chat/completions')"
_PROVIDER_DESC = "Filter to a single provider (e.g. 'openai')"
_SOURCE_DESC = "Filter to a single provenance source (e.g. 'gateway' or 'claude_code')"
_SOURCE_LABEL_DESC = "Filter to a single session/project label (the source_label carried by imported usage)"
_API_KEY_DESC = "Filter to a single API key id"
_PRICED_DESC = "Filter by pricing state: true = only rows with a cost, false = only unpriced rows (cost is null)"
_COUNTS_DESC = (
    "Filter by budget participation: true = only enforced gateway rows, "
    "false = only imported rows that never touch a budget"
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
    user_id: str | None,
    status: str | None,
    model: str | None,
    endpoint: str | None,
    provider: str | None = None,
    source: str | None = None,
    source_label: str | None = None,
    api_key_id: str | None = None,
    priced: bool | None = None,
    counts_toward_budget: bool | None = None,
    status_code: int | None = None,
) -> list[ColumnElement[bool]]:
    """Build the shared WHERE conditions for the list and count endpoints.

    Keeping this in one place guarantees the paginator's total (``/count``)
    always matches the rows ``list_usage`` returns for the same filters.
    """
    conditions: list[ColumnElement[bool]] = []
    if start_date is not None:
        conditions.append(UsageLog.timestamp >= start_date)
    if end_date is not None:
        conditions.append(UsageLog.timestamp < end_date)
    if user_id is not None:
        conditions.append(UsageLog.user_id == user_id)
    if status is not None:
        conditions.append(UsageLog.status == status)
    if status_code is not None:
        conditions.append(UsageLog.status_code == status_code)
    if model is not None:
        conditions.append(UsageLog.model == model)
    if endpoint is not None:
        conditions.append(UsageLog.endpoint == endpoint)
    if provider is not None:
        conditions.append(UsageLog.provider == provider)
    if source is not None:
        conditions.append(UsageLog.source == source)
    if source_label is not None:
        conditions.append(UsageLog.source_label == source_label)
    if api_key_id is not None:
        conditions.append(UsageLog.api_key_id == api_key_id)
    if priced is True:
        conditions.append(UsageLog.cost.is_not(None))
    elif priced is False:
        conditions.append(UsageLog.cost.is_(None))
    if counts_toward_budget is not None:
        conditions.append(UsageLog.counts_toward_budget.is_(counts_toward_budget))
    return conditions


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: str | None = Query(default=None, description=_API_KEY_DESC),
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, status, failure status code,
    model, endpoint, provider, source, and session (``source_label``).
    Paginated via skip/limit. The return shape is a bare JSON array; external
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
        counts_toward_budget=counts_toward_budget,
    )
    stmt = (
        select(UsageLog)
        .where(*conditions)
        .order_by(UsageLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [UsageEntry.from_model(log) for log in logs]


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
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: str | None = Query(default=None, description=_API_KEY_DESC),
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
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
        counts_toward_budget=counts_toward_budget,
    )
    stmt: Any = select(func.count()).select_from(UsageLog).where(*conditions)
    total = (await db.execute(stmt)).scalar_one()
    return UsageCount(total=total)


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


class UsageGroupRow(BaseModel):
    """One breakdown row (a model, a user, an API key, a session, ...).

    ``key`` is None both for the synthesized fold row (``is_other=True``) and for a
    real group whose column was NULL (e.g. usage from a since-deleted user, with
    ``is_other=False``). ``is_other`` disambiguates the two so the UI does not
    mislabel deleted-user usage as the fold.
    """

    key: str | None
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
    identical across SQLite and PostgreSQL for the same underlying instant."""

    bucket_start: str
    cost: float
    tokens: int
    requests: int


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
    # Failures only, so the taxonomy is not swamped by the successes that carry
    # no status code. Counts sum to ``totals.error_count``, unless a window
    # somehow held more than ``_BREAKDOWN_TOP_N`` distinct codes, in which case
    # the tail is omitted rather than folded (there is no synthesized "other"
    # row: a null key would collide with the real "no code recorded" group).
    errors_by_status_code: list[UsageErrorCodeRow]
    series: list[UsageSeriesPoint]


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


def _bucket_expr(dialect_name: str, bucket: Bucket) -> Any:
    """A SQL expression that truncates ``timestamp`` to the bucket start, in UTC.

    PostgreSQL ``date_trunc`` honors the session ``TimeZone``, so we pin UTC with
    ``AT TIME ZONE 'UTC'`` (``func.timezone``) rather than trusting engine config,
    otherwise buckets would silently shift per deployment and break across DST.
    SQLite ``strftime`` already normalizes any stored offset to UTC. ``bucket`` is
    a validated ``Literal`` (never raw client text), so there is no injection surface.
    """
    if dialect_name == "sqlite":
        fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
        return func.strftime(fmt, UsageLog.timestamp)
    # PostgreSQL (and anything else that speaks date_trunc).
    return func.date_trunc(bucket, func.timezone("UTC", UsageLog.timestamp))


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


async def _totals(db: AsyncSession, conditions: list[ColumnElement[bool]]) -> UsageTotals:
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
                func.count(),
                func.coalesce(func.sum(case((UsageLog.status == "error", 1), else_=0)), 0),
                func.avg(UsageLog.latency_ms),
                # Unpriced *served* rows only; see UsageTotals.unpriced_requests.
                func.coalesce(
                    func.sum(case(((UsageLog.status == "success") & UsageLog.cost.is_(None), 1), else_=0)),
                    0,
                ),
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
    )


async def _breakdown(
    db: AsyncSession,
    column: Any,
    conditions: list[ColumnElement[bool]],
    totals: UsageTotals,
    *,
    limit: int | None,
) -> list[UsageGroupRow]:
    """Spend/tokens/requests grouped by ``column``, biggest spend first.

    When ``limit`` is set, only the top rows are returned and the remainder is
    folded into a synthesized ``other`` row derived from the grand totals, so the
    breakdown always reconciles with the tiles. ``limit=None`` returns every group
    (used by the CSV export, which must not truncate).
    """
    cost_sum = func.coalesce(func.sum(UsageLog.cost), 0.0)
    stmt = (
        select(
            column,
            cost_sum,
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.count(),
        )
        .where(*conditions)
        .group_by(column)
        .order_by(cost_sum.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    rows = (await db.execute(stmt)).all()
    result = [
        UsageGroupRow(key=row[0], cost=float(row[1]), tokens=int(row[2]), requests=int(row[3])) for row in rows
    ]
    if limit is not None:
        seen_requests = sum(r.requests for r in result)
        # request_count is an exact integer, so a positive residual is the reliable
        # signal that groups were folded; cost/tokens residuals follow from totals.
        residual_requests = totals.request_count - seen_requests
        if residual_requests > 0:
            result.append(
                UsageGroupRow(
                    key=None,
                    cost=totals.cost - sum(r.cost for r in result),
                    tokens=totals.total_tokens - sum(r.tokens for r in result),
                    requests=residual_requests,
                    is_other=True,
                )
            )
    return result


def error_class_for(status_code: int | None) -> ErrorClass:
    """Coarse display bucket for a failure's HTTP status code.

    401 and 403 read as ``auth`` rather than as a budget denial because the codes
    that actually reach this column come from upstream: a provider rejecting the
    gateway's credentials. Budget and blocked-user rejections are refused before
    anything is logged, so they have no row here to classify (see #317).
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
    user_id: str | None,
    status: str | None,
    model: str | None,
    endpoint: str | None,
    provider: str | None = None,
    source: str | None = None,
    source_label: str | None = None,
    api_key_id: str | None = None,
    priced: bool | None = None,
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
        counts_toward_budget=counts_toward_budget,
    )
    totals = await _totals(db, conditions)
    return start, end, conditions, totals


# Upper bound on zero-filled series points, so a pathological range/bucket combo
# (e.g. hourly over a year) cannot balloon the payload; beyond it the endpoint
# returns the sparse populated buckets instead.
_MAX_SERIES_POINTS = 1000


def _dense_series(
    start: datetime,
    end: datetime,
    bucket: Bucket,
    rows: list[tuple[str, float, int, int]],
) -> list[UsageSeriesPoint]:
    """Fill every bucket in ``[floor(start), end)`` so the chart's x-axis is linear
    in time. ``GROUP BY`` omits empty buckets, so without this a sparse range (say
    usage on day 1 and day 20 of a month) would render as two adjacent bars and
    misread the trend. Falls back to the sparse buckets past ``_MAX_SERIES_POINTS``.
    An empty window (no rows at all) returns an empty series, not a wall of zeros.
    """
    if not rows:
        return []
    populated = {key: (cost, tokens, requests) for key, cost, tokens, requests in rows}
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
    if bucket == "hour":
        cursor = start.replace(minute=0, second=0, microsecond=0)
        fmt = "%Y-%m-%dT%H:00:00Z"
    else:
        cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
        fmt = "%Y-%m-%dT00:00:00Z"
    points: list[UsageSeriesPoint] = []
    while cursor < end:
        if len(points) >= _MAX_SERIES_POINTS:
            return [
                UsageSeriesPoint(bucket_start=key, cost=cost, tokens=tokens, requests=requests)
                for key, (cost, tokens, requests) in sorted(populated.items())
            ]
        key = cursor.strftime(fmt)
        cost, tokens, requests = populated.get(key, (0.0, 0, 0))
        points.append(UsageSeriesPoint(bucket_start=key, cost=cost, tokens=tokens, requests=requests))
        cursor += step
    return points


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def usage_summary(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: str | None = Query(default=None, description=_API_KEY_DESC),
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
    dimensions: list[SummaryDimension] | None = Query(default=None, description=_DIMENSIONS_DESC),
) -> UsageSummary:
    """Aggregate spend, tokens, and request volume for the dashboard Usage page.

    Range-bounded (default last 30 days, hard-capped): unlike the raw ``/v1/usage``
    list, every aggregate is scoped to a bounded window so it stays served by the
    timestamp index. Returns grand totals, breakdowns by model / user / API key /
    source / session (``source_label``) / endpoint / provider (top rows plus a
    reconciling ``other`` fold), the error taxonomy grouped by failure status code,
    and a UTC-bucketed time series.

    Each breakdown is its own ``GROUP BY`` pass, so a caller that reads only the
    totals or the series should narrow ``dimensions`` rather than pay for all eight
    (the dashboard's tiles, timeline context, and model typeahead all do). Omitting
    the parameter keeps the full set.
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
        counts_toward_budget=counts_toward_budget,
    )
    # ``none`` is dropped rather than rejected: it exists only so a caller can send
    # an empty selection, and it never contributes a dimension of its own.
    requested: set[str] = _ALL_SUMMARY_DIMENSIONS if dimensions is None else {d for d in dimensions if d != "none"}
    breakdowns = {
        name: await _breakdown(db, column, conditions, totals, limit=cap)
        for name, (column, cap) in _SUMMARY_DIMENSIONS.items()
        if name in requested
    }
    # The failure taxonomy is a GROUP BY pass like the others, so it answers to the
    # same selector rather than being charged to every caller: the tiles and the
    # timeline ask for no dimensions at all.
    errors_by_status_code = (
        await _errors_by_status_code(db, conditions) if _ERROR_TAXONOMY_DIMENSION in requested else []
    )

    expr = _bucket_expr(_dialect_name(db), bucket)
    series_rows = (
        await db.execute(
            select(
                expr,
                func.coalesce(func.sum(UsageLog.cost), 0.0),
                func.coalesce(func.sum(UsageLog.total_tokens), 0),
                func.count(),
            )
            .where(*conditions)
            .group_by(expr)
        )
    ).all()
    # Zero-fill empty buckets so the chart is time-linear (GROUP BY drops gaps).
    populated = [(_canonical_bucket(row[0], bucket), float(row[1]), int(row[2]), int(row[3])) for row in series_rows]
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
        errors_by_status_code=errors_by_status_code,
        series=series,
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
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: str | None = Query(default=None, description=_API_KEY_DESC),
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
) -> Response:
    """Download every breakdown the summary reports, as one CSV.

    One row per (dimension, key): model, user, API key, source, session
    (``source_label``), endpoint, and provider. A dedicated route rather than a
    ``format=csv`` flag on ``/summary`` so that endpoint keeps a single JSON
    response model and a clean OpenAPI schema. The export is **uncapped** (no
    top-N fold): finance wants every row. Kept separate from the bare-array
    ``/v1/usage`` contract, which is untouched.
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
        counts_toward_budget=counts_toward_budget,
    )
    # Driven off the same dimension table as ``/summary`` so a new breakdown lands
    # in the export without a second edit here.
    dimensions = [
        (_CSV_DIMENSION_LABELS.get(name, name), await _breakdown(db, column, conditions, totals, limit=None))
        for name, (column, _cap) in _SUMMARY_DIMENSIONS.items()
    ]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["dimension", "key", "cost", "tokens", "requests"])
    for dimension, rows in dimensions:
        for row in rows:
            writer.writerow(
                [dimension, _csv_safe(row.key or ""), f"{row.cost:.6f}", row.tokens, row.requests]
            )

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="usage-summary.csv"'},
    )
