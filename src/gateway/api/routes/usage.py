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

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import UsageLog

router = APIRouter(prefix="/v1/usage", tags=["usage"])

# The analytics summary is range-bounded, unlike the raw list. Absent a start_date
# it looks back this far; a wider explicit window is clamped to the hard cap so a
# single request can never turn into an unbounded full-table scan on a growing log.
_DEFAULT_SUMMARY_LOOKBACK = timedelta(days=30)
_MAX_SUMMARY_SPAN = timedelta(days=366)

# How many rows each breakdown returns before the remainder is folded into a
# single synthesized "other" row (so the tables still reconcile with the totals).
_BREAKDOWN_TOP_N = 100

Bucket = Literal["hour", "day"]


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
    cost: float | None
    status: str
    error_message: str | None
    latency_ms: int | None

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageEntry":
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cache_read_tokens=log.cache_read_tokens,
            cache_write_tokens=log.cache_write_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
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
_MODEL_DESC = "Filter to a single model"
_ENDPOINT_DESC = "Filter to a single endpoint (e.g. '/v1/chat/completions')"


def _usage_filters(
    *,
    start_date: datetime | None,
    end_date: datetime | None,
    user_id: str | None,
    status: str | None,
    model: str | None,
    endpoint: str | None,
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
    if model is not None:
        conditions.append(UsageLog.model == model)
    if endpoint is not None:
        conditions.append(UsageLog.endpoint == endpoint)
    return conditions


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, status, model, and endpoint.
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
        model=model,
        endpoint=endpoint,
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


@router.get("/count", dependencies=[Depends(verify_master_key)])
async def count_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
) -> UsageCount:
    """Total number of usage logs matching the given filters.

    Serves the dashboard paginator's "N of M" total without changing the bare
    array contract of ``GET /v1/usage``. Runs only when the client asks (a
    separate request), so the ``COUNT(*)`` is not paid on every page load.
    """
    conditions = _usage_filters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        model=model,
        endpoint=endpoint,
    )
    stmt: Any = select(func.count()).select_from(UsageLog).where(*conditions)
    total = (await db.execute(stmt)).scalar_one()
    return UsageCount(total=total)


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
    request_count: int
    error_count: int
    avg_latency_ms: float | None


class UsageGroupRow(BaseModel):
    """One breakdown row (a model, a user, or an API key). ``key`` is None for
    rows whose grouping column is NULL (e.g. usage from a since-deleted user)."""

    key: str | None
    cost: float
    tokens: int
    requests: int


class UsageSeriesPoint(BaseModel):
    """One time bucket. ``bucket_start`` is canonical ISO-8601 UTC (``...Z``),
    identical across SQLite and PostgreSQL for the same underlying instant."""

    bucket_start: str
    cost: float
    tokens: int
    requests: int


class UsageSummary(BaseModel):
    """Aggregate spend/volume for the Usage & analytics page."""

    start_date: str
    end_date: str
    bucket: Bucket
    totals: UsageTotals
    by_model: list[UsageGroupRow]
    by_user: list[UsageGroupRow]
    by_api_key: list[UsageGroupRow]
    series: list[UsageSeriesPoint]


def _resolve_window(start_date: datetime | None, end_date: datetime | None) -> tuple[datetime, datetime]:
    """Clamp the requested window to a bounded, forward-ordered range.

    A summary must never scan an unbounded log: absent a start we look back
    ``_DEFAULT_SUMMARY_LOOKBACK``; a span wider than ``_MAX_SUMMARY_SPAN`` has its
    start pulled forward so the aggregates stay bounded by the timestamp index.
    """
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
                func.count(),
                func.coalesce(func.sum(case((UsageLog.status == "error", 1), else_=0)), 0),
                func.avg(UsageLog.latency_ms),
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
        request_count=int(row[6]),
        error_count=int(row[7]),
        avg_latency_ms=float(row[8]) if row[8] is not None else None,
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
                )
            )
    return result


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def usage_summary(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
) -> UsageSummary:
    """Aggregate spend, tokens, and request volume for the dashboard Usage page.

    Range-bounded (default last 30 days, hard-capped): unlike the raw ``/v1/usage``
    list, every aggregate is scoped to a bounded window so it stays served by the
    timestamp index. Returns grand totals, breakdowns by model / user / API key
    (top rows plus a reconciling ``other`` fold), and a UTC-bucketed time series.
    """
    start, end = _resolve_window(start_date, end_date)
    conditions = _usage_filters(
        start_date=start,
        end_date=end,
        user_id=user_id,
        status=status,
        model=model,
        endpoint=endpoint,
    )
    totals = await _totals(db, conditions)
    by_model = await _breakdown(db, UsageLog.model, conditions, totals, limit=_BREAKDOWN_TOP_N)
    by_user = await _breakdown(db, UsageLog.user_id, conditions, totals, limit=_BREAKDOWN_TOP_N)
    by_api_key = await _breakdown(db, UsageLog.api_key_id, conditions, totals, limit=_BREAKDOWN_TOP_N)

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
            .order_by(expr)
        )
    ).all()
    series = [
        UsageSeriesPoint(
            bucket_start=_canonical_bucket(row[0], bucket),
            cost=float(row[1]),
            tokens=int(row[2]),
            requests=int(row[3]),
        )
        for row in series_rows
    ]

    return UsageSummary(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        bucket=bucket,
        totals=totals,
        by_model=by_model,
        by_user=by_user,
        by_api_key=by_api_key,
        series=series,
    )


# Leading characters a spreadsheet may interpret as a formula. Any cell starting
# with one is prefixed with a single quote so opening the CSV in Excel/Sheets can
# never execute attacker-influenced text (model / user ids are caller-supplied).
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


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
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
) -> Response:
    """Download the per-model / per-user / per-key breakdown as CSV.

    A dedicated route rather than a ``format=csv`` flag on ``/summary`` so that
    endpoint keeps a single JSON response model and a clean OpenAPI schema. The
    export is **uncapped** (no top-N fold): finance wants every row. Kept separate
    from the bare-array ``/v1/usage`` contract, which is untouched.
    """
    start, end = _resolve_window(start_date, end_date)
    conditions = _usage_filters(
        start_date=start,
        end_date=end,
        user_id=user_id,
        status=status,
        model=model,
        endpoint=endpoint,
    )
    totals = await _totals(db, conditions)
    dimensions = (
        ("model", await _breakdown(db, UsageLog.model, conditions, totals, limit=None)),
        ("user", await _breakdown(db, UsageLog.user_id, conditions, totals, limit=None)),
        ("api_key", await _breakdown(db, UsageLog.api_key_id, conditions, totals, limit=None)),
    )

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
