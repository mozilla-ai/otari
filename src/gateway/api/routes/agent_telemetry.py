"""Read and purge endpoints for captured coding-agent telemetry.

`agent_telemetry` holds two kinds of content-free, non-billable row: behavioral
events (tool results, decisions, prompts, errors) and outcome-metric points
(lines changed, commits, pull requests, active time). Neither is worth much on
its own; joined against recorded spend they answer "what did this cost per unit
of work", which is what `/summary` computes.

The aggregation queries live here rather than in a service module, matching
`usage.py`'s own layout, and use SQLAlchemy core against the ORM-mapped classes
so the route layer stays free of `sqlalchemy.orm`.
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import ColumnElement, case, func, null, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key

# The window, bucket-grid, and fold conventions are `usage.py`'s: a summary read
# from both endpoints has to describe the same window the same way, so they share
# one implementation rather than two that drift.
from gateway.api.routes.usage import (
    _MAX_SERIES_POINTS,
    _SERIES_TOP_N,
    Bucket,
    _bucket_expr,
    _canonical_bucket,
    _dense_series,
    _dialect_name,
    _request_count_expr,
    _resolve_window,
)
from gateway.core.sql import MAX_FILTER_VALUES, match_any
from gateway.models.entities import AgentTelemetry, UsageLog
from gateway.services.agent_telemetry_admin_service import (
    AgentTelemetryDeleteRequest,
    AgentTelemetryDeleteResult,
    delete_agent_telemetry,
)
from gateway.services.agent_telemetry_service import (
    DELTA,
    METRIC_ACTIVE_TIME,
    METRIC_COMMITS,
    METRIC_LINES_OF_CODE,
    METRIC_PULL_REQUESTS,
    compute_series_increment,
    series_point_increments,
)

router = APIRouter(prefix="/v1/agent-telemetry", tags=["agent-telemetry"])

TelemetryGroupBy = Literal["user_id", "api_key_id"]

_SECONDS_PER_HOUR = 3600.0

# How many tools the mix reports before the tail is dropped. A session uses a
# handful; this only bounds a pathological one.
_TOOL_MIX_TOP_N = 50

_START_DESC = "Return rows with timestamp >= start_date (ISO 8601 or Unix epoch seconds)"
_END_DESC = "Return rows with timestamp < end_date (ISO 8601 or Unix epoch seconds)"
_NAME_DESC = "Filter to a single event type or metric name (e.g. 'tool_result', 'claude_code.commit.count')"
_USER_DESC = (
    "Filter to one or more users; repeatable (user_id=a&user_id=b). Several values match any of them. "
    f"At most {MAX_FILTER_VALUES} per call."
)
_API_KEY_DESC = (
    "Filter to one or more API key ids; repeatable (api_key_id=a&api_key_id=b). Several values match "
    f"any of them. At most {MAX_FILTER_VALUES} per call."
)
_SESSION_DESC = (
    "Filter to a single agent session. Matches agent_telemetry.session_label and, on the usage side of "
    "the join, the usage_logs.source_label that /v1/usage/summary filters on"
)
_BUCKET_DESC = "Time-series granularity: 'hour' or 'day'"

_UserFilter = Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_DESC)]
_ApiKeyFilter = Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_DESC)]


class AgentTelemetryOutcomes(BaseModel):
    """How much work the agent produced inside the window.

    Each value is the read-time increment for its metric: a delta series is
    summed, a cumulative one is diffed per series generation, so a counter that
    is re-reported in full on every export is never counted twice.
    ``lines_of_code`` sums the added and removed series, which are stored
    separately. ``active_time`` is in seconds, as the agent reports it.
    """

    commits: float = 0.0
    pull_requests: float = 0.0
    lines_of_code: float = 0.0
    active_time: float = 0.0


class AgentTelemetryToolRow(BaseModel):
    """Tool call volume for one tool. ``tool`` is null for a call with no name."""

    tool: str | None
    calls: int


class AgentTelemetryBehavior(BaseModel):
    """Counts from the behavioral events already captured on the logs signal."""

    tool_calls: int = 0
    by_tool: list[AgentTelemetryToolRow] = []
    tool_accepts: int = 0
    tool_rejects: int = 0
    turns: int = 0
    sessions: int = 0
    api_errors: int = 0


class AgentTelemetryUsage(BaseModel):
    """The recorded spend the measures below divide, reported so they can be checked."""

    cost: float = 0.0
    requests: int = 0


class AgentTelemetryMeasures(BaseModel):
    """Cost and quality per unit of work. Each is null when its denominator is zero.

    ``edit_acceptance_rate`` and ``tool_acceptance_rate`` are the same quantity
    today, both derived from the ``tool_decision`` event: the agent's own
    ``code_edit_tool.decision`` metric is deliberately never stored, since
    ``tool_decision`` already carries that accept/reject signal. Both names are
    reported so a caller reading either vocabulary gets the right number.
    """

    cost_per_commit: float | None = None
    cost_per_pull_request: float | None = None
    cost_per_line: float | None = None
    spend_per_active_hour: float | None = None
    edit_acceptance_rate: float | None = None
    tool_acceptance_rate: float | None = None
    turns_per_session: float | None = None
    # Agent-observed API errors per recorded request in the window.
    error_rate: float | None = None


class AgentTelemetrySeriesPoint(BaseModel):
    """One UTC time bucket of outcomes, behavior, and the spend beside them.

    A cumulative series' increment is attributed to the bucket of the later point
    of each consecutive pair, so a bucket reports the growth observed inside it.
    """

    bucket_start: str
    cost: float = 0.0
    commits: float = 0.0
    pull_requests: float = 0.0
    lines_of_code: float = 0.0
    active_time: float = 0.0
    tool_calls: int = 0
    turns: int = 0
    api_errors: int = 0


class AgentTelemetrySummary(BaseModel):
    """Agent outcomes and behavior for the window, joined against recorded spend."""

    start_date: str
    end_date: str
    bucket: Bucket
    usage: AgentTelemetryUsage
    outcomes: AgentTelemetryOutcomes
    behavior: AgentTelemetryBehavior
    measures: AgentTelemetryMeasures
    series: list[AgentTelemetrySeriesPoint]


class AgentTelemetryCount(BaseModel):
    """Total number of agent_telemetry rows matching a set of filters."""

    total: int


class AgentTelemetryGroupRow(BaseModel):
    """One group of a grouped series. ``key`` is null for the ``other`` fold and
    for a real group whose column is NULL (e.g. a since-deleted user);
    ``is_other`` separates the two."""

    key: str | None
    rows: int
    is_other: bool = False


class AgentTelemetryGroupedSeriesPoint(BaseModel):
    """One (bucket, group) cell: how many rows that group recorded in that bucket."""

    bucket_start: str
    key: str | None
    is_other: bool = False
    rows: int


class AgentTelemetryGroupedSeries(BaseModel):
    """A per-group row-volume series, for stacked charting."""

    start_date: str
    end_date: str
    bucket: Bucket
    group_by: TelemetryGroupBy
    groups: list[AgentTelemetryGroupRow]
    points: list[AgentTelemetryGroupedSeriesPoint]


def _telemetry_filters(
    *,
    start_date: datetime | None,
    end_date: datetime | None,
    user_id: list[str] | None,
    api_key_id: list[str] | None,
    name: str | None = None,
    session_label: str | None = None,
) -> list[ColumnElement[bool]]:
    """Shared WHERE conditions, so `/count` sizes exactly what the others read.

    ``session_label`` is `/summary`'s alone: `/count` and `/series` mirror the
    purge endpoint's filter set, and a read filter the purge cannot express would
    size a selection it could not delete.
    """
    conditions: list[ColumnElement[bool]] = []
    if start_date is not None:
        conditions.append(AgentTelemetry.timestamp >= start_date)
    if end_date is not None:
        conditions.append(AgentTelemetry.timestamp < end_date)
    if user_id:
        conditions.append(match_any(AgentTelemetry.user_id, user_id))
    if api_key_id:
        conditions.append(match_any(AgentTelemetry.api_key_id, api_key_id))
    if name is not None:
        conditions.append(AgentTelemetry.name == name)
    if session_label is not None:
        conditions.append(AgentTelemetry.session_label == session_label)
    return conditions


def _usage_filters(
    *,
    start: datetime,
    end: datetime,
    user_id: list[str] | None,
    api_key_id: list[str] | None,
    session_label: str | None = None,
) -> list[ColumnElement[bool]]:
    """The same scope, expressed against usage_logs, for the cost side of the join.

    A session is named ``source_label`` on this side of the join, which is the
    column `/v1/usage/summary` already filters sessions on.
    """
    conditions: list[ColumnElement[bool]] = [UsageLog.timestamp >= start, UsageLog.timestamp < end]
    if user_id:
        conditions.append(match_any(UsageLog.user_id, user_id))
    if api_key_id:
        conditions.append(match_any(UsageLog.api_key_id, api_key_id))
    if session_label is not None:
        conditions.append(UsageLog.source_label == session_label)
    return conditions


def _bucket_key(timestamp: datetime, bucket: Bucket) -> str:
    """The canonical UTC bucket a point falls in, matching the SQL bucket grid."""
    aware = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
    utc = aware.astimezone(UTC)
    fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
    return utc.strftime(fmt)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


async def _metric_increments(
    db: AsyncSession, conditions: list[ColumnElement[bool]], bucket: Bucket
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Window increments per metric name, in total and per time bucket.

    The delta arithmetic is deliberately not in SQL: a cumulative point is stored
    exactly as reported, so turning a series into "how much happened here" means
    walking its points in time order and diffing, split at each series generation
    (a changed ``series_start`` is a counter reset). The scan is bounded by the
    window and served by the ``(series_key, timestamp)`` index.
    """
    rows = (
        await db.execute(
            select(
                AgentTelemetry.name,
                AgentTelemetry.series_key,
                AgentTelemetry.series_start,
                AgentTelemetry.temporality,
                AgentTelemetry.timestamp,
                AgentTelemetry.value,
            ).where(*conditions, AgentTelemetry.kind == "metric", AgentTelemetry.value.is_not(None))
        )
    ).all()

    generations: dict[tuple[str, str | None, datetime | None, str | None], list[tuple[datetime, float]]] = defaultdict(
        list
    )
    for name, series_key, series_start, temporality, timestamp, value in rows:
        generations[(name, series_key, series_start, temporality)].append((timestamp, float(value)))

    totals: dict[str, float] = defaultdict(float)
    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (name, _series_key, _series_start, temporality), points in generations.items():
        increments = series_point_increments(points, temporality or DELTA)
        totals[name] += compute_series_increment(increments, DELTA)
        for timestamp, increment in increments:
            by_bucket[_bucket_key(timestamp, bucket)][name] += increment
    return totals, by_bucket


async def _behavior(
    db: AsyncSession, conditions: list[ColumnElement[bool]]
) -> AgentTelemetryBehavior:
    """Behavioral-event counts, from one grouped pass plus the session count."""
    behavioral = [*conditions, AgentTelemetry.kind.is_(None)]
    rows = (
        await db.execute(
            select(AgentTelemetry.name, AgentTelemetry.tool_name, AgentTelemetry.decision, func.count())
            .where(*behavioral)
            .group_by(AgentTelemetry.name, AgentTelemetry.tool_name, AgentTelemetry.decision)
        )
    ).all()
    sessions = (
        await db.execute(
            select(func.count(func.distinct(AgentTelemetry.session_label))).where(*behavioral)
        )
    ).scalar_one()

    behavior = AgentTelemetryBehavior(sessions=int(sessions))
    by_tool: dict[str | None, int] = defaultdict(int)
    for name, tool_name, decision, count in rows:
        count = int(count)
        if name == "tool_result":
            behavior.tool_calls += count
            by_tool[tool_name] += count
        elif name == "tool_decision":
            if decision == "accept":
                behavior.tool_accepts += count
            elif decision == "reject":
                behavior.tool_rejects += count
        elif name == "user_prompt":
            behavior.turns += count
        elif name == "api_error":
            behavior.api_errors += count
    behavior.by_tool = [
        AgentTelemetryToolRow(tool=tool, calls=calls)
        for tool, calls in sorted(by_tool.items(), key=lambda item: item[1], reverse=True)[:_TOOL_MIX_TOP_N]
    ]
    return behavior


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def agent_telemetry_summary(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: _UserFilter = None,
    api_key_id: _ApiKeyFilter = None,
    session_label: str | None = Query(default=None, description=_SESSION_DESC),
    bucket: Bucket = Query(default="day", description=_BUCKET_DESC),
) -> AgentTelemetrySummary:
    """What the coding agent produced in a window, and what it cost (standalone).

    Range-bounded like `/v1/usage/summary` (default last 30 days, hard-capped),
    so the aggregates stay served by the timestamp index. Returns the outcome
    totals (commits, pull requests, lines changed, active time), the behavioral
    counts already captured from the logs signal (tool calls and their mix, tool
    accept/reject, turns, API errors), the recorded spend over the same scope,
    and the derived per-unit measures: cost per commit / pull request / line,
    spend per active hour, acceptance rate, turns per session, and error rate.
    Each measure is null rather than an error when its denominator is zero.
    Filterable by user, API key, and `session_label`, so cost per outcome can be
    read for one agent session as well as for a whole window.

    Outcome metrics are stored exactly as the agent reported them, so a
    cumulative counter is converted to a window increment here, at read time,
    diffed per series generation: a re-exported total adds nothing, and a counter
    reset never reads as negative work. Master-key only.
    """
    start, end = _resolve_window(start_date, end_date)
    conditions = _telemetry_filters(
        start_date=start, end_date=end, user_id=user_id, api_key_id=api_key_id, session_label=session_label
    )
    usage_conditions = _usage_filters(
        start=start, end=end, user_id=user_id, api_key_id=api_key_id, session_label=session_label
    )

    outcome_totals, outcomes_by_bucket = await _metric_increments(db, conditions, bucket)
    behavior = await _behavior(db, conditions)
    # Requests, not rows, the same way /v1/usage/summary counts them: a routed
    # request writes one row per recovered attempt, and counting those would
    # deflate the error rate against a request volume the Usage page never shows.
    usage_row = (
        await db.execute(
            select(func.coalesce(func.sum(UsageLog.cost), 0.0), _request_count_expr()).where(*usage_conditions)
        )
    ).one()
    usage = AgentTelemetryUsage(cost=float(usage_row[0]), requests=int(usage_row[1]))

    outcomes = AgentTelemetryOutcomes(
        commits=outcome_totals.get(METRIC_COMMITS, 0.0),
        pull_requests=outcome_totals.get(METRIC_PULL_REQUESTS, 0.0),
        lines_of_code=outcome_totals.get(METRIC_LINES_OF_CODE, 0.0),
        active_time=outcome_totals.get(METRIC_ACTIVE_TIME, 0.0),
    )
    decisions = behavior.tool_accepts + behavior.tool_rejects
    acceptance = _ratio(behavior.tool_accepts, decisions)
    measures = AgentTelemetryMeasures(
        cost_per_commit=_ratio(usage.cost, outcomes.commits),
        cost_per_pull_request=_ratio(usage.cost, outcomes.pull_requests),
        cost_per_line=_ratio(usage.cost, outcomes.lines_of_code),
        spend_per_active_hour=_ratio(usage.cost, outcomes.active_time / _SECONDS_PER_HOUR),
        edit_acceptance_rate=acceptance,
        tool_acceptance_rate=acceptance,
        turns_per_session=_ratio(behavior.turns, behavior.sessions),
        error_rate=_ratio(behavior.api_errors, usage.requests),
    )

    series = await _summary_series(
        db,
        conditions,
        usage_conditions,
        outcomes_by_bucket,
        bucket=bucket,
        start=start,
        end=end,
    )
    return AgentTelemetrySummary(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        bucket=bucket,
        usage=usage,
        outcomes=outcomes,
        behavior=behavior,
        measures=measures,
        series=series,
    )


async def _summary_series(
    db: AsyncSession,
    conditions: list[ColumnElement[bool]],
    usage_conditions: list[ColumnElement[bool]],
    outcomes_by_bucket: dict[str, dict[str, float]],
    *,
    bucket: Bucket,
    start: datetime,
    end: datetime,
) -> list[AgentTelemetrySeriesPoint]:
    """Merge the per-bucket outcome increments with bucketed behavior and spend."""
    dialect = _dialect_name(db)
    telemetry_bucket = _bucket_expr(dialect, bucket, AgentTelemetry.timestamp)
    behavior_rows = (
        await db.execute(
            select(telemetry_bucket, AgentTelemetry.name, func.count())
            .where(*conditions, AgentTelemetry.kind.is_(None))
            .group_by(telemetry_bucket, AgentTelemetry.name)
        )
    ).all()
    usage_bucket = _bucket_expr(dialect, bucket)
    usage_rows = (
        await db.execute(
            select(usage_bucket, func.coalesce(func.sum(UsageLog.cost), 0.0))
            .where(*usage_conditions)
            .group_by(usage_bucket)
        )
    ).all()

    populated: dict[str, AgentTelemetrySeriesPoint] = {}

    def point_for(key: str) -> AgentTelemetrySeriesPoint:
        return populated.setdefault(key, AgentTelemetrySeriesPoint(bucket_start=key))

    for key, outcomes in outcomes_by_bucket.items():
        point = point_for(key)
        point.commits = outcomes.get(METRIC_COMMITS, 0.0)
        point.pull_requests = outcomes.get(METRIC_PULL_REQUESTS, 0.0)
        point.lines_of_code = outcomes.get(METRIC_LINES_OF_CODE, 0.0)
        point.active_time = outcomes.get(METRIC_ACTIVE_TIME, 0.0)
    for raw_bucket, name, count in behavior_rows:
        point = point_for(_canonical_bucket(raw_bucket, bucket))
        if name == "tool_result":
            point.tool_calls += int(count)
        elif name == "user_prompt":
            point.turns += int(count)
        elif name == "api_error":
            point.api_errors += int(count)
    for raw_bucket, cost in usage_rows:
        point_for(_canonical_bucket(raw_bucket, bucket)).cost = float(cost)

    return _dense_series(
        start, end, bucket, populated, lambda key: AgentTelemetrySeriesPoint(bucket_start=key)
    )


@router.get("/count", dependencies=[Depends(verify_master_key)])
async def count_agent_telemetry(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: _UserFilter = None,
    api_key_id: _ApiKeyFilter = None,
    name: str | None = Query(default=None, description=_NAME_DESC),
) -> AgentTelemetryCount:
    """Total agent_telemetry rows matching the given filters (standalone).

    The filter set mirrors the purge endpoint's, so this sizes exactly what a
    "delete all N matching" would remove. Behavioral and metric rows are counted
    together: neither this nor the purge distinguishes them. Master-key only.
    """
    conditions = _telemetry_filters(
        start_date=start_date, end_date=end_date, user_id=user_id, api_key_id=api_key_id, name=name
    )
    stmt: Any = select(func.count()).select_from(AgentTelemetry).where(*conditions)
    return AgentTelemetryCount(total=(await db.execute(stmt)).scalar_one())


@router.get("/series", dependencies=[Depends(verify_master_key)])
async def agent_telemetry_series(
    db: Annotated[AsyncSession, Depends(get_db)],
    group_by: TelemetryGroupBy = Query(description="Dimension to split the series by"),
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: _UserFilter = None,
    api_key_id: _ApiKeyFilter = None,
    name: str | None = Query(default=None, description=_NAME_DESC),
    bucket: Bucket = Query(default="day", description=_BUCKET_DESC),
) -> AgentTelemetryGroupedSeries:
    """Row volume over time, split by user or API key (standalone).

    Mirrors `/v1/usage/series`: same window bounds and bucket-grid cap, the top
    groups as their own series with the remainder folded into a reconciling
    ``other``, and sparse points (populated cells only). Counts rows, not spend,
    so it charts telemetry volume rather than cost. Master-key only.
    """
    start, end = _resolve_window(start_date, end_date)
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
    if (end - start) / step > _MAX_SERIES_POINTS:
        raise HTTPException(
            status_code=422,
            detail=f"window spans more than {_MAX_SERIES_POINTS} {bucket} buckets; use bucket=day or narrow the range",
        )
    conditions = _telemetry_filters(
        start_date=start, end_date=end, user_id=user_id, api_key_id=api_key_id, name=name
    )
    column = AgentTelemetry.user_id if group_by == "user_id" else AgentTelemetry.api_key_id

    row_count = func.count()
    group_rows = (
        await db.execute(
            select(column, row_count).where(*conditions).group_by(column).order_by(row_count.desc()).limit(
                _SERIES_TOP_N
            )
        )
    ).all()
    total = (
        await db.execute(select(func.count()).select_from(AgentTelemetry).where(*conditions))
    ).scalar_one()
    groups = [AgentTelemetryGroupRow(key=row[0], rows=int(row[1])) for row in group_rows]
    folded = int(total) - sum(group.rows for group in groups)
    if folded > 0:
        groups.append(AgentTelemetryGroupRow(key=None, rows=folded, is_other=True))

    # Groups past the top N collapse in SQL, so the grid stays bounded by
    # buckets x (top N + 2) however high the dimension's cardinality is. The fold
    # is encoded as (key NULL, flag) rather than a sentinel key, which no value
    # could be trusted never to collide with; the flag then separates a real NULL
    # group that ranked in the top N from the remainder.
    named = {group.key for group in groups if group.key is not None}
    keeps_null = any(group.key is None and not group.is_other for group in groups)
    key_expr = case((column.in_(named), column), else_=null())
    if keeps_null:
        fold_expr = case((column.is_(None), 0), (column.in_(named), 0), else_=1)
    else:
        fold_expr = case((column.in_(named), 0), else_=1)
    bucket_expr = _bucket_expr(_dialect_name(db), bucket, AgentTelemetry.timestamp)
    rows = (
        await db.execute(
            select(bucket_expr, key_expr, fold_expr, func.count())
            .where(*conditions)
            .group_by(bucket_expr, key_expr, fold_expr)
        )
    ).all()
    points = [
        AgentTelemetryGroupedSeriesPoint(
            bucket_start=_canonical_bucket(row[0], bucket),
            key=row[1],
            is_other=bool(row[2]),
            rows=int(row[3]),
        )
        for row in rows
    ]
    points.sort(key=lambda point: point.bucket_start)

    return AgentTelemetryGroupedSeries(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        bucket=bucket,
        group_by=group_by,
        groups=groups,
        points=points,
    )


@router.delete("", dependencies=[Depends(verify_master_key)])
async def delete_agent_telemetry_rows(
    request: AgentTelemetryDeleteRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentTelemetryDeleteResult:
    """Delete agent_telemetry rows by explicit ids or by filter (standalone).

    Target either an explicit selection (`ids`) or everything matching a filter
    (`by_filter: true` plus optional `user_id` / `api_key_id` / `name` / date
    range). A selection matching zero rows succeeds with `deleted: 0`.
    Master-key only.
    """
    return await delete_agent_telemetry(db, request)
