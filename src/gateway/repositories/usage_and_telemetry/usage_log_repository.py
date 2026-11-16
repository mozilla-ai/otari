"""Data access for ``usage_logs``, one row per attempt the gateway served or imported. Flushes, never commits."""

import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Never, cast

from sqlalchemy import ColumnElement, Select, and_, case, false, func, null, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.metered_pricing import ChargeLine
from gateway.core.sql import bucket_expr, canonical_bucket, dialect_name, match_any, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.usage import SERVED_HERE_SLUG, UsageLog
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.usage_and_telemetry import (
    ANY_TOOL,
    GATEWAY_TOOL_NAMES,
    TOOL_METER_NAMESPACE,
    Bucket,
    UsageGroupedSeriesPoint,
    UsageGroupRow,
    UsageSeriesPoint,
    UsageToolRow,
    UsageTotals,
)

# The statuses that end a request. ``absorbed`` is an attempt a later one recovered
# from, so counting it would count one request twice.
_REQUEST_STATUSES = ("success", "error", "incomplete")


def new_usage_row(
    *,
    workspace_id: uuid.UUID,
    api_key_id: str | None,
    user_id: str,
    model: str,
    provider: str | None,
    endpoint: str,
    status: str,
    latency_ms: int,
    counts_toward_budget: bool,
    request_group_id: str,
    error_message: str | None = None,
    status_code: int | None = None,
    cost: Decimal | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    deployment_paid: bool = False,
) -> UsageLog:
    """A new usage row holding these values, with a fresh id and the current time."""
    return UsageLog(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC),
        workspace_id=workspace_id,
        api_key_id=api_key_id,
        user_id=user_id,
        model=model,
        provider=provider,
        endpoint=endpoint,
        status=status,
        latency_ms=latency_ms,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
        error_message=error_message,
        status_code=status_code,
        cost=cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        deployment_paid=deployment_paid,
    )


def settled_job_row(
    *,
    workspace_id: uuid.UUID,
    api_key_id: str | None,
    user_id: str,
    model: str,
    provider: str,
    endpoint: str,
    counts_toward_budget: bool,
    deployment_paid: bool,
    cost: Decimal | None,
    billing_meters: dict[str, int | float] | None,
    pricing_breakdown: list[ChargeLine] | None,
    pricing_source: str | None,
    pricing_reference: str | None,
    pricing_effective_at: datetime | None,
) -> UsageLog:
    """The row that settles an asynchronous job once it completes.

    No tokens and no latency: the work ran after its request returned. The row
    is priced now, so ``calculated_at`` is its timestamp where it carries a cost.
    """
    now = datetime.now(UTC)
    return UsageLog(
        id=str(uuid.uuid4()),
        timestamp=now,
        workspace_id=workspace_id,
        api_key_id=api_key_id,
        user_id=user_id,
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        counts_toward_budget=counts_toward_budget,
        deployment_paid=deployment_paid,
        cost=cost,
        calculated_at=now if cost is not None else None,
        billing_meters=billing_meters,
        pricing_breakdown=pricing_breakdown,
        pricing_source=pricing_source,
        pricing_reference=pricing_reference,
        pricing_effective_at=pricing_effective_at,
    )


@dataclass(frozen=True)
class ServedTraffic:
    """The requests this gateway served for one name sent, one offering reached and one API key.

    ``model`` is the provider's own id, or a whole selector on a row written before
    ``provider`` was recorded. ``fell_back`` counts the successful requests a
    later attempt served.
    """

    provider: str | None
    model: str
    policy_name: str | None
    api_key_id: str | None
    requests: int
    fell_back: int


@dataclass(frozen=True)
class UsageScope:
    """The rows a read may cover, derived from the caller and never from a client filter.

    ``workspaces`` is a set of workspace ids, or a subquery selecting them, and an
    empty set covers nothing. ``None`` covers every workspace, and has to be passed:
    no read reaches every tenant's rows by leaving the scope out. ``api_key_id``
    narrows the scope to the rows one key wrote.
    """

    workspaces: Collection[uuid.UUID] | Select[uuid.UUID] | None
    api_key_id: str | None = None


@dataclass(frozen=True)
class UsageFilters:
    """What a client narrowed a usage read to. A field left ``None`` matches every row.

    ``tags`` maps each tag key to the values it may hold: values for one key match
    any of them, and different keys must all match.
    """

    start_date: datetime | None = None
    end_date: datetime | None = None
    user_id: list[str] | None = None
    status: str | None = None
    status_code: int | None = None
    model: list[str] | None = None
    endpoint: str | None = None
    provider: str | None = None
    source: str | None = None
    source_label: str | None = None
    tags: Mapping[str, list[str]] = field(default_factory=dict)
    api_key_id: list[str] | None = None
    priced: bool | None = None
    tool: str | None = None
    counts_toward_budget: bool | None = None
    request_group_id: list[str] | None = None
    workspace_id: uuid.UUID | None = None


@dataclass(frozen=True)
class RequestRows:
    """The sums over the rows one request wrote, before anything is decided from them."""

    row_count: int
    cost: Decimal | None
    priced_rows: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    served: int
    errors: int
    incomplete: int
    tool_cost: Decimal | None
    # Priced rows that recorded how much of their cost is tool charges.
    tool_costed_rows: int


# The columns a breakdown can group by, named by the summary's dimension vocabulary.
_GROUP_COLUMNS: dict[str, Any] = {
    "model": UsageLog.model,
    "user": UsageLog.user_id,
    "api_key": UsageLog.api_key_id,
    "source": UsageLog.source,
    "source_label": UsageLog.source_label,
    "endpoint": UsageLog.endpoint,
    "provider": UsageLog.provider,
}


def _scope_conditions(scope: UsageScope) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = []
    workspaces = scope.workspaces
    if isinstance(workspaces, Select):
        conditions.append(UsageLog.workspace_id.in_(workspaces))
    elif workspaces is not None:
        ids = sorted(workspaces)
        if not ids:
            # Belongs to no workspace yet: nothing was refused, there is simply nothing here.
            conditions.append(false())
        elif len(ids) == 1:
            conditions.append(UsageLog.workspace_id == ids[0])
        else:
            conditions.append(UsageLog.workspace_id.in_(ids))
    if scope.api_key_id is not None:
        conditions.append(UsageLog.api_key_id == scope.api_key_id)
    return conditions


def usage_conditions(filters: UsageFilters, scope: UsageScope) -> list[ColumnElement[bool]]:
    """Build the WHERE conditions every usage read shares.

    Keeping this in one place is what makes a paginator's total match the rows
    the list returns, and the summary's tiles match both.

    The scope is ANDed in alongside whatever the client asked for, so a client's
    ``workspace_id`` or ``api_key_id`` can only ever narrow it further.

    Bounds are pinned to UTC here rather than only where a summary resolves its
    window: an offset-less bound would otherwise resolve against the process's
    local timezone, so the same query would size a different set of rows per
    deployment.
    """
    conditions = _scope_conditions(scope)
    if filters.workspace_id is not None:
        conditions.append(UsageLog.workspace_id == filters.workspace_id)
    if filters.start_date is not None:
        conditions.append(UsageLog.timestamp >= utc_bound(filters.start_date))
    if filters.end_date is not None:
        conditions.append(UsageLog.timestamp < utc_bound(filters.end_date))
    if filters.user_id:
        conditions.append(match_any(UsageLog.user_id, filters.user_id))
    if filters.status is not None:
        conditions.append(UsageLog.status == filters.status)
    if filters.status_code is not None:
        conditions.append(UsageLog.status_code == filters.status_code)
        if filters.status is None:
            # Only a failure carries a status code (see ``UsageLog.status_code``),
            # so a bare code filter means "these failures" rather than "whatever
            # rows happen to hold this code": it stays error-scoped even if a
            # future write path starts stamping a code on a non-error row, and it
            # is served by the (status, timestamp) index instead of scanning the
            # window. An explicit ``status`` wins, so the combination stays a
            # literal query rather than a silently contradictory one.
            conditions.append(UsageLog.status == "error")
    if filters.model:
        conditions.append(match_any(UsageLog.model, filters.model))
    if filters.endpoint is not None:
        conditions.append(UsageLog.endpoint == filters.endpoint)
    if filters.provider is not None:
        conditions.append(UsageLog.provider == filters.provider)
    if filters.source is not None:
        conditions.append(UsageLog.source == filters.source)
    if filters.source_label is not None:
        conditions.append(UsageLog.source_label == filters.source_label)
    conditions.extend(match_any(_tag_expr(key), values) for key, values in filters.tags.items())
    if filters.api_key_id:
        conditions.append(match_any(UsageLog.api_key_id, filters.api_key_id))
    if filters.request_group_id:
        # A one-id lookup stays an equality test so it uses the index the same way
        # a single-row fetch would; the IN form is for the dashboard's batched
        # page lookup.
        conditions.append(match_any(UsageLog.request_group_id, filters.request_group_id))
    if filters.priced is True:
        conditions.append(~_needs_pricing_expr())
    elif filters.priced is False:
        conditions.append(_needs_pricing_expr())
    if filters.tool is not None:
        conditions.append(_tool_used_expr(filters.tool))
    if filters.counts_toward_budget is not None:
        conditions.append(UsageLog.counts_toward_budget.is_(filters.counts_toward_budget))
    return conditions


def _tag_expr(key: str) -> Any:
    """The value of tag ``key`` on a row, or null when the row does not carry it."""
    return UsageLog.tags[key].as_string()


def _request_count_expr(status_filter: str | None = None) -> Any:
    """Count requests, not rows.

    Used by every request count a usage read reports (totals, breakdowns and both
    series) so the number means one thing everywhere and the breakdowns still sum
    to the total. A row count deliberately does not use it: that one paginates the
    activity list, where an absorbed attempt is a row the operator can see and
    page through.

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
    ``prompt_tokens`` (see ``core/metered_pricing.billable_usage``), so raw
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
    if tool == ANY_TOOL:
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


class UsageLogRepository(BaseRepository[UsageLog, Never, Never]):
    """Read usage rows inside a set of workspaces."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, UsageLog)

    async def add(self, row: UsageLog) -> None:
        """Stage ``row`` in the open block, so it commits with the step that settled it."""
        self.db.add(row)
        await self.db.flush()

    async def add_many(self, rows: list[UsageLog]) -> None:
        """Stage usage rows together in the current transaction."""
        if not rows:
            return
        self.db.add_all(rows)
        await self.db.flush()

    async def served_traffic(self, *, workspace_ids: Collection[uuid.UUID], since: datetime) -> list[ServedTraffic]:
        """The requests served here since ``since`` in these workspaces, by name sent, offering reached and key.

        One grouped read. Imported usage is left out, because it was never served here.
        """
        if not workspace_ids:
            return []
        fell_back = func.sum(case((and_(UsageLog.status == "success", UsageLog.attempt_position > 1), 1), else_=0))
        stmt = (
            select(
                UsageLog.provider,
                UsageLog.model,
                UsageLog.policy_name,
                UsageLog.api_key_id,
                func.count().label("requests"),
                func.coalesce(fell_back, 0).label("fell_back"),
            )
            .where(
                UsageLog.workspace_id.in_(sorted(workspace_ids)),
                UsageLog.source == SERVED_HERE_SLUG,
                UsageLog.status.in_(_REQUEST_STATUSES),
                UsageLog.timestamp >= since,
            )
            .group_by(UsageLog.provider, UsageLog.model, UsageLog.policy_name, UsageLog.api_key_id)
        )
        return [
            ServedTraffic(
                provider=row.provider,
                model=row.model,
                policy_name=row.policy_name,
                api_key_id=row.api_key_id,
                requests=int(row.requests),
                fell_back=int(row.fell_back),
            )
            for row in (await self.db.execute(stmt)).all()
        ]

    async def page(self, filters: UsageFilters, scope: UsageScope, *, skip: int, limit: int) -> list[UsageLog]:
        """The rows matching these filters inside this scope, most recent first."""
        stmt = (
            select(UsageLog)
            .where(*usage_conditions(filters, scope))
            .order_by(UsageLog.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def count_matching(self, filters: UsageFilters, scope: UsageScope, *, imported_only: bool = False) -> int:
        """How many rows match these filters inside this scope; with ``imported_only``, only rows not served here."""
        conditions = usage_conditions(filters, scope)
        if imported_only:
            conditions.append(UsageLog.source != SERVED_HERE_SLUG)
        stmt: Any = select(func.count()).select_from(UsageLog).where(*conditions)
        return int((await self.db.execute(stmt)).scalar_one())

    async def request_rows(self, request_id: str, *, api_key_id: str | None) -> RequestRows:
        """The sums over the rows one request wrote, confined to one key's rows when ``api_key_id`` is set."""
        conditions: list[ColumnElement[bool]] = [UsageLog.request_group_id == request_id]
        if api_key_id is not None:
            conditions.append(UsageLog.api_key_id == api_key_id)
        stmt = select(
            func.count(),
            func.sum(UsageLog.cost),
            func.count(UsageLog.cost),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            # A vision describe side-call's success row is written before the main
            # call runs and is the only one with no latency, so it settles nothing.
            func.count(case(((UsageLog.status == "success") & UsageLog.latency_ms.is_not(None), 1))),
            func.count(case((UsageLog.status == "error", 1))),
            func.count(case((UsageLog.status == "incomplete", 1))),
            func.sum(UsageLog.tool_cost),
            func.count(case((UsageLog.cost.is_not(None) & UsageLog.tool_cost.is_not(None), 1))),
        ).where(*conditions)
        row = (await self.db.execute(stmt)).one()
        return RequestRows(
            row_count=int(row[0]),
            cost=row[1],
            priced_rows=int(row[2]),
            prompt_tokens=int(row[3] or 0),
            completion_tokens=int(row[4] or 0),
            total_tokens=int(row[5] or 0),
            served=int(row[6]),
            errors=int(row[7]),
            incomplete=int(row[8]),
            tool_cost=row[9],
            tool_costed_rows=int(row[10]),
        )

    async def cost_and_requests(self, filters: UsageFilters, scope: UsageScope) -> tuple[float, int]:
        """The spend and the request count over the matching rows."""
        stmt = select(func.coalesce(func.sum(UsageLog.cost), 0.0), _request_count_expr(filters.status)).where(
            *usage_conditions(filters, scope)
        )
        cost, requests = (await self.db.execute(stmt)).one()
        return float(cost), int(requests)

    async def totals(self, filters: UsageFilters, scope: UsageScope) -> UsageTotals:
        """Grand totals over the matching rows."""
        row = (
            await self.db.execute(
                select(
                    func.coalesce(func.sum(UsageLog.cost), 0.0),
                    func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
                    func.coalesce(func.sum(UsageLog.completion_tokens), 0),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cache_read_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cache_write_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cache_write_1h_tokens), 0),
                    _request_count_expr(filters.status),
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
                    func.coalesce(func.sum(UsageLog.reasoning_tokens), 0),
                ).where(*usage_conditions(filters, scope))
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
            reasoning_tokens=int(row[13]),
        )

    async def groups(
        self,
        filters: UsageFilters,
        scope: UsageScope,
        *,
        dimension: str | None = None,
        tag_key: str | None = None,
        limit: int | None,
    ) -> list[UsageGroupRow]:
        """Spend, billed tokens and requests grouped by a dimension or by one tag's values, biggest spend first.

        ``tokens`` is the *billed* total (input including both cache buckets, plus
        output), the same quantity both series report, so every analytics surface
        agrees on what a token count means.
        """
        column = _tag_expr(tag_key) if tag_key is not None else _GROUP_COLUMNS[cast("str", dimension)]
        cost_sum = func.coalesce(func.sum(UsageLog.cost), 0.0)
        stmt = (
            select(
                column,
                cost_sum,
                _billed_input_sum() + _billed_output_sum(),
                _request_count_expr(filters.status),
            )
            .where(*usage_conditions(filters, scope))
            .group_by(column)
            .order_by(cost_sum.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return [
            UsageGroupRow(key=row[0], cost=float(row[1]), tokens=int(row[2]), requests=int(row[3]))
            for row in (await self.db.execute(stmt)).all()
        ]

    async def tool_usage(self, filters: UsageFilters, scope: UsageScope) -> list[UsageToolRow]:
        """Per-tool calls, failures, spend and requests over the matching rows, biggest spend first.

        One aggregate per known tool rather than a ``GROUP BY`` over the meter map:
        a JSON map's keys cannot be grouped portably across SQLite and PostgreSQL
        (``json_each`` versus ``jsonb_each``, on a plain ``json`` column), and the set
        of gateway-run tools is small and fixed. Cost comes from each row's charge
        line rather than the row total, because a row's ``cost`` also carries tokens.

        ``calls`` and ``errors`` count every call a request made, including calls made by
        an attempt a routing policy later abandoned: the tally is shared across a
        request's attempts and settled onto the row that served, so those calls are on
        that one row rather than spread across the absorbed ones. ``requests`` counts
        requests, so the two answer different questions on purpose: "how much tool work
        did we do" and "how many requests used a tool".
        """
        conditions = usage_conditions(filters, scope)
        out: list[UsageToolRow] = []
        for tool in GATEWAY_TOOL_NAMES:
            calls = _tool_calls_expr(tool)
            errors = UsageLog.billing_meters[(TOOL_METER_NAMESPACE, tool, "errors")].as_integer()
            row = (
                await self.db.execute(
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
                UsageToolRow(tool=tool, calls=int(row[0]), errors=int(row[1]), requests=int(row[2]), cost=float(row[3]))
            )
        return sorted(out, key=lambda r: r.cost, reverse=True)

    async def errors_by_status_code(
        self, filters: UsageFilters, scope: UsageScope, *, limit: int
    ) -> list[tuple[int | None, int]]:
        """Failures among the matching rows by status code, most frequent first, as ``(status_code, requests)``."""
        request_count = func.count()
        rows = (
            await self.db.execute(
                select(UsageLog.status_code, request_count)
                .where(*usage_conditions(filters, scope), UsageLog.status == "error")
                .group_by(UsageLog.status_code)
                .order_by(request_count.desc())
                .limit(limit)
            )
        ).all()
        return [(row[0], int(row[1])) for row in rows]

    async def series(self, filters: UsageFilters, scope: UsageScope, *, bucket: Bucket) -> list[UsageSeriesPoint]:
        """The populated time buckets of the matching rows."""
        expr = bucket_expr(dialect_name(self.db), bucket, UsageLog.timestamp)
        rows = (
            await self.db.execute(
                select(
                    expr,
                    func.coalesce(func.sum(UsageLog.cost), 0.0),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    _request_count_expr(filters.status),
                    func.coalesce(func.sum(case((UsageLog.status == "error", 1), else_=0)), 0),
                    _billed_input_sum(),
                    func.coalesce(func.sum(_billed_expr("cache_read_tokens", UsageLog.cache_read_tokens)), 0),
                    func.coalesce(func.sum(_billed_expr("cache_write_tokens", UsageLog.cache_write_tokens)), 0),
                    _billed_output_sum(),
                )
                .where(*usage_conditions(filters, scope))
                .group_by(expr)
            )
        ).all()
        return [
            UsageSeriesPoint(
                bucket_start=canonical_bucket(row[0], bucket),
                cost=float(row[1]),
                tokens=int(row[2]),
                requests=int(row[3]),
                errors=int(row[4]),
                input_tokens=int(row[5]),
                cache_read_tokens=int(row[6]),
                cache_write_tokens=int(row[7]),
                output_tokens=int(row[8]),
            )
            for row in rows
        ]

    async def grouped_series(
        self,
        filters: UsageFilters,
        scope: UsageScope,
        *,
        bucket: Bucket,
        dimension: str,
        named: Collection[str],
        keeps_null: bool,
    ) -> list[UsageGroupedSeriesPoint]:
        """The populated (time bucket, group) cells of the matching rows, groups outside ``named`` folded into one.

        One grouped query for the whole grid: groups outside the named ones
        collapse into the fold in SQL rather than being fetched and folded by the
        caller, so the row count stays bounded by buckets × (named + 2) regardless
        of cardinality. The synthesized groups are encoded as (key NULL, fold
        flag) rather than a sentinel key string: GROUP BY treats NULLs as equal on
        both dialects, and no sentinel can be trusted never to collide with a real
        key. A NULL column value never matches ``IN``, so it lands in the CASE's
        ``else`` arm; with ``keeps_null`` the fold flag then separates a NULL group
        that ranked among the named ones (a real ``key=None`` series, e.g. a
        deleted user) from the remainder.
        """
        column = _GROUP_COLUMNS[dimension]
        key_expr = case((column.in_(named), column), else_=null())
        if keeps_null:
            fold_expr = case((column.is_(None), 0), (column.in_(named), 0), else_=1)
        else:
            fold_expr = case((column.in_(named), 0), else_=1)
        expr = bucket_expr(dialect_name(self.db), bucket, UsageLog.timestamp)
        rows = (
            await self.db.execute(
                select(
                    expr,
                    key_expr,
                    fold_expr,
                    func.coalesce(func.sum(UsageLog.cost), 0.0),
                    _billed_input_sum() + _billed_output_sum(),
                    _request_count_expr(filters.status),
                )
                .where(*usage_conditions(filters, scope))
                .group_by(expr, key_expr, fold_expr)
            )
        ).all()
        return [
            UsageGroupedSeriesPoint(
                bucket_start=canonical_bucket(row[0], bucket),
                key=row[1],
                is_other=bool(row[2]),
                cost=float(row[3]),
                tokens=int(row[4]),
                requests=int(row[5]),
            )
            for row in rows
        ]
