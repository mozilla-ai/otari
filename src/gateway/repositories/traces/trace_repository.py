"""Data access for ``traces``, one row per agent session. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import astuple, dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Never, cast

from sqlalchemy import (
    BigInteger,
    ColumnElement,
    DateTime,
    Integer,
    Numeric,
    String,
    Uuid,
    and_,
    bindparam,
    case,
    column,
    delete,
    false,
    func,
    or_,
    select,
    text,
    tuple_,
    update,
    values,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import InstrumentedAttribute

from gateway.core.sql import LIKE_ESCAPE, contains_pattern, dialect_name, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.base import UtcDateTime
from gateway.models.traces import Trace
from gateway.repositories.base_repository import BaseRepository
from gateway.types.traces import TraceFilter, TraceSort

# A trace's key: its workspace and its id.
TraceKey = tuple[uuid.UUID, str]


@dataclass(frozen=True)
class TraceGrowth:
    """What one write adds to a trace's totals. Field order is the VALUES list's column order."""

    workspace_id: uuid.UUID
    trace_id: str
    steps: int
    spans: int
    errors: int
    failing_errors: int
    input_tokens: int
    output_tokens: int
    cost: Decimal
    earliest: datetime
    latest: datetime


def _grown(
    steps: Any,
    spans: Any,
    errors: Any,
    failing_errors: Any,
    input_tokens: Any,
    output_tokens: Any,
    cost: Any,
    earliest: Any,
    latest: Any,
) -> dict[str, Any]:
    return {
        "step_count": Trace.step_count + steps,
        "span_count": Trace.span_count + spans,
        "error_count": Trace.error_count + errors,
        "failing_error_count": Trace.failing_error_count + failing_errors,
        "input_tokens": Trace.input_tokens + input_tokens,
        "output_tokens": Trace.output_tokens + output_tokens,
        "cost_snapshot": Trace.cost_snapshot + cost,
        "started_at": case((Trace.started_at > earliest, earliest), else_=Trace.started_at),
        "last_activity_at": case((Trace.last_activity_at < latest, latest), else_=Trace.last_activity_at),
    }


class TraceRepository(BaseRepository[Trace, Never, Never]):
    """Create traces, grow their totals, and find and delete them inside a set of workspaces.

    ``workspace_ids`` is None only for a deployment-wide read; an empty set matches nothing.
    """

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, Trace)

    async def holders(self, keys: Collection[TraceKey]) -> dict[TraceKey, tuple[str | None, int]]:
        """Return the owner and span count of each of these traces that exists, in one read."""
        if not keys:
            return {}
        result = await self.db.execute(
            select(Trace.workspace_id, Trace.trace_id, Trace.user_id, Trace.span_count).where(
                tuple_(Trace.workspace_id, Trace.trace_id).in_(sorted(keys))
            )
        )
        return {(row[0], row[1]): (row[2], int(row[3])) for row in result.all()}

    async def create_absent(self, rows: Sequence[dict[str, Any]]) -> None:
        """Stage the traces not stored yet, in one statement; a concurrent writer's row wins.

        Rows go in key order, so two writers creating the same traces lock them in
        the same order and cannot deadlock.
        """
        if not rows:
            return
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        ordered = sorted(rows, key=lambda row: (str(row["workspace_id"]), row["trace_id"]))
        await self.db.execute(
            insert(Trace).values(ordered).on_conflict_do_nothing(index_elements=[Trace.workspace_id, Trace.trace_id])
        )
        await self.db.flush()

    async def add_totals(self, growth: Sequence[TraceGrowth]) -> None:
        """Grow traces' totals by newly inserted spans, so concurrent writers add up.

        One statement for the whole batch on PostgreSQL, joined to the growth as a
        VALUES list. SQLite cannot name a VALUES list's columns, so there it is a
        statement per trace; it serves one process, where the count matters less.
        """
        if not growth:
            return
        ordered = sorted(growth, key=lambda row: (str(row.workspace_id), row.trace_id))
        if dialect_name(self.db) != "postgresql":
            for row in ordered:
                await self.db.execute(
                    update(Trace)
                    .where(Trace.workspace_id == row.workspace_id, Trace.trace_id == row.trace_id)
                    .values(
                        **_grown(
                            row.steps,
                            row.spans,
                            row.errors,
                            row.failing_errors,
                            row.input_tokens,
                            row.output_tokens,
                            row.cost,
                            row.earliest,
                            row.latest,
                        )
                    )
                )
            await self.db.flush()
            return
        grown = values(
            column("workspace_id", Uuid),
            column("trace_id", String),
            column("steps", Integer),
            column("spans", Integer),
            column("errors", Integer),
            column("failing_errors", Integer),
            column("input_tokens", BigInteger),
            column("output_tokens", BigInteger),
            column("cost", Numeric),
            column("earliest", DateTime(timezone=True)),
            column("latest", DateTime(timezone=True)),
            name="grown",
        ).data([astuple(row) for row in ordered])
        await self.db.execute(
            update(Trace)
            .where(Trace.workspace_id == grown.c.workspace_id, Trace.trace_id == grown.c.trace_id)
            .values(
                **_grown(
                    grown.c.steps,
                    grown.c.spans,
                    grown.c.errors,
                    grown.c.failing_errors,
                    grown.c.input_tokens,
                    grown.c.output_tokens,
                    grown.c.cost,
                    grown.c.earliest,
                    grown.c.latest,
                )
            )
        )
        await self.db.flush()

    async def page(
        self,
        workspace_ids: Collection[uuid.UUID] | None,
        query: TraceFilter,
        *,
        limit: int,
        offset: int,
        sort: TraceSort = TraceSort(),
    ) -> Sequence[Trace]:
        """Return up to ``limit`` matching traces after ``offset``, in ``sort`` order.

        Ties fall back to newest activity, then the trace's key, so an offset page
        never repeats or skips a trace whose sort value it shares with another.
        """
        key = self._sort_expression(sort)
        result = await self.db.execute(
            select(Trace)
            .where(*_scoped(workspace_ids, _conditions(query)))
            .order_by(
                key.desc() if sort.descending else key.asc(),
                Trace.last_activity_at.desc(),
                Trace.trace_id,
                Trace.workspace_id,
            )
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    def _sort_expression(self, sort: TraceSort) -> ColumnElement[Any] | InstrumentedAttribute[Any]:
        match sort.key:
            case "last_activity":
                return Trace.last_activity_at
            case "latency":
                # SQLite stores timestamps as text, so the span is taken in days, which loses
                # sub-millisecond differences; ties then fall to newest activity. Postgres is exact.
                if dialect_name(self.db) == "sqlite":
                    return func.julianday(Trace.last_activity_at) - func.julianday(Trace.started_at)
                return Trace.last_activity_at - Trace.started_at
            case "tokens":
                return Trace.input_tokens + Trace.output_tokens
            case "cost":
                return Trace.cost_snapshot
            case "errors":
                return Trace.error_count

    async def count_matching(self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter) -> int:
        """Count the matching traces without loading them."""
        result = await self.db.execute(
            select(func.count()).select_from(Trace).where(*_scoped(workspace_ids, _conditions(query)))
        )
        return result.scalar_one()

    async def activity_bounds(
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter
    ) -> tuple[datetime | None, datetime | None]:
        """Return the first start and the last activity of the matching sessions, whatever their time window."""
        conditions = _scoped(workspace_ids, _conditions(replace(query, start=None, end=None)))
        row = (
            await self.db.execute(
                select(func.min(Trace.started_at), func.max(Trace.last_activity_at)).where(*conditions)
            )
        ).one()
        return utc_bound(row[0]), utc_bound(row[1])

    async def count_active(
        self,
        workspace_ids: Collection[uuid.UUID] | None,
        query: TraceFilter,
        *,
        starts: Sequence[datetime],
        grain: timedelta,
    ) -> list[tuple[datetime, int, int]]:
        """Return ``(bucket start, succeeded, failed)`` for each bucket a matching session overlaps, oldest first.

        One join of the buckets against the sessions, so the counting stays in the
        database however many there are; a session outside the buckets is never read.
        The filter's own time window is not applied: the buckets are the window.
        A trace failed when it holds an error that fails its session, as the writer counted them.
        """
        if not starts:
            return []
        conditions = _scoped(workspace_ids, _conditions(replace(query, start=None, end=None)))
        buckets = _bucket_rows(starts, grain)
        failed = func.sum(case((Trace.failing_error_count > 0, 1), else_=0))
        result = await self.db.execute(
            select(buckets.c.bucket_start, func.count(), failed)
            .select_from(buckets)
            .join(
                Trace,
                and_(Trace.started_at < buckets.c.bucket_end, Trace.last_activity_at >= buckets.c.bucket_start),
            )
            .where(*conditions, Trace.last_activity_at >= starts[0], Trace.started_at < starts[-1] + grain)
            .group_by(buckets.c.bucket_start)
            .order_by(buckets.c.bucket_start)
        )
        return [(row[0], int(row[1]) - int(row[2] or 0), int(row[2] or 0)) for row in result.all()]

    async def find(
        self, workspace_ids: Collection[uuid.UUID] | None, trace_id: str, *, workspace_id: uuid.UUID | None = None
    ) -> Sequence[Trace]:
        """Return the scope's traces with this id: at most two, enough to tell one from several.

        A trace is keyed by its workspace and its id, so ``workspace_id`` names the one meant.
        """
        conditions = [Trace.trace_id == trace_id]
        if workspace_id is not None:
            conditions.append(Trace.workspace_id == workspace_id)
        result = await self.db.execute(
            select(Trace).where(*_scoped(workspace_ids, conditions)).order_by(Trace.workspace_id).limit(2)
        )
        return result.scalars().all()

    async def delete_for_user(self, user_id: str) -> int:
        """Delete every trace a user owns, in every workspace."""
        return await self._delete(Trace.user_id == user_id)

    async def delete_expired(self, *, idle_before: datetime, started_before: datetime) -> int:
        """Delete every trace idle since ``idle_before``, and each session's trace started before ``started_before``."""
        idle = utc_bound(idle_before)
        started = utc_bound(started_before)
        assert idle is not None and started is not None
        return await self._delete(
            or_(
                Trace.last_activity_at < idle,
                and_(Trace.started_at < started, Trace.session_source != "none"),
            )
        )

    async def _delete(self, condition: ColumnElement[bool]) -> int:
        result = cast("CursorResult[Any]", await self.db.execute(delete(Trace).where(condition)))
        await self.db.flush()
        return int(result.rowcount or 0)


def _bucket_rows(starts: Sequence[datetime], grain: timedelta) -> Any:
    """The buckets as a ``(bucket_start, bucket_end)`` table to join against.

    Written as text because SQLite refuses the column list SQLAlchemy puts on a
    ``VALUES`` alias; both engines name a ``VALUES`` table's columns ``column1``,
    ``column2``, so one spelling serves both. Every value is a bound parameter.
    """
    rows = ", ".join(f"(:start_{index}, :end_{index})" for index in range(len(starts)))
    bounds = [
        bindparam(f"{edge}_{index}", value, type_=UtcDateTime())
        for index, start in enumerate(starts)
        for edge, value in (("start", start), ("end", start + grain))
    ]
    return (
        # Aliased: PostgreSQL before 16 refuses an unnamed VALUES in FROM.
        text(f"SELECT column1 AS bucket_start, column2 AS bucket_end FROM (VALUES {rows}) AS bucket_values")  # noqa: S608 - placeholders only
        .bindparams(*bounds)
        .columns(column("bucket_start", UtcDateTime()), column("bucket_end", UtcDateTime()))
        .subquery("buckets")
    )


def _scoped(
    workspace_ids: Collection[uuid.UUID] | None, conditions: Sequence[ColumnElement[bool]]
) -> list[ColumnElement[bool]]:
    """Put the tenant predicate first. None is every workspace; an empty set matches nothing."""
    if workspace_ids is None:
        return list(conditions)
    if not workspace_ids:
        return [false()]
    return [Trace.workspace_id.in_(list(workspace_ids)), *conditions]


def _conditions(query: TraceFilter) -> list[ColumnElement[bool]]:
    """Build the conditions a query narrows by. An unset value adds none."""
    conditions: list[ColumnElement[bool]] = []
    if query.workspace_ids:
        conditions.append(Trace.workspace_id.in_(list(query.workspace_ids)))
    if (lower := utc_bound(query.start)) is not None:
        conditions.append(Trace.last_activity_at >= lower)
    if (upper := utc_bound(query.end)) is not None:
        conditions.append(Trace.started_at < upper)
    if query.user_ids:
        conditions.append(Trace.user_id.in_(list(query.user_ids)))
    if query.api_key_ids:
        conditions.append(Trace.api_key_id.in_(list(query.api_key_ids)))
    if query.harnesses:
        conditions.append(Trace.harness.in_(list(query.harnesses)))
    if query.session_sources:
        conditions.append(Trace.session_source.in_(list(query.session_sources)))
    if query.has_error is True:
        conditions.append(Trace.error_count > 0)
    elif query.has_error is False:
        conditions.append(Trace.error_count == 0)
    if query.search:
        # The list names a session by its name or harness and a short id with the "s-" cut off,
        # so a search finds any part of those, or of the user.
        pattern = contains_pattern(query.search)
        conditions.append(
            or_(
                *(
                    func.lower(field).like(pattern, escape=LIKE_ESCAPE)
                    for field in (Trace.trace_id, Trace.name, Trace.harness, Trace.user_id)
                )
            )
        )
    return conditions
