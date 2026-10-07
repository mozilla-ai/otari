"""Data access for ``traces``, one row per agent session. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import astuple, dataclass
from datetime import datetime
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
    case,
    column,
    delete,
    false,
    func,
    select,
    tuple_,
    update,
    values,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

from gateway.core.sql import dialect_name, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import Trace
from gateway.repositories.base_repository import BaseRepository
from gateway.types.traces import TraceFilter

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
    input_tokens: int
    output_tokens: int
    cost: Decimal
    earliest: datetime
    latest: datetime


def _grown(
    steps: Any, spans: Any, errors: Any, input_tokens: Any, output_tokens: Any, cost: Any, earliest: Any, latest: Any
) -> dict[str, Any]:
    return {
        "step_count": Trace.step_count + steps,
        "span_count": Trace.span_count + spans,
        "error_count": Trace.error_count + errors,
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
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter, *, limit: int, offset: int
    ) -> Sequence[Trace]:
        """Return up to ``limit`` matching traces after ``offset``, newest activity first."""
        result = await self.db.execute(
            select(Trace)
            .where(*_scoped(workspace_ids, _conditions(query)))
            .order_by(Trace.last_activity_at.desc(), Trace.trace_id)
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    async def count_matching(self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter) -> int:
        """Count the matching traces without loading them."""
        result = await self.db.execute(
            select(func.count()).select_from(Trace).where(*_scoped(workspace_ids, _conditions(query)))
        )
        return result.scalar_one()

    async def find(
        self, workspace_ids: Collection[uuid.UUID] | None, trace_id: str, *, workspace_id: uuid.UUID | None = None
    ) -> Trace | None:
        """Return one trace inside the scope, or None.

        A trace is keyed by its workspace and its id, so ``workspace_id`` names the
        one meant. Without it the same id in two workspaces of the scope resolves
        to the most recently active, never to whichever row comes back first.
        """
        conditions = [Trace.trace_id == trace_id]
        if workspace_id is not None:
            conditions.append(Trace.workspace_id == workspace_id)
        result = await self.db.execute(
            select(Trace)
            .where(*_scoped(workspace_ids, conditions))
            .order_by(Trace.last_activity_at.desc(), Trace.workspace_id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def delete_for_user(self, user_id: str) -> int:
        """Delete every trace a user owns, in every workspace."""
        return await self._delete(Trace.user_id == user_id)

    async def delete_inactive_before(self, before: datetime) -> int:
        """Delete every trace whose last activity is before ``before``."""
        bound = utc_bound(before)
        assert bound is not None
        return await self._delete(Trace.last_activity_at < bound)

    async def _delete(self, condition: ColumnElement[bool]) -> int:
        result = cast("CursorResult[Any]", await self.db.execute(delete(Trace).where(condition)))
        await self.db.flush()
        return int(result.rowcount or 0)


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
        conditions.append(Trace.last_activity_at < upper)
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
    if query.trace_id_prefix:
        escaped = query.trace_id_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions.append(Trace.trace_id.like(f"{escaped}%", escape="\\"))
    return conditions
