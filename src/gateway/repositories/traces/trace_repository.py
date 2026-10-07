"""Data access for ``traces``, one row per agent session. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Never, Protocol, cast

from sqlalchemy import ColumnElement, case, delete, false, func, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

from gateway.core.sql import dialect_name, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import Trace
from gateway.repositories.base_repository import BaseRepository


class TraceQuery(Protocol):
    """What narrows a trace read or delete. The port's ``TraceFilter`` has this shape."""

    @property
    def workspace_ids(self) -> Collection[uuid.UUID]: ...
    @property
    def start(self) -> datetime | None: ...
    @property
    def end(self) -> datetime | None: ...
    @property
    def user_ids(self) -> Collection[str]: ...
    @property
    def api_key_ids(self) -> Collection[str]: ...
    @property
    def harnesses(self) -> Collection[str]: ...
    @property
    def session_sources(self) -> Collection[str]: ...
    @property
    def has_error(self) -> bool | None: ...
    @property
    def trace_id_prefix(self) -> str | None: ...


class TraceRepository(BaseRepository[Trace, Never, Never]):
    """Create traces, grow their totals, and find and delete them inside a set of workspaces.

    ``workspace_ids`` is None only for a deployment-wide read; an empty set matches nothing.
    """

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, Trace)

    async def owners(self, workspace_id: uuid.UUID, trace_ids: Collection[str]) -> dict[str, str | None]:
        """Return the owning user of each of these traces that already exists."""
        if not trace_ids:
            return {}
        result = await self.db.execute(
            select(Trace.trace_id, Trace.user_id).where(
                Trace.workspace_id == workspace_id, Trace.trace_id.in_(list(trace_ids))
            )
        )
        return {trace_id: user_id for trace_id, user_id in result.all()}

    async def create_if_absent(self, values: dict[str, Any]) -> None:
        """Stage a trace unless one with the same key exists; a concurrent writer's row wins."""
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(Trace).values(**values).on_conflict_do_nothing(index_elements=[Trace.workspace_id, Trace.trace_id])
        )
        await self.db.flush()

    async def add_totals(
        self,
        workspace_id: uuid.UUID,
        trace_id: str,
        *,
        steps: int,
        spans: int,
        errors: int,
        input_tokens: int,
        output_tokens: int,
        cost: Decimal,
        earliest: datetime,
        latest: datetime,
    ) -> None:
        """Grow a trace's totals by newly inserted spans, in one statement so concurrent writers add up."""
        await self.db.execute(
            update(Trace)
            .where(Trace.workspace_id == workspace_id, Trace.trace_id == trace_id)
            .values(
                step_count=Trace.step_count + steps,
                span_count=Trace.span_count + spans,
                error_count=Trace.error_count + errors,
                input_tokens=Trace.input_tokens + input_tokens,
                output_tokens=Trace.output_tokens + output_tokens,
                cost_snapshot=Trace.cost_snapshot + cost,
                started_at=case((Trace.started_at > earliest, earliest), else_=Trace.started_at),
                last_activity_at=case((Trace.last_activity_at < latest, latest), else_=Trace.last_activity_at),
            )
        )
        await self.db.flush()

    async def page(
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceQuery, *, limit: int, offset: int
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

    async def count_matching(self, workspace_ids: Collection[uuid.UUID] | None, query: TraceQuery) -> int:
        """Count the matching traces without loading them."""
        result = await self.db.execute(
            select(func.count()).select_from(Trace).where(*_scoped(workspace_ids, _conditions(query)))
        )
        return result.scalar_one()

    async def find(self, workspace_ids: Collection[uuid.UUID] | None, trace_id: str) -> Trace | None:
        """Return one trace inside the scope, or None."""
        result = await self.db.execute(
            select(Trace).where(*_scoped(workspace_ids, [Trace.trace_id == trace_id])).limit(1)
        )
        return result.scalar_one_or_none()

    async def delete_matching(self, workspace_ids: Collection[uuid.UUID] | None, query: TraceQuery) -> int:
        """Delete the matching traces; their spans go with them by the foreign key's cascade."""
        keys = select(Trace.workspace_id, Trace.trace_id).where(*_scoped(workspace_ids, _conditions(query)))
        return await self._delete(tuple_(Trace.workspace_id, Trace.trace_id).in_(keys))

    async def delete_for_user(self, user_id: str) -> int:
        """Delete every trace a user owns, in every workspace."""
        return await self._delete(Trace.user_id == user_id)

    async def inactive_before(self, before: datetime, *, limit: int) -> list[tuple[uuid.UUID, str]]:
        """Sessions whose last activity is before ``before``, the ones the next expiry deletes."""
        bound = utc_bound(before)
        assert bound is not None
        result = await self.db.execute(
            select(Trace.workspace_id, Trace.trace_id).where(Trace.last_activity_at < bound).limit(limit)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def matching_sessions(
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceQuery, *, limit: int
    ) -> list[tuple[uuid.UUID, str]]:
        """The sessions a purge with these filters deletes, a page at a time."""
        result = await self.db.execute(
            select(Trace.workspace_id, Trace.trace_id).where(*_scoped(workspace_ids, _conditions(query))).limit(limit)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def sessions_for_user(self, user_id: str, *, limit: int) -> list[tuple[uuid.UUID, str]]:
        result = await self.db.execute(
            select(Trace.workspace_id, Trace.trace_id).where(Trace.user_id == user_id).limit(limit)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def existing(self, keys: Collection[tuple[uuid.UUID, str]]) -> set[tuple[uuid.UUID, str]]:
        """Which of these sessions are stored."""
        if not keys:
            return set()
        result = await self.db.execute(
            select(Trace.workspace_id, Trace.trace_id).where(tuple_(Trace.workspace_id, Trace.trace_id).in_(list(keys)))
        )
        return {(row[0], row[1]) for row in result.all()}

    async def delete_sessions(self, keys: Collection[tuple[uuid.UUID, str]]) -> int:
        if not keys:
            return 0
        return await self._delete(tuple_(Trace.workspace_id, Trace.trace_id).in_(list(keys)))

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


def _conditions(query: TraceQuery) -> list[ColumnElement[bool]]:
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
