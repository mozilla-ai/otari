"""The core TraceStoragePort adapters: this deployment's database, and none.

``LocalTraceStorage`` keeps traces in the ``traces`` and ``trace_spans`` tables. It
is bound wherever the deployment serves the control plane, and an overlay may
rebind the port to a store built for many tenants. ``NullTraceStorage`` stands
where no store is reachable yet, and keeps nothing.

The local adapter reaches its tables only through the traces repositories, which
it receives as a builder rather than importing: a domain's repositories are
imported by its own packages and the builders in ``api/deps.py`` alone.
``gateway.types.trace_tables`` says what it needs from them.
"""

import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import Trace, TraceSpan
from gateway.ports.trace_storage_port import (
    SpanRecord,
    TraceBucket,
    TraceBucketGrain,
    TraceDetail,
    TraceFilter,
    TracePage,
    TraceScope,
    TraceStoragePort,
    TraceSummary,
    TraceWrite,
    WriteResult,
)
from gateway.types.trace_tables import TraceRows, TraceTables

TraceTablesBuilder = Callable[[UnitOfWork], TraceTables]
UnitOfWorkOpener = Callable[[], AbstractAsyncContextManager[UnitOfWork]]


def _workspace_ids(scope: TraceScope) -> frozenset[uuid.UUID] | None:
    return None if scope.deployment_wide else scope.workspace_ids


def _span_values(trace: TraceWrite, span: SpanRecord) -> dict[str, Any]:
    return {
        "workspace_id": trace.workspace_id,
        "trace_id": trace.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "kind": span.kind,
        "origin": span.origin,
        "name": span.name,
        "operation": span.operation,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": span.duration_ms,
        "outcome": span.outcome,
        "recovered": span.recovered,
        "error_class": span.error_class,
        "opens_turn": span.opens_turn,
        "model": span.model,
        "provider": span.provider,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "cost_snapshot": span.cost_snapshot,
        "tool_name": span.tool_name,
        "tool_type": span.tool_type,
        "tool_call_id": span.tool_call_id,
        "request_id": span.request_id,
        "otel_trace_id": span.otel_trace_id,
        "otel_span_id": span.otel_span_id,
        "attributes": dict(span.attributes) or None,
    }


def _span_record(row: TraceSpan) -> SpanRecord:
    return SpanRecord(
        span_id=row.span_id,
        kind=row.kind,
        origin=row.origin,
        name=row.name,
        outcome=row.outcome,
        parent_span_id=row.parent_span_id,
        operation=row.operation,
        start_time=row.start_time,
        end_time=row.end_time,
        duration_ms=row.duration_ms,
        recovered=row.recovered,
        error_class=row.error_class,
        opens_turn=row.opens_turn,
        model=row.model,
        provider=row.provider,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cost_snapshot=row.cost_snapshot,
        tool_name=row.tool_name,
        tool_type=row.tool_type,
        tool_call_id=row.tool_call_id,
        request_id=row.request_id,
        otel_trace_id=row.otel_trace_id,
        otel_span_id=row.otel_span_id,
        attributes=row.attributes or {},
    )


def _summary(row: Trace) -> TraceSummary:
    return TraceSummary(
        workspace_id=row.workspace_id,
        trace_id=row.trace_id,
        user_id=row.user_id,
        api_key_id=row.api_key_id,
        session_source=row.session_source,
        harness=row.harness,
        name=row.name,
        started_at=row.started_at,
        last_activity_at=row.last_activity_at,
        step_count=row.step_count,
        span_count=row.span_count,
        error_count=row.error_count,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cost_snapshot=row.cost_snapshot,
    )


def _span_moment(span: SpanRecord, written_at: datetime) -> tuple[datetime, datetime]:
    """The earliest and latest instant a span vouches for, so a trace's window covers it."""
    earliest = span.start_time or span.end_time or written_at
    return earliest, span.end_time or earliest


def _unique_spans(spans: Sequence[SpanRecord]) -> list[SpanRecord]:
    """Drop a span repeated inside one write, so it counts once whatever the store does."""
    seen: dict[str, SpanRecord] = {}
    for span in spans:
        seen.setdefault(span.span_id, span)
    return list(seen.values())


class LocalTraceStorage(TraceStoragePort):
    """Traces in this deployment's own database.

    Each call opens its own Unit of Work on the metering pool and settles before it
    returns, which is the port's contract, so a trace write never shares a
    transaction with accounting and never waits on the request pool.
    """

    def __init__(self, open_unit_of_work: UnitOfWorkOpener, tables: TraceTablesBuilder) -> None:
        self._open = open_unit_of_work
        self._tables = tables

    async def write(self, traces: tuple[TraceWrite, ...]) -> WriteResult:
        accepted = duplicate = rejected = 0
        written_at = datetime.now(UTC)
        by_workspace: dict[uuid.UUID, list[TraceWrite]] = defaultdict(list)
        for trace in traces:
            by_workspace[trace.workspace_id].append(trace)

        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                for workspace_id, writes in by_workspace.items():
                    for trace in writes:
                        await tables.traces.create_if_absent(self._new_trace(trace, written_at))
                    # Read after creating, so a trace a concurrent writer created first
                    # is judged by the owner it actually has.
                    owners = await tables.traces.owners(workspace_id, {trace.trace_id for trace in writes})
                    for trace in writes:
                        spans = _unique_spans(trace.spans)
                        if owners.get(trace.trace_id) != trace.user_id:
                            rejected += len(spans)
                            continue
                        inserted = await tables.spans.insert_new([_span_values(trace, span) for span in spans])
                        new_spans = [span for span in spans if span.span_id in inserted]
                        accepted += len(new_spans)
                        duplicate += len(spans) - len(new_spans)
                        if new_spans:
                            await self._grow(tables.traces, trace, new_spans, written_at)
        return WriteResult(accepted=accepted, duplicate=duplicate, rejected=rejected)

    @staticmethod
    def _new_trace(trace: TraceWrite, written_at: datetime) -> dict[str, Any]:
        moments = [_span_moment(span, written_at) for span in trace.spans] or [(written_at, written_at)]
        return {
            "workspace_id": trace.workspace_id,
            "trace_id": trace.trace_id,
            "user_id": trace.user_id,
            "api_key_id": trace.api_key_id,
            "session_source": trace.session_source,
            "harness": trace.harness,
            "name": trace.name,
            "started_at": min(earliest for earliest, _ in moments),
            "last_activity_at": max(latest for _, latest in moments),
        }

    @staticmethod
    async def _grow(rows: TraceRows, trace: TraceWrite, spans: list[SpanRecord], written_at: datetime) -> None:
        moments = [_span_moment(span, written_at) for span in spans]
        llm = [span for span in spans if span.kind == "llm"]
        await rows.add_totals(
            trace.workspace_id,
            trace.trace_id,
            steps=sum(1 for span in spans if span.kind == "step"),
            spans=len(spans),
            errors=sum(1 for span in spans if span.outcome == "error" and not span.recovered),
            input_tokens=sum(span.input_tokens or 0 for span in llm),
            output_tokens=sum(span.output_tokens or 0 for span in llm),
            cost=sum((span.cost_snapshot or Decimal(0) for span in llm), Decimal(0)),
            earliest=min(earliest for earliest, _ in moments),
            latest=max(latest for _, latest in moments),
        )

    async def search(self, scope: TraceScope, filters: TraceFilter, *, limit: int, offset: int) -> TracePage:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                rows = await tables.traces.page(_workspace_ids(scope), filters, limit=limit + 1, offset=offset)
                return TracePage(items=tuple(_summary(row) for row in rows[:limit]), has_more=len(rows) > limit)

    async def count(self, scope: TraceScope, filters: TraceFilter) -> int:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                return await tables.traces.count_matching(_workspace_ids(scope), filters)

    async def get(self, scope: TraceScope, trace_id: str, *, span_limit: int) -> TraceDetail | None:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                row = await tables.traces.find(_workspace_ids(scope), trace_id)
                if row is None:
                    return None
                spans = await tables.spans.for_trace(row.workspace_id, row.trace_id, limit=span_limit + 1)
                return TraceDetail(
                    summary=_summary(row),
                    spans=tuple(_span_record(span) for span in spans[:span_limit]),
                    truncated=len(spans) > span_limit,
                )

    async def series(
        self, scope: TraceScope, filters: TraceFilter, *, bucket: TraceBucketGrain
    ) -> tuple[TraceBucket, ...]:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                rows = await tables.traces.bucket_counts(_workspace_ids(scope), filters, bucket=bucket)
                return tuple(TraceBucket(bucket=key, succeeded=ok, failed=bad) for key, ok, bad in rows)

    async def purge(self, scope: TraceScope, filters: TraceFilter) -> int:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                return await tables.traces.delete_matching(_workspace_ids(scope), filters)

    async def purge_user(self, user_id: str) -> int:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                return await tables.traces.delete_for_user(user_id)

    async def expire(self, before: datetime) -> int:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                return await tables.traces.delete_inactive_before(before)


class NullTraceStorage(TraceStoragePort):
    """Keeps no traces: every span is dropped and every read is empty.

    Bound where the deployment holds no trace store and has no peer to send one
    to, so recording costs nothing and the rest of the gateway never asks
    whether a store exists.
    """

    async def write(self, traces: tuple[TraceWrite, ...]) -> WriteResult:
        return WriteResult()

    async def search(self, scope: TraceScope, filters: TraceFilter, *, limit: int, offset: int) -> TracePage:
        return TracePage(items=(), has_more=False)

    async def count(self, scope: TraceScope, filters: TraceFilter) -> int:
        return 0

    async def get(self, scope: TraceScope, trace_id: str, *, span_limit: int) -> TraceDetail | None:
        return None

    async def series(
        self, scope: TraceScope, filters: TraceFilter, *, bucket: TraceBucketGrain
    ) -> tuple[TraceBucket, ...]:
        return ()

    async def purge(self, scope: TraceScope, filters: TraceFilter) -> int:
        return 0

    async def purge_user(self, user_id: str) -> int:
        return 0

    async def expire(self, before: datetime) -> int:
        return 0
