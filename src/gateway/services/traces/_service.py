"""The traces use cases: storing a session's spans, reading sessions back, and erasing them."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import Trace, TraceSpan
from gateway.repositories.traces import TraceGrowth, TraceKey, TracesRepositories
from gateway.types.traces import (
    SpanRecord,
    TraceDetail,
    TraceFilter,
    TracePage,
    TraceScope,
    TraceSummary,
    TraceWrite,
    WriteResult,
)


class TraceService:
    """Agent traces in this deployment's database, inside a tenant scope.

    A write is idempotent on ``(workspace_id, trace_id, span_id)``: a span repeated
    is counted as a duplicate, and a stored span is never changed. A span whose
    trace already belongs to another user is rejected, never merged. Each call
    settles in a block of its own on the Unit of Work it was built with.
    """

    def __init__(self, uow: UnitOfWork, repositories: TracesRepositories) -> None:
        self._uow = uow
        self._traces = repositories.traces
        self._spans = repositories.spans

    async def write(self, traces: Sequence[TraceWrite]) -> WriteResult:
        """Store these traces' spans, creating each trace on its first span.

        A fixed number of statements however many traces the batch holds: create
        the traces not stored yet, read their owners, insert the spans, grow the
        totals. Each runs in key order, so two writers sharing traces lock them in
        the same order.
        """
        written_at = datetime.now(UTC)
        async with self._uow:
            new_traces: dict[TraceKey, dict[str, Any]] = {}
            for trace in traces:
                new_traces.setdefault(_key(trace), _new_trace(trace, written_at))
            await self._traces.create_absent(list(new_traces.values()))
            # Read after creating, so a trace a concurrent writer created first is
            # judged by the owner it actually has.
            holders = await self._traces.holders(new_traces.keys())

            rejected = 0
            kept: list[tuple[TraceWrite, list[SpanRecord]]] = []
            rows: dict[tuple[uuid.UUID, str, str], dict[str, Any]] = {}
            for trace in traces:
                spans = _unique_spans(trace.spans)
                holder = holders.get(_key(trace))
                if holder is None or holder[0] != trace.user_id:
                    rejected += len(spans)
                    continue
                kept.append((trace, spans))
                for span in spans:
                    rows.setdefault((trace.workspace_id, trace.trace_id, span.span_id), _span_values(trace, span))
            inserted = await self._spans.insert_new(list(rows.values()))

            accepted = duplicate = 0
            grown: dict[TraceKey, list[SpanRecord]] = defaultdict(list)
            for trace, spans in kept:
                for span in spans:
                    key = (trace.workspace_id, trace.trace_id, span.span_id)
                    if key in inserted:
                        # Claimed once, so a span repeated across this batch's writes counts once.
                        inserted.discard(key)
                        grown[_key(trace)].append(span)
                        accepted += 1
                    else:
                        duplicate += 1
            await self._traces.add_totals([_growth(key, spans, written_at) for key, spans in grown.items()])
        return WriteResult(accepted=accepted, duplicate=duplicate, rejected=rejected)

    async def search(self, scope: TraceScope, filters: TraceFilter, *, limit: int, offset: int) -> TracePage:
        """Return one page of the scope's traces, newest activity first."""
        async with self._uow:
            rows = await self._traces.page(_workspace_ids(scope), filters, limit=limit + 1, offset=offset)
            return TracePage(items=tuple(_summary(row) for row in rows[:limit]), has_more=len(rows) > limit)

    async def count(self, scope: TraceScope, filters: TraceFilter) -> int:
        """Return how many of the scope's traces match."""
        async with self._uow:
            return await self._traces.count_matching(_workspace_ids(scope), filters)

    async def get(
        self, scope: TraceScope, trace_id: str, *, span_limit: int, workspace_id: uuid.UUID | None = None
    ) -> TraceDetail | None:
        """Return one trace with up to ``span_limit`` spans, or None when the scope holds no such trace.

        A trace in another tenant's workspace reads as absent, never as forbidden.
        """
        async with self._uow:
            row = await self._traces.find(_workspace_ids(scope), trace_id, workspace_id=workspace_id)
            if row is None:
                return None
            spans = await self._spans.for_trace(row.workspace_id, row.trace_id, limit=span_limit + 1)
            return TraceDetail(
                summary=_summary(row),
                spans=tuple(_span_record(span) for span in spans[:span_limit]),
                truncated=len(spans) > span_limit,
            )

    async def purge_user(self, user_id: str) -> int:
        """Delete every trace a user owns, in every workspace, for erasure."""
        async with self._uow:
            return await self._traces.delete_for_user(user_id)

    async def expire(self, *, idle_before: datetime, started_before: datetime) -> int:
        """Delete every trace idle since before ``idle_before``, or started before ``started_before``.

        The second bound is what ends a session a client keeps alive: its id then
        starts a new trace. Returns the traces deleted.
        """
        async with self._uow:
            return await self._traces.delete_expired(idle_before=idle_before, started_before=started_before)


def _workspace_ids(scope: TraceScope) -> frozenset[uuid.UUID] | None:
    return None if scope.deployment_wide else scope.workspace_ids


def _span_moment(span: SpanRecord, written_at: datetime) -> tuple[datetime, datetime]:
    """The earliest and latest instant a span vouches for, so a trace's window covers it."""
    earliest = span.start_time or span.end_time or written_at
    return earliest, span.end_time or earliest


def _unique_spans(spans: Sequence[SpanRecord]) -> list[SpanRecord]:
    """Drop a span repeated inside one write, so it counts once."""
    seen: dict[str, SpanRecord] = {}
    for span in spans:
        seen.setdefault(span.span_id, span)
    return list(seen.values())


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


def _key(trace: TraceWrite) -> TraceKey:
    return (trace.workspace_id, trace.trace_id)


def _growth(key: TraceKey, spans: list[SpanRecord], written_at: datetime) -> TraceGrowth:
    moments = [_span_moment(span, written_at) for span in spans]
    llm = [span for span in spans if span.kind == "llm"]
    return TraceGrowth(
        workspace_id=key[0],
        trace_id=key[1],
        steps=sum(1 for span in spans if span.kind == "step"),
        spans=len(spans),
        errors=sum(1 for span in spans if span.outcome == "error" and not span.recovered),
        input_tokens=sum(span.input_tokens or 0 for span in llm),
        output_tokens=sum(span.output_tokens or 0 for span in llm),
        cost=sum((span.cost_snapshot or Decimal(0) for span in llm), Decimal(0)),
        earliest=min(earliest for earliest, _ in moments),
        latest=max(latest for _, latest in moments),
    )


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
