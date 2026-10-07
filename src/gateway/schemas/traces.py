"""Response models for the trace read API, and their mapping from the traces service's views."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from gateway.models.money import as_float
from gateway.ports.trace_storage_port import SpanRecord, TraceBucket, TracePage, TraceSummary
from gateway.types.trace_views import TraceView, TurnView


class TraceSummaryPublic(BaseModel):
    """One agent session as the session list shows it."""

    trace_id: str
    workspace_id: uuid.UUID
    user_id: str | None
    api_key_id: str | None
    session_source: Literal["client", "harness", "otlp", "none"] = Field(
        description="What grouped the session's requests: the client's own id, its agent harness's id, "
        "an OTLP conversation id, or nothing (a single request)."
    )
    harness: str | None = Field(description="The agent or client that sent the requests, from its User-Agent.")
    name: str | None
    started_at: datetime
    last_activity_at: datetime
    step_count: int = Field(description="Requests the gateway served in the session.")
    span_count: int
    error_count: int = Field(description="Spans that ended in an error no fallback recovered.")
    input_tokens: int
    output_tokens: int
    cost: float = Field(description="USD, a snapshot of the usage rows' cost. Usage stays the billing record.")


class TraceListPublic(BaseModel):
    items: list[TraceSummaryPublic]
    has_more: bool


class TraceCountPublic(BaseModel):
    count: int


class TraceBucketPublic(BaseModel):
    bucket: str = Field(description="Bucket start, ISO 8601 UTC.")
    succeeded: int
    failed: int


class TraceSeriesPublic(BaseModel):
    bucket: Literal["hour", "day"]
    points: list[TraceBucketPublic]


class SpanPublic(BaseModel):
    """One span of a session. Content is never part of a span."""

    span_id: str
    parent_span_id: str | None
    kind: str
    origin: Literal["gateway", "otlp"]
    name: str
    operation: str | None
    start_time: datetime | None
    end_time: datetime | None
    duration_ms: int | None
    approximate: bool = Field(description="True when the start was inferred rather than measured.")
    outcome: Literal["ok", "error", "unknown"]
    recovered: bool
    error_class: str | None
    opens_turn: bool
    model: str | None
    provider: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost: float | None
    tool_name: str | None
    tool_type: str | None
    tool_call_id: str | None
    request_id: str | None
    attributes: dict[str, str | int | float | bool]


class TurnPublic(BaseModel):
    """One user prompt and the steps it caused."""

    index: int
    state: Literal["active", "completed", "failed", "incomplete"]
    continued: bool = Field(description="True for steps a session started with before any prompt was seen.")
    started_at: datetime | None
    ended_at: datetime | None
    step_ids: list[str]
    tool_calls: int
    llm_calls: int
    errors: int
    input_tokens: int
    output_tokens: int
    cost: float


class TraceDetailPublic(BaseModel):
    summary: TraceSummaryPublic
    state: Literal["active", "idle"]
    turns: list[TurnPublic]
    spans: list[SpanPublic]
    truncated: bool = Field(description="True when the session holds more spans than one read returns.")


def summary_public(summary: TraceSummary) -> TraceSummaryPublic:
    return TraceSummaryPublic(
        trace_id=summary.trace_id,
        workspace_id=summary.workspace_id,
        user_id=summary.user_id,
        api_key_id=summary.api_key_id,
        session_source=summary.session_source,  # type: ignore[arg-type]
        harness=summary.harness,
        name=summary.name,
        started_at=summary.started_at,
        last_activity_at=summary.last_activity_at,
        step_count=summary.step_count,
        span_count=summary.span_count,
        error_count=summary.error_count,
        input_tokens=summary.input_tokens,
        output_tokens=summary.output_tokens,
        cost=as_float(summary.cost_snapshot) or 0.0,
    )


def list_public(page: TracePage) -> TraceListPublic:
    return TraceListPublic(items=[summary_public(item) for item in page.items], has_more=page.has_more)


def series_public(bucket: Literal["hour", "day"], points: tuple[TraceBucket, ...]) -> TraceSeriesPublic:
    return TraceSeriesPublic(
        bucket=bucket,
        points=[TraceBucketPublic(bucket=p.bucket, succeeded=p.succeeded, failed=p.failed) for p in points],
    )


def _span_public(span: SpanRecord, approximate: frozenset[str]) -> SpanPublic:
    return SpanPublic(
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        kind=span.kind,
        origin=span.origin,  # type: ignore[arg-type]
        name=span.name,
        operation=span.operation,
        start_time=span.start_time,
        end_time=span.end_time,
        duration_ms=span.duration_ms,
        approximate=span.span_id in approximate,
        outcome=span.outcome,  # type: ignore[arg-type]
        recovered=span.recovered,
        error_class=span.error_class,
        opens_turn=span.opens_turn,
        model=span.model,
        provider=span.provider,
        input_tokens=span.input_tokens,
        output_tokens=span.output_tokens,
        cost=as_float(span.cost_snapshot),
        tool_name=span.tool_name,
        tool_type=span.tool_type,
        tool_call_id=span.tool_call_id,
        request_id=span.request_id,
        attributes=dict(span.attributes),
    )


def _turn_public(turn: TurnView) -> TurnPublic:
    return TurnPublic(
        index=turn.index,
        state=turn.state,
        continued=turn.continued,
        started_at=turn.started_at,
        ended_at=turn.ended_at,
        step_ids=list(turn.step_ids),
        tool_calls=turn.tool_calls,
        llm_calls=turn.llm_calls,
        errors=turn.errors,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        cost=as_float(turn.cost) or 0.0,
    )


def detail_public(view: TraceView) -> TraceDetailPublic:
    return TraceDetailPublic(
        summary=summary_public(view.summary),
        state=view.state,
        turns=[_turn_public(turn) for turn in view.turns],
        spans=[_span_public(span, view.approximate) for span in view.spans],
        truncated=view.truncated,
    )
