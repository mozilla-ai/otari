"""Response models for the trace read API, and their mapping from the traces service's views."""

import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gateway.models.money import as_float
from gateway.types.trace_views import TraceView, TurnView
from gateway.types.traces import (
    BREAK_GLASS_REASON_MAX_LENGTH,
    BREAK_GLASS_REASON_MIN_LENGTH,
    BREAK_GLASS_REFERENCE_MAX_LENGTH,
    SHARE_NOTE_MAX_LENGTH,
    BreakGlassKind,
    BreakGlassRead,
    ContentAccess,
    ContentRead,
    ContentShare,
    ShareAudience,
    SpanRecord,
    TraceBucket,
    TraceBucketGrain,
    TracePage,
    TraceSummary,
)
from otari_agent.domain.definitions import DIGEST_LENGTH, MAX_DEFINITION_GATES, MAX_GATE_MESSAGE_LENGTH, MAX_GATE_RUNS
from otari_agent.domain.policy import MAX_GATE_ID_LENGTH
from otari_agent.domain.types import GateType, RunsAt


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
    error_count: int = Field(
        description="Spans that ended in an error no fallback recovered, including tools' errors, "
        "which do not fail the session."
    )
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
    succeeded: int = Field(description="Sessions active in the bucket that did not fail.")
    failed: int = Field(
        description="Sessions active in the bucket that hold an unrecovered error other than a tool's. "
        "A tool's error, an MCP tool's included, goes back to the model, so it alone does not fail a session."
    )


class TraceSeriesPublic(BaseModel):
    bucket: TraceBucketGrain
    points: list[TraceBucketPublic]


class SpanPublic(BaseModel):
    """One span of a session. Its content is empty unless the deployment keeps content."""

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
    outcome: Literal["ok", "error", "unknown", "blocked"]
    recovered: bool
    error_class: str | None
    opens_turn: bool
    model: str | None
    provider: str | None
    input_tokens: int | None = Field(description="The whole prompt, cached tokens included.")
    output_tokens: int | None
    cache_read_tokens: int | None = Field(description="Input tokens read from the prompt cache.")
    cache_write_tokens: int | None = Field(description="Input tokens written to the prompt cache.")
    reasoning_tokens: int | None = Field(description="Output tokens spent reasoning.")
    context_window: int | None = Field(
        description="The model's context window in tokens, where the catalog knows it. Null on a span that is "
        "not a model request, and for a model the catalog does not list."
    )
    cost: float | None
    tool_name: str | None
    tool_type: str | None
    tool_call_id: str | None
    request_id: str | None
    otel_trace_id: str | None = Field(
        description="The OpenTelemetry trace id an OTLP span was exported with, in lowercase hex. "
        "Null on a span the gateway recorded itself."
    )
    otel_span_id: str | None = Field(
        description="The OpenTelemetry span id an OTLP span was exported with, in lowercase hex. "
        "Null on a span the gateway recorded itself."
    )
    attributes: dict[str, str | int | float | bool]
    content: dict[str, str] = Field(
        description=(
            "What the span was about (prompt, command, file_path, gate.evidence, gate.detail), only where the "
            "workspace keeps content, the agent sent it, and the reader may open it. Empty otherwise."
        )
    )
    content_private: bool = Field(
        default=False,
        description=(
            "The span holds content this reader may not open: it is private to the session's own user, unless "
            "they share the session."
        ),
    )


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
    cache_read_tokens: int
    cost: float


class GateSummaryPublic(BaseModel):
    """How often one agent guardrail gate ran in a session, by what it decided."""

    gate_id: str
    passed: int
    blocked: int = Field(description="Checks where the gate stopped the agent.")
    flagged: int = Field(description="Checks an advisory gate failed; the agent went ahead.")
    refused: int = Field(description="Checks where the gate could not decide and held the agent anyway.")
    undecided: int = Field(description="Checks where the gate could not decide; the agent went ahead unchecked.")
    span_id: str | None = Field(
        description="The check to open first: the earliest that did anything but pass. Null for a gate that never ran."
    )
    gate_type: str | None = Field(description="From the guardrail's definition, when it was sent.")
    enforcement: Literal["required", "advisory"] | None
    message: str | None = Field(description="What the gate says when it fails, from the guardrail's definition.")
    runs: list[str] = Field(description="The moments the gate runs at (pre_tool_use.edit_target, stop.session, ...).")
    origin: Literal["repo", "user"] | None = Field(
        description="Whether the gate is the repository's (.otari/) or the user's own (~/.otari/)."
    )


class ToolCostPublic(BaseModel):
    """An estimate of what one tool call cost, from the model requests either side of it."""

    span_id: str
    total: float
    calling: float = Field(description="Its share of the output of the request that called it.")
    writing: float = Field(description="Writing its result into the context, on the next request.")
    rereading: float = Field(description="Reading its result again in later requests, until the context was compacted.")
    result_tokens: int = Field(description="About how many tokens its result added to the context.")
    later_requests: int = Field(description="The later requests that read its result again.")
    calls_sharing_request: int = Field(description="Tool calls the request that called it made, sharing its output.")
    calls_sharing_result: int = Field(description="Tool calls whose results arrived together, sharing that growth.")
    subagent: float | None = Field(description="What a subagent the call ran cost, exactly.")


class TraceContentPolicyPublic(BaseModel):
    """What this deployment keeps of agent sessions, and who besides a session's owner can open it."""

    content_available: bool = Field(
        description=(
            "Whether this deployment lets a workspace keep what its agent sessions were about. A workspace "
            "keeps it only once its managers turn it on as well."
        )
    )
    platform_break_glass: bool = Field(
        description=(
            "Whether the deployment's operator may open any organization's session content without its owner's "
            "share, recorded with a reason and a reference."
        )
    )
    retention_days: int = Field(
        description=("Days after its last activity that a session, with what it was about and its shares, is deleted.")
    )
    session_max_age_days: int = Field(
        description="Days after it started that a session is deleted, however active it still is."
    )


class ContentReadPublic(BaseModel):
    """One read of a session's content through a share."""

    reader_user_id: uuid.UUID | None = Field(description="Who read it; null once that account is gone.")
    reader_email: str = Field(description="Their email address when they read it.")
    read_at: datetime


class ContentSharePublic(BaseModel):
    """A session's owner letting an audience read what it was about."""

    id: uuid.UUID
    audience: ShareAudience = Field(
        description="Who may read: the organization's owners and admins, or the deployment's operator."
    )
    note: str | None = Field(description="Why it was shared, in its owner's words.")
    created_at: datetime
    revoked_at: datetime | None = Field(description="When it ended: its owner ended it, or lost the workspace.")
    active: bool = Field(description="Whether it still lets its audience read.")
    reads: list[ContentReadPublic] = Field(
        default_factory=list,
        description=(
            "Its newest reads, newest first; GET .../shares/{share_id}/reads pages through all of them. A reader's "
            "reads are recorded at most once an hour. Only its owner sees them."
        ),
    )
    read_count: int = Field(default=0, description="How many reads it has had in all.")


class ContentShareListPublic(BaseModel):
    items: list[ContentSharePublic]
    has_more: bool = Field(default=False, description="More shares follow this page.")


class ContentReadListPublic(BaseModel):
    items: list[ContentReadPublic]
    has_more: bool = Field(description="More reads follow this page.")


class ContentShareCreate(BaseModel):
    """Let an audience read what one of your sessions was about."""

    audience: ShareAudience
    note: str | None = Field(default=None, max_length=SHARE_NOTE_MAX_LENGTH)


class BreakGlassCreate(BaseModel):
    """Why you are opening a session's content without its owner's share. Recorded with the opening."""

    # Stripped first, so a reason of spaces is no reason.
    model_config = ConfigDict(str_strip_whitespace=True)

    reason: str = Field(min_length=BREAK_GLASS_REASON_MIN_LENGTH, max_length=BREAK_GLASS_REASON_MAX_LENGTH)


class PlatformBreakGlassCreate(BreakGlassCreate):
    """Why the deployment's operator is opening a session's content, and what asked for it."""

    reference: str = Field(
        min_length=1,
        max_length=BREAK_GLASS_REFERENCE_MAX_LENGTH,
        description="What the opening answers: a legal request's number, a ticket.",
    )


class BreakGlassReadPublic(BaseModel):
    """One recorded opening of a session's content without its owner's share."""

    id: uuid.UUID
    kind: BreakGlassKind = Field(
        description="'organization' for an owner of the organization, 'deployment' for the deployment's operator."
    )
    organization_id: uuid.UUID
    workspace_id: uuid.UUID
    trace_id: str
    session_user_id: str | None = Field(description="Whose session it is.")
    reader_user_id: uuid.UUID | None = Field(description="Who opened it; null once that account is gone.")
    reader_email: str = Field(description="Their email address when they opened it.")
    reason: str
    reference: str | None
    read_at: datetime


class BreakGlassReadListPublic(BaseModel):
    items: list[BreakGlassReadPublic]
    has_more: bool


def break_glass_read_public(read: BreakGlassRead) -> BreakGlassReadPublic:
    return BreakGlassReadPublic(
        id=read.id,
        kind=read.kind,
        organization_id=read.organization_id,
        workspace_id=read.workspace_id,
        trace_id=read.trace_id,
        session_user_id=read.session_user_id,
        reader_user_id=read.reader_user_id,
        reader_email=read.reader_email,
        reason=read.reason,
        reference=read.reference,
        read_at=read.read_at,
    )


class TraceDetailPublic(BaseModel):
    summary: TraceSummaryPublic
    state: Literal["active", "idle"]
    turns: list[TurnPublic]
    spans: list[SpanPublic]
    truncated: bool = Field(description="True when the session holds more spans than one read returns.")
    gates: list[GateSummaryPublic] = Field(
        description="Each agent guardrail gate a coding agent's hook reported, the ones that did something first.",
    )
    tool_costs: list[ToolCostPublic] = Field(
        description="An estimate for each tool call between two priced model requests, in the order they were made.",
    )
    content_access: ContentAccess = Field(
        default="none",
        description=(
            "How this read reached the session's content: 'owner' for its own user, 'share' through a share its "
            "user made, 'break_glass' for a recorded opening without one, 'none' otherwise."
        ),
    )
    content_share: ContentSharePublic | None = Field(
        default=None, description="The share the content was read through, with its note. Its reads are not listed."
    )


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


def series_public(bucket: TraceBucketGrain, points: tuple[TraceBucket, ...]) -> TraceSeriesPublic:
    return TraceSeriesPublic(
        bucket=bucket,
        points=[TraceBucketPublic(bucket=p.bucket, succeeded=p.succeeded, failed=p.failed) for p in points],
    )


def _span_public(span: SpanRecord, approximate: frozenset[str], context_windows: Mapping[str, int]) -> SpanPublic:
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
        cache_read_tokens=span.cache_read_tokens,
        cache_write_tokens=span.cache_write_tokens,
        reasoning_tokens=span.reasoning_tokens,
        context_window=context_windows.get(span.span_id),
        cost=as_float(span.cost_snapshot),
        tool_name=span.tool_name,
        tool_type=span.tool_type,
        tool_call_id=span.tool_call_id,
        request_id=span.request_id,
        otel_trace_id=span.otel_trace_id,
        otel_span_id=span.otel_span_id,
        attributes=dict(span.attributes),
        content=dict(span.content),
        content_private=span.content_private,
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
        cache_read_tokens=turn.cache_read_tokens,
        cost=as_float(turn.cost) or 0.0,
    )


def share_public(share: ContentShare) -> ContentSharePublic:
    return ContentSharePublic(
        id=share.id,
        audience=share.audience,
        note=share.note,
        created_at=share.created_at,
        revoked_at=share.revoked_at,
        active=share.active,
        reads=[read_public(read) for read in share.reads],
        read_count=share.read_count,
    )


def read_public(read: ContentRead) -> ContentReadPublic:
    return ContentReadPublic(reader_user_id=read.reader_user_id, reader_email=read.reader_email, read_at=read.read_at)


def detail_public(view: TraceView) -> TraceDetailPublic:
    return TraceDetailPublic(
        summary=summary_public(view.summary),
        state=view.state,
        turns=[_turn_public(turn) for turn in view.turns],
        spans=[_span_public(span, view.approximate, view.context_windows) for span in view.spans],
        truncated=view.truncated,
        gates=[
            GateSummaryPublic(
                gate_id=gate.gate_id,
                passed=gate.passed,
                blocked=gate.blocked,
                flagged=gate.flagged,
                refused=gate.refused,
                undecided=gate.undecided,
                span_id=gate.span_id,
                gate_type=gate.gate_type,
                enforcement=gate.enforcement,  # type: ignore[arg-type]
                message=gate.message,
                runs=list(gate.runs),
                origin=gate.origin,  # type: ignore[arg-type]
            )
            for gate in view.gates
        ],
        tool_costs=[
            ToolCostPublic(
                span_id=cost.span_id,
                total=float(cost.total),
                calling=float(cost.calling),
                writing=float(cost.writing),
                rereading=float(cost.rereading),
                result_tokens=cost.result_tokens,
                later_requests=cost.later_requests,
                calls_sharing_request=cost.calls_sharing_request,
                calls_sharing_result=cost.calls_sharing_result,
                subagent=float(cost.subagent) if cost.subagent is not None else None,
            )
            for cost in view.tool_costs
        ],
        content_access=view.content_access,
        content_share=share_public(view.content_share) if view.content_share is not None else None,
    )


class GateDefinitionRequest(BaseModel):
    """One gate of a guardrail version: what it is, when it runs, and what it says when it fails."""

    model_config = ConfigDict(extra="forbid")

    gate_id: str = Field(min_length=1, max_length=MAX_GATE_ID_LENGTH)
    gate_type: GateType
    enforcement: Literal["required", "advisory"]
    origin: Literal["repo", "user"]
    runs: list[RunsAt] = Field(max_length=MAX_GATE_RUNS)
    message: str = Field(max_length=MAX_GATE_MESSAGE_LENGTH)


class GuardrailDefinitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    digest: str = Field(pattern=rf"^[0-9a-f]{{{DIGEST_LENGTH}}}$")
    gates: list[GateDefinitionRequest] = Field(min_length=1, max_length=MAX_DEFINITION_GATES)

    @model_validator(mode="after")
    def _one_definition_per_gate(self) -> "GuardrailDefinitionRequest":
        ids = [gate.gate_id for gate in self.gates]
        if len(ids) != len(set(ids)):
            raise ValueError("each gate_id may appear once")
        return self


class GuardrailDefinitionResponse(BaseModel):
    digest: str
