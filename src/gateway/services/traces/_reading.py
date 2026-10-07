"""Reading traces back: the session list, one session with its turns, and the per-bucket counts.

A trace is stored as spans; what a person reads is turns. A turn is derived here,
from step order, each time a trace is read, so a span that arrives late is part
of the answer as soon as it is stored and no stored state can disagree with it:

- a step whose last input was the user's own text opens a turn, and the steps
  after it, which hand tool results back, belong to it;
- a tool the client ran ends when the next step arrived, and starts when the
  step before it ended, which is when the model asked for it, so its duration
  is marked approximate;
- a turn's state follows its last step: failed when that step failed, active
  while the session has been busy within the idle window, completed otherwise.
  A failed attempt a fallback recovered does not fail anything.
"""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.exceptions.traces_exceptions import TraceNotFoundError
from gateway.ports.trace_storage_port import (
    SpanRecord,
    TraceBucket,
    TraceBucketGrain,
    TraceFilter,
    TracePage,
    TraceScope,
    TraceStoragePort,
)
from gateway.types.trace_views import TraceView, TurnState, TurnView

# How long a session may be quiet and still count as in progress.
IDLE_TIMEOUT = timedelta(minutes=10)
# The most spans one read returns; past it the detail says it was cut.
MAX_SPANS_PER_READ = 5_000


@dataclass
class _Turn:
    continued: bool
    steps: list[SpanRecord] = field(default_factory=list)
    children: list[SpanRecord] = field(default_factory=list)


def _moment(span: SpanRecord) -> datetime:
    return span.start_time or span.end_time or datetime.min.replace(tzinfo=UTC)


def _fill_client_tool_starts(spans: list[SpanRecord], steps: list[SpanRecord]) -> tuple[list[SpanRecord], set[str]]:
    """Give each client-run tool the end of the step before the one that answered it as its start."""
    approximate: set[str] = set()
    filled: list[SpanRecord] = []
    for span in spans:
        if span.kind == "tool" and span.start_time is None and span.end_time is not None:
            before = [step for step in steps if step.end_time is not None and step.end_time <= span.end_time]
            if before:
                start = before[-1].end_time
                assert start is not None
                duration = max(0, round((span.end_time - start).total_seconds() * 1000))
                span = replace(span, start_time=start, duration_ms=duration, parent_span_id=before[-1].span_id)
                approximate.add(span.span_id)
        filled.append(span)
    return filled, approximate


def _turns(spans: list[SpanRecord], *, now: datetime, last_activity: datetime) -> tuple[TurnView, ...]:
    steps = sorted((span for span in spans if span.kind == "step"), key=_moment)
    turns: list[_Turn] = []
    owner: dict[str, _Turn] = {}
    for step in steps:
        if step.opens_turn or not turns:
            turns.append(_Turn(continued=not step.opens_turn))
        turns[-1].steps.append(step)
        owner[step.span_id] = turns[-1]
    for span in spans:
        if span.kind != "step" and span.parent_span_id in owner:
            owner[span.parent_span_id].children.append(span)
    # A gateway tool loop records each model round as an unbilled LLM span under the billed one.
    llm_ids = {span.span_id for span in spans if span.kind == "llm"}
    rounds: dict[str, int] = {}
    for span in spans:
        if span.kind == "llm" and span.parent_span_id in llm_ids:
            assert span.parent_span_id is not None
            rounds[span.parent_span_id] = rounds.get(span.parent_span_id, 0) + 1
    recent = now - last_activity < IDLE_TIMEOUT
    views: list[TurnView] = []
    for index, turn in enumerate(turns):
        last = turn.steps[-1]
        if last.outcome == "error":
            state: TurnState = "incomplete" if last.error_class == "incomplete" else "failed"
        elif recent and index == len(turns) - 1:
            state = "active"
        else:
            state = "completed"
        llm = [span for span in turn.children if span.kind == "llm"]
        views.append(
            TurnView(
                index=index,
                state=state,
                started_at=turn.steps[0].start_time,
                ended_at=last.end_time,
                step_ids=tuple(step.span_id for step in turn.steps),
                tool_calls=sum(1 for span in turn.children if span.kind == "tool"),
                llm_calls=sum(rounds.get(span.span_id, 1) for span in llm),
                errors=sum(1 for span in turn.children + turn.steps if span.outcome == "error" and not span.recovered),
                input_tokens=sum(span.input_tokens or 0 for span in llm),
                output_tokens=sum(span.output_tokens or 0 for span in llm),
                cost=sum((span.cost_snapshot or Decimal(0) for span in llm), Decimal(0)),
                continued=turn.continued,
            )
        )
    return tuple(views)


class TraceService:
    """Reads traces inside a caller's scope. Recording does not go through here; it needs no read."""

    def __init__(self, store: TraceStoragePort) -> None:
        self._store = store

    async def search(self, scope: TraceScope, filters: TraceFilter, *, limit: int, offset: int) -> TracePage:
        return await self._store.search(scope, filters, limit=limit, offset=offset)

    async def count(self, scope: TraceScope, filters: TraceFilter) -> int:
        return await self._store.count(scope, filters)

    async def series(
        self, scope: TraceScope, filters: TraceFilter, *, bucket: TraceBucketGrain
    ) -> tuple[TraceBucket, ...]:
        return await self._store.series(scope, filters, bucket=bucket)

    async def detail(self, scope: TraceScope, trace_id: str) -> TraceView:
        """One session with its turns.

        Raises:
            TraceNotFoundError: the scope holds no such trace, including one in a workspace outside it.
        """
        detail = await self._store.get(scope, trace_id, span_limit=MAX_SPANS_PER_READ)
        if detail is None:
            raise TraceNotFoundError(trace_id)
        now = datetime.now(UTC)
        steps = sorted((span for span in detail.spans if span.kind == "step"), key=_moment)
        spans, approximate = _fill_client_tool_starts(list(detail.spans), steps)
        spans.sort(key=_moment)
        last_activity = detail.summary.last_activity_at
        return TraceView(
            summary=detail.summary,
            state="active" if now - last_activity < IDLE_TIMEOUT else "idle",
            turns=_turns(spans, now=now, last_activity=last_activity),
            spans=tuple(spans),
            approximate=frozenset(approximate),
            truncated=detail.truncated,
            content_span_ids=detail.content_span_ids,
        )
