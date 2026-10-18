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
  A failed attempt a fallback recovered does not fail anything;
- a gate an agent's hook reported hangs under the tool call it checked, by call
  id, or else under the step it ran in (a Stop gate checks no tool call).
  It arrives on its own export, so this is the only place the two meet.
"""

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.types.trace_views import GateSummary, TraceView, TurnState, TurnView
from gateway.types.traces import GuardrailDefinition, SpanRecord, TraceDetail

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
    # By end, not start: the step that ended last, which a long or overlapping step may not be the last to start.
    ended = sorted(
        ((step.end_time, step.span_id, step) for step in steps if step.end_time is not None), key=lambda e: e[:2]
    )
    ends = [end for end, _, _ in ended]
    for span in spans:
        if span.kind == "tool" and span.start_time is None and span.end_time is not None:
            position = bisect_right(ends, span.end_time)
            if position:
                start, _, asker = ended[position - 1]
                duration = max(0, round((span.end_time - start).total_seconds() * 1000))
                span = replace(span, start_time=start, duration_ms=duration, parent_span_id=asker.span_id)
                approximate.add(span.span_id)
        filled.append(span)
    return filled, approximate


def _settle_tools(spans: list[SpanRecord]) -> list[SpanRecord]:
    """Give a tool call what its decision and its run say about it, wherever they were stored.

    An agent may export a tool's children apart from the tool itself, and the
    projection only sees the children of one export, so a refusal or a failed run
    the tool never heard of is settled here.
    """
    decided: dict[str, SpanRecord] = {}
    failed: set[str] = set()
    for span in spans:
        if span.parent_span_id is None:
            continue
        if span.attributes.get("otari.tool.decision") is not None and span.kind != "tool":
            decided[span.parent_span_id] = span
        if span.attributes.get("otari.tool.succeeded") is False:
            failed.add(span.parent_span_id)
    settled: list[SpanRecord] = []
    for span in spans:
        if span.kind == "tool" and span.outcome == "ok":
            decision = decided.get(span.span_id)
            if decision is not None and decision.attributes.get("otari.tool.decision") == "reject":
                span = replace(span, outcome="blocked", attributes={**span.attributes, **decision.attributes})
            elif span.span_id in failed:
                span = replace(span, attributes={**span.attributes, "otari.tool.succeeded": False})
        settled.append(span)
    return settled


def _nearest_tool(candidates: list[SpanRecord], gate: SpanRecord) -> str | None:
    """The tool call a gate checked, among those sharing its call id.

    A client may reuse a call id across turns (``call_1`` each time), so the one
    that started last at or before the check wins; failing that, the earliest.
    """
    if not candidates:
        return None
    at = _moment(gate)
    started = [tool for tool in candidates if _moment(tool) <= at]
    if started:
        return max(started, key=_moment).span_id
    return min(candidates, key=_moment).span_id


def _attach_gates(spans: list[SpanRecord], steps: list[SpanRecord]) -> list[SpanRecord]:
    """Give each reported gate with no parent the tool call it checked, or else the step it ran in."""
    tools: dict[str, list[SpanRecord]] = {}
    for span in spans:
        if span.kind == "tool" and span.tool_call_id:
            tools.setdefault(span.tool_call_id, []).append(span)
    starts = [_moment(step) for step in steps]
    attached: list[SpanRecord] = []
    for span in spans:
        if span.kind == "guardrail" and span.origin == "otlp" and span.parent_span_id is None:
            parent = _nearest_tool(tools.get(span.tool_call_id or "", []), span)
            if parent is None:
                position = bisect_right(starts, _moment(span))
                parent = steps[position - 1].span_id if position else None
            if parent is not None:
                span = replace(span, parent_span_id=parent)
        attached.append(span)
    return attached


_VERDICTS = ("passed", "blocked", "flagged", "refused", "undecided")


def _gate_verdict(span: SpanRecord) -> str:
    outcome = span.attributes.get("otari.gate.outcome")
    if outcome == "pass":
        return "passed"
    stopped = span.attributes.get("otari.gate.blocked") is True
    if outcome == "fail":
        return "blocked" if stopped else "flagged"
    # A required gate that could not decide still holds the agent; an advisory one lets it through.
    return "refused" if stopped else "undecided"


def _gate_summaries(
    spans: list[SpanRecord], definitions: tuple[GuardrailDefinition, ...] = ()
) -> tuple[GateSummary, ...]:
    """Each gate the session's guardrail declares or its hook reported, the ones that did something first.

    A declared gate that never ran is listed with no checks. Where the guardrail
    changed during the session, the version seen last describes each gate.
    """
    counts: dict[str, dict[str, int]] = {}
    first: dict[str, str] = {}
    notable: dict[str, str] = {}
    seen: list[str] = []
    for span in spans:
        gate_id = span.attributes.get("otari.gate.id")
        if span.kind != "guardrail" or not isinstance(gate_id, str):
            continue
        digest = span.attributes.get("otari.policy.digest")
        if isinstance(digest, str) and digest not in seen:
            seen.append(digest)
        verdict = _gate_verdict(span)
        counts.setdefault(gate_id, dict.fromkeys(_VERDICTS, 0))[verdict] += 1
        first.setdefault(gate_id, span.span_id)
        if verdict != "passed":
            notable.setdefault(gate_id, span.span_id)
    by_digest = {definition.digest: definition for definition in definitions}
    declared = {gate.gate_id: gate for digest in seen if digest in by_digest for gate in by_digest[digest].gates}
    summaries = [
        GateSummary(
            gate_id=gate_id,
            span_id=notable.get(gate_id, first.get(gate_id)),
            gate_type=declared[gate_id].gate_type if gate_id in declared else None,
            enforcement=declared[gate_id].enforcement if gate_id in declared else None,
            message=(declared[gate_id].message or None) if gate_id in declared else None,
            runs=declared[gate_id].runs if gate_id in declared else (),
            origin=declared[gate_id].origin if gate_id in declared else None,
            **counts.get(gate_id, dict.fromkeys(_VERDICTS, 0)),
        )
        for gate_id in {**counts, **declared}
    ]

    def order(gate: GateSummary) -> tuple[int, bool, str]:
        acted = gate.blocked + gate.refused + gate.undecided + gate.flagged
        return (-acted, gate.passed + acted == 0, gate.gate_id)

    return tuple(sorted(summaries, key=order))


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
                cache_read_tokens=sum(span.cache_read_tokens or 0 for span in llm),
                cost=sum((span.cost_snapshot or Decimal(0) for span in llm), Decimal(0)),
                continued=turn.continued,
            )
        )
    return tuple(views)


def trace_view(detail: TraceDetail, *, now: datetime) -> TraceView:
    """One stored trace as a person reads it: its spans in order, grouped into turns."""
    steps = sorted((span for span in detail.spans if span.kind == "step"), key=_moment)
    spans, approximate = _fill_client_tool_starts(_settle_tools(_attach_gates(list(detail.spans), steps)), steps)
    spans.sort(key=_moment)
    last_activity = detail.summary.last_activity_at
    turns = _turns(spans, now=now, last_activity=last_activity)
    return TraceView(
        summary=detail.summary,
        state="active" if now - last_activity < IDLE_TIMEOUT else "idle",
        turns=turns,
        spans=tuple(spans),
        approximate=frozenset(approximate),
        truncated=detail.truncated,
        gates=_gate_summaries(spans, detail.definitions),
    )
