"""How a read derives a turn's totals from its spans."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.services.traces._reading import (
    _attach_gates,
    _fill_client_tool_starts,
    _gate_summaries,
    _price_requests,
    _settle_tools,
    _turns,
)
from gateway.types.traces import SpanRecord

_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)


def _span(span_id: str, kind: str, parent: str | None = None, **fields: object) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        kind=kind,
        origin="gateway",
        name="x",
        outcome="ok",
        parent_span_id=parent,
        start_time=_T0,
        end_time=_T0 + timedelta(seconds=1),
        **fields,  # type: ignore[arg-type]
    )


def test_each_round_of_a_tool_loop_is_an_llm_call_and_only_the_billed_span_is_counted_for_cost() -> None:
    spans = [
        _span("req-1", "step", opens_turn=True),
        _span(
            "llm-1",
            "llm",
            "req-1",
            input_tokens=30,
            output_tokens=6,
            cache_read_tokens=20,
            cost_snapshot=Decimal("0.03"),
        ),
        _span("round-1", "llm", "llm-1", attributes={"otari.tool_loop.round": 1}),
        _span("round-2", "llm", "llm-1", attributes={"otari.tool_loop.round": 2}),
        _span("round-3", "llm", "llm-1", attributes={"otari.tool_loop.round": 3}),
        _span("req-2", "step"),
        _span(
            "llm-2", "llm", "req-2", input_tokens=5, output_tokens=1, cache_read_tokens=4, cost_snapshot=Decimal("0.01")
        ),
    ]

    [turn] = _turns(spans, now=_T0 + timedelta(hours=1), last_activity=_T0)

    assert turn.llm_calls == 4
    assert (turn.input_tokens, turn.output_tokens, turn.cache_read_tokens, turn.cost) == (35, 7, 24, Decimal("0.04"))


def test_a_client_tool_starts_when_the_step_that_ended_last_before_it_ended() -> None:
    """A long step can start first and end last, so the asker is chosen by end, not start."""
    long_step = replace(_span("long", "step"), end_time=_T0 + timedelta(seconds=9))
    short_step = replace(
        _span("short", "step"), start_time=_T0 + timedelta(seconds=2), end_time=_T0 + timedelta(seconds=3)
    )
    tool = SpanRecord(
        span_id="call-1", kind="tool", origin="gateway", name="x", outcome="ok", end_time=_T0 + timedelta(seconds=10)
    )

    [filled, *_], approximate = _fill_client_tool_starts([tool], [long_step, short_step])

    assert (filled.parent_span_id, filled.start_time) == ("long", _T0 + timedelta(seconds=9))
    assert approximate == {"call-1"}


def test_each_client_tool_in_a_long_session_starts_after_the_step_that_asked_for_it() -> None:
    """A long session answers a tool every step; each must find its own asker, in any order."""
    steps = [
        replace(
            _span(f"req-{n}", "step"),
            start_time=_T0 + timedelta(seconds=10 * n),
            end_time=_T0 + timedelta(seconds=10 * n + 5),
        )
        for n in range(2_500)
    ]
    tools = [
        SpanRecord(
            span_id=f"call-{n}",
            kind="tool",
            origin="gateway",
            name="x",
            outcome="ok",
            end_time=_T0 + timedelta(seconds=10 * n + 7),
        )
        for n in range(2_500)
    ]

    filled, approximate = _fill_client_tool_starts(tools[::-1], steps[::-1])

    assert len(approximate) == 2_500
    assert all(span.parent_span_id == f"req-{span.span_id.removeprefix('call-')}" for span in filled)


def test_a_tool_a_guardrail_blocked_is_not_an_error() -> None:
    spans = [_span("turn-1", "step", opens_turn=True), replace(_span("tool-1", "tool", "turn-1"), outcome="blocked")]

    [turn] = _turns(spans, now=_T0 + timedelta(hours=1), last_activity=_T0)

    assert (turn.state, turn.errors, turn.tool_calls) == ("completed", 0, 1)


def test_a_refusal_stored_apart_from_its_tool_still_blocks_it() -> None:
    """An agent may export a tool's children in another batch than the tool."""
    tool = _span("tool-1", "tool")
    decision = _span(
        "ask-1",
        "other",
        "tool-1",
        attributes={"otari.tool.decision": "reject", "otari.tool.decision_source": "hook"},
    )

    [settled, _] = _settle_tools([tool, decision])

    assert settled.outcome == "blocked"
    assert settled.attributes["otari.tool.decision_source"] == "hook"


def test_a_failed_run_stored_apart_from_its_tool_still_marks_it_as_returning_an_error() -> None:
    tool = _span("tool-1", "tool")
    run = _span("run-1", "other", "tool-1", attributes={"otari.tool.succeeded": False})

    [settled, _] = _settle_tools([tool, run])

    assert (settled.outcome, settled.attributes["otari.tool.succeeded"]) == ("ok", False)


def _gate(span_id: str, at: datetime, **fields: object) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        kind="guardrail",
        origin="otlp",
        name="gate:no-changelog",
        outcome="error",
        start_time=at,
        end_time=at,
        **fields,  # type: ignore[arg-type]
    )


def test_a_reported_gate_hangs_under_the_tool_call_it_checked() -> None:
    step = _span("turn-1", "step", opens_turn=True)
    tool = _span("tool-1", "tool", "turn-1", tool_call_id="toolu_01Y5eZnM7cScX5y8GyV7fT7q")
    gate = _gate("gate-1", _T0, tool_call_id="toolu_01Y5eZnM7cScX5y8GyV7fT7q")

    attached = _attach_gates([step, tool, gate], [step])

    assert attached[2].parent_span_id == "tool-1"


def test_a_gate_with_no_tool_call_hangs_under_the_step_it_ran_in() -> None:
    first = _span("turn-1", "step", opens_turn=True)
    second = replace(_span("turn-2", "step", opens_turn=True), start_time=_T0 + timedelta(minutes=1))
    stop_gate = _gate("gate-1", _T0 + timedelta(seconds=30))

    [*_, attached] = _attach_gates([first, second, stop_gate], [first, second])

    assert attached.parent_span_id == "turn-1"


def test_a_gate_the_agent_already_parented_is_left_where_it_is() -> None:
    step = _span("turn-1", "step", opens_turn=True)
    gate = _gate("gate-1", _T0, parent_span_id="elsewhere")

    [_, attached] = _attach_gates([step, gate], [step])

    assert attached.parent_span_id == "elsewhere"


def test_a_guardrail_doing_its_job_is_not_an_error() -> None:
    spans = [
        _span("turn-1", "step", opens_turn=True),
        replace(_span("tool-1", "tool", "turn-1"), outcome="blocked"),
        replace(_gate("gate-1", _T0), outcome="blocked", parent_span_id="turn-1"),
    ]

    [turn] = _turns(spans, now=_T0 + timedelta(hours=1), last_activity=_T0)

    assert (turn.state, turn.errors, turn.tool_calls) == ("completed", 0, 1)


def _checked(span_id: str, gate_id: str, outcome: str, *, blocked: bool = False, seconds: int = 0) -> SpanRecord:
    return _gate(
        span_id,
        _T0 + timedelta(seconds=seconds),
        attributes={"otari.gate.id": gate_id, "otari.gate.outcome": outcome, "otari.gate.blocked": blocked},
    )


def test_a_sessions_gates_are_counted_by_what_they_decided_the_ones_that_acted_first() -> None:
    spans = [
        _checked("a1", "python-needs-check", "pass", seconds=1),
        _checked("b1", "no-changelog", "pass", seconds=2),
        _checked("b2", "no-changelog", "fail", blocked=True, seconds=3),
        _checked("b3", "no-changelog", "fail", blocked=True, seconds=4),
        _checked("c1", "docs-updated", "fail", seconds=5),
        _checked("d1", "judge-review", "error", seconds=6),
        _span("turn-1", "step", opens_turn=True),
    ]

    gates = _gate_summaries(spans)

    assert [(g.gate_id, g.passed, g.blocked, g.flagged, g.undecided, g.span_id) for g in gates] == [
        ("no-changelog", 1, 2, 0, 0, "b2"),
        ("docs-updated", 0, 0, 1, 0, "c1"),
        ("judge-review", 0, 0, 0, 1, "d1"),
        ("python-needs-check", 1, 0, 0, 0, "a1"),
    ]


def test_a_required_gate_that_could_not_decide_is_counted_as_holding_the_agent() -> None:
    spans = [
        _checked("j1", "judge-review", "error", blocked=True, seconds=1),
        _checked("j2", "advisory-review", "error", seconds=2),
    ]

    gates = {gate.gate_id: gate for gate in _gate_summaries(spans)}

    assert (gates["judge-review"].refused, gates["judge-review"].undecided) == (1, 0)
    assert (gates["advisory-review"].refused, gates["advisory-review"].undecided) == (0, 1)


def test_a_gate_hangs_under_the_call_it_checked_when_a_client_reuses_call_ids() -> None:
    steps = [
        _span("turn-1", "step", opens_turn=True),
        replace(_span("turn-2", "step", opens_turn=True), start_time=_T0 + timedelta(seconds=50)),
    ]
    first = _span("tool-1", "tool", "turn-1", tool_call_id="call_1")
    second = replace(_span("tool-2", "tool", "turn-2", tool_call_id="call_1"), start_time=_T0 + timedelta(seconds=51))
    early = _gate("gate-1", _T0 + timedelta(seconds=1), tool_call_id="call_1")
    late = _gate("gate-2", _T0 + timedelta(seconds=52), tool_call_id="call_1")

    attached = {
        span.span_id: span.parent_span_id for span in _attach_gates([*steps, first, second, early, late], steps)
    }

    assert (attached["gate-1"], attached["gate-2"]) == ("tool-1", "tool-2")


def test_a_judge_gates_cost_is_read_on_the_gate_and_not_counted_as_a_turns_request() -> None:
    usage = _span("usage-1", "llm", operation="usage", request_id="00000000000000b2", cost_snapshot=Decimal("0.02"))
    spans = _price_requests(
        [
            _span("turn", "step", opens_turn=True),
            _span("gate-1", "guardrail", "turn", request_id="00000000000000b2"),
            usage,
        ]
    )

    assert [(span.span_id, span.cost_snapshot) for span in spans] == [("turn", None), ("gate-1", Decimal("0.02"))]
    [turn] = _turns(spans, now=_T0 + timedelta(hours=1), last_activity=_T0)
    assert turn.cost == Decimal(0)


def test_a_judges_further_model_is_added_to_its_gates_cost() -> None:
    main = _span("usage-1", "llm", operation="usage", request_id="00000000000000b2", cost_snapshot=Decimal("0.02"))
    side = replace(
        main, span_id="usage-2", request_id="00000000000000b2:00000000000000c3", cost_snapshot=Decimal("0.005")
    )
    spans = _price_requests([_span("gate-1", "guardrail", request_id="00000000000000b2"), main, side])

    assert [(span.span_id, span.cost_snapshot) for span in spans] == [("gate-1", Decimal("0.025"))]


def test_a_requests_cost_the_ledger_priced_later_is_read_on_the_request_and_counted_in_its_turn() -> None:
    usage = _span("usage-1", "llm", operation="usage", request_id="req_1", cost_snapshot=Decimal("0.25"))
    spans = _price_requests(
        [
            _span("turn", "step", opens_turn=True),
            _span("llm-1", "llm", "turn", request_id="req_1", input_tokens=10, output_tokens=2),
            usage,
            # A cost whose request was never exported is the session's, not a span to read.
            replace(usage, span_id="usage-2", request_id="req_lost"),
        ]
    )

    assert [(span.span_id, span.cost_snapshot) for span in spans] == [("turn", None), ("llm-1", Decimal("0.25"))]
    [turn] = _turns(spans, now=_T0 + timedelta(hours=1), last_activity=_T0)
    assert turn.cost == Decimal("0.25")
