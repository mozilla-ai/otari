"""How a read derives a turn's totals from its spans."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.services.traces._reading import _fill_client_tool_starts, _settle_tools, _turns
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
