"""How a read derives a turn's totals from its spans."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.ports.trace_storage_port import SpanRecord
from gateway.services.traces._reading import _turns

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
        _span("llm-1", "llm", "req-1", input_tokens=30, output_tokens=6, cost_snapshot=Decimal("0.03")),
        _span("round-1", "llm", "llm-1", attributes={"otari.tool_loop.round": 1}),
        _span("round-2", "llm", "llm-1", attributes={"otari.tool_loop.round": 2}),
        _span("round-3", "llm", "llm-1", attributes={"otari.tool_loop.round": 3}),
        _span("req-2", "step"),
        _span("llm-2", "llm", "req-2", input_tokens=5, output_tokens=1, cost_snapshot=Decimal("0.01")),
    ]

    [turn] = _turns(spans, now=_T0 + timedelta(hours=1), last_activity=_T0)

    assert turn.llm_calls == 4
    assert (turn.input_tokens, turn.output_tokens, turn.cost) == (35, 7, Decimal("0.04"))
