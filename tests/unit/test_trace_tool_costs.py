"""What a tool call cost, estimated from the priced model requests either side of it."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.services.traces._tool_costs import tool_costs
from gateway.types.trace_views import ToolCost
from gateway.types.traces import SpanRecord

_T0 = datetime(2026, 10, 19, 9, tzinfo=UTC)
# USD per million tokens: input, output, cache read, cache write.
_RATES = {"input": "3", "output": "15", "cache_read": "0.3", "cache_write": "3.75"}


def _at(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _request(
    span_id: str,
    seconds: float,
    *,
    read: int = 0,
    written: int = 0,
    output: int = 0,
    parent: str = "turn",
    model: str = "claude-opus-5-5",
    priced: bool = True,
) -> SpanRecord:
    attributes = {f"otari.llm.rate.{meter}": rate for meter, rate in _RATES.items()} if priced else {}
    return SpanRecord(
        span_id=span_id,
        kind="llm",
        origin="otlp",
        name="x",
        outcome="ok",
        parent_span_id=parent,
        start_time=_at(seconds),
        end_time=_at(seconds + 0.5),
        model=model,
        input_tokens=read + written,
        output_tokens=output,
        cache_read_tokens=read,
        cache_write_tokens=written,
        cost_snapshot=Decimal("0.10"),
        attributes=attributes,
    )


def _tool(span_id: str, seconds: float, *, parent: str = "turn", tool: str = "Read") -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        kind="tool",
        origin="otlp",
        name="x",
        outcome="ok",
        parent_span_id=parent,
        start_time=_at(seconds),
        end_time=_at(seconds + 1),
        tool_name=tool,
    )


def _turn() -> SpanRecord:
    return SpanRecord(span_id="turn", kind="step", origin="otlp", name="x", outcome="ok", start_time=_T0)


def test_a_call_costs_its_request_its_result_written_once_and_read_again_until_the_context_is_compacted() -> None:
    [cost] = tool_costs(
        [
            _turn(),
            _request("r1", 0, written=10_000, output=200),
            _tool("read", 1),
            # The context grew by the call's request (200) and its result (4,000).
            _request("r2", 3, read=10_000, written=4_200, output=100),
            _request("r3", 10, read=14_200, written=100),
            # Compacted: what came before is no longer read.
            _request("r4", 20, written=3_000),
        ]
    )

    assert cost == ToolCost(
        span_id="read",
        calling=Decimal("0.003"),
        writing=Decimal("0.015"),
        rereading=Decimal("0.0012"),
        result_tokens=4_000,
        later_requests=1,
        calls_sharing_request=1,
        calls_sharing_result=1,
    )
    assert cost.total == Decimal("0.0192")


def test_calls_one_request_made_together_share_its_output_and_their_results_growth() -> None:
    costs = tool_costs(
        [
            _turn(),
            _request("r1", 0, written=10_000, output=300),
            _tool("a", 1),
            _tool("b", 1.5),
            _request("r2", 4, read=10_000, written=8_300),
        ]
    )

    assert [(c.span_id, c.calling, c.result_tokens, c.calls_sharing_request) for c in costs] == [
        ("a", Decimal("0.00225"), 4_000, 2),
        ("b", Decimal("0.00225"), 4_000, 2),
    ]


def test_a_result_read_after_the_cache_expired_is_priced_as_written_again() -> None:
    [cost] = tool_costs(
        [
            _turn(),
            _request("r1", 0, written=10_000),
            _tool("read", 1),
            _request("r2", 3, read=10_000, written=1_000),
            _request("r3", 600, written=11_000),
        ]
    )

    assert cost.rereading == Decimal("0.00375")


def test_a_subagent_has_a_context_of_its_own_and_the_call_that_ran_it_carries_its_exact_cost() -> None:
    run = SpanRecord(
        span_id="agent-run",
        kind="other",
        origin="otlp",
        name="x",
        outcome="ok",
        operation="tool.execution",
        parent_span_id="agent",
        start_time=_at(1),
    )
    costs = tool_costs(
        [
            _turn(),
            _request("r1", 0, written=10_000, output=100),
            _tool("agent", 1, tool="Agent"),
            run,
            _request("s1", 2, written=2_000, output=50, parent="agent-run"),
            _tool("sub-read", 3, parent="agent-run"),
            _request("s2", 5, read=2_000, written=550, parent="agent-run"),
            _request("r2", 9, read=10_000, written=1_100),
        ]
    )

    by_id = {cost.span_id: cost for cost in costs}
    assert by_id["agent"].subagent == Decimal("0.20")
    # The subagent's call is read in its own context, not the main one.
    assert by_id["sub-read"].result_tokens == 500


def test_a_side_request_on_another_model_is_not_read_as_the_context_shrinking() -> None:
    [cost] = tool_costs(
        [
            _turn(),
            _request("r1", 0, written=10_000),
            _tool("read", 1),
            _request("r2", 3, read=10_000, written=1_000),
            _request("title", 4, written=300, model="claude-haiku-4-5-20251001"),
            _request("r3", 5, read=11_000, written=100),
        ]
    )

    assert cost.later_requests == 1


def test_a_session_whose_requests_were_not_priced_by_the_token_has_no_estimates() -> None:
    assert (
        tool_costs(
            [
                _turn(),
                _request("r1", 0, written=10_000, priced=False),
                _tool("read", 1),
                _request("r2", 3, read=10_000, written=1_000, priced=False),
            ]
        )
        == ()
    )


def test_a_request_starting_as_a_call_starts_asked_for_it_and_one_starting_as_it_ends_read_its_result() -> None:
    [cost] = tool_costs(
        [
            _turn(),
            _request("r0", -1, written=8_000, output=50),
            _request("r1", 0, written=10_000, output=200),
            # Starts with r1 and ends as r2 starts.
            _tool("read", 0),
            _request("r2", 1, read=10_000, written=4_200),
            _request("r3", 2, read=14_200),
        ]
    )

    assert cost.calling == Decimal(200) * Decimal(15) / Decimal(1_000_000)
    assert cost.result_tokens == 4_000
    assert cost.later_requests == 1
