"""Spans an instrumented agent exports become trace documents, projected, never passed through."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue

from gateway.api.routes.otlp import _attributes
from gateway.core.config import GatewayConfig
from gateway.services.traces import MAX_SPANS_PER_EXPORT, OtlpSpan, TraceWriter, project_otlp_spans
from gateway.services.traces._collector import call_id_digest
from gateway.services.traces._otlp import MAX_TRACES_PER_EXPORT, PricedRequest, record_priced_requests
from gateway.services.traces._reading import _gate_summaries
from gateway.types.traces import TraceWrite, is_identifier

_WORKSPACE = uuid.uuid4()
_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)
_NOW = _T0 + timedelta(hours=1)
_RETENTION = timedelta(days=30)
_INT32_MAX = 2_147_483_647


def _span(
    span_id: str,
    *,
    trace_id: str = "t" * 32,
    parent: str | None = None,
    name: str = "span",
    failed: bool = False,
    attrs: Any = None,
    start: datetime | None = _T0,
    end: datetime | None = _T0 + timedelta(seconds=2),
    scope: str = "",
) -> OtlpSpan:
    return OtlpSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_time=start,
        end_time=end,
        failed=failed,
        attributes=attrs or {},
        scope=scope,
    )


def _project(
    *spans: OtlpSpan, user_id: str | None = "alice", api_key_id: str = "k", max_traces: int = 1_000
) -> list[TraceWrite]:
    return list(
        project_otlp_spans(
            list(spans),
            workspace_id=_WORKSPACE,
            user_id=user_id,
            api_key_id=api_key_id,
            retention=_RETENTION,
            now=_NOW,
            max_traces=max_traces,
        ).writes
    )


def test_an_agent_run_keeps_its_tree_and_each_span_its_kind() -> None:
    [write] = _project(
        _span("a1", name="invoke_agent qa", attrs={"gen_ai.operation.name": "invoke_agent"}),
        _span(
            "c1",
            parent="a1",
            attrs={"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-5", "gen_ai.usage.input_tokens": 12},
        ),
        _span("x1", parent="a1", attrs={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "searchDocs"}),
    )

    ids = {span.otel_span_id: span.span_id for span in write.spans}
    kinds = {span.otel_span_id: (span.kind, span.parent_span_id) for span in write.spans}
    assert kinds == {
        "a1": ("agent", None),
        "c1": ("llm", ids["a1"]),
        "x1": ("tool", ids["a1"]),
    }
    assert all(span.span_id.startswith("otlp-") and is_identifier(span.span_id) for span in write.spans)
    llm = next(span for span in write.spans if span.kind == "llm")
    assert (llm.model, llm.input_tokens, llm.origin) == ("gpt-5", 12, "otlp")
    assert write.session_source == "otlp"


def test_spans_naming_one_conversation_share_a_trace_across_otel_traces() -> None:
    writes = _project(
        _span("a", trace_id="1" * 32, attrs={"gen_ai.conversation.id": "conv-7"}),
        _span("b", trace_id="2" * 32, attrs={"gen_ai.conversation.id": "conv-7"}),
        _span("c", trace_id="3" * 32),
    )

    assert sorted(len(write.spans) for write in writes) == [1, 2]


def test_a_claude_sessions_threads_share_one_trace() -> None:
    session = {"anthropic.session.id": "sesn_1"}
    writes = _project(
        _span("a", trace_id="1" * 32, attrs={**session, "gen_ai.conversation.id": "sthr_main"}),
        _span("b", trace_id="2" * 32, attrs={**session, "gen_ai.conversation.id": "sthr_sub"}),
    )

    assert [len(write.spans) for write in writes] == [2]


_CLAUDE_CODE = "com.anthropic.claude_code.tracing"
_TOOL_USE_ID = "toolu_01Y5eZnM7cScX5y8GyV7fT7q"


def _claude(span_id: str, span_type: str, *, parent: str | None = None, **attrs: Any) -> OtlpSpan:
    return _span(
        span_id,
        trace_id=attrs.pop("trace_id", "c" * 32),
        parent=parent,
        name=f"claude_code.{span_type}",
        attrs={"span.type": span_type, "session.id": "77192d4d-b969-44eb-bb05-e7ca84f6dfab", **attrs},
        scope=_CLAUDE_CODE,
    )


def test_a_claude_code_interaction_is_a_turn_with_its_model_calls_and_tools() -> None:
    [write] = _project(
        _claude("i1", "interaction"),
        _claude(
            "l1",
            "llm_request",
            parent="i1",
            model="claude-haiku-4-5-20251001",
            **{"gen_ai.system": "anthropic"},
            input_tokens="10",
            output_tokens="300",
            cache_read_tokens="18157",
            cache_creation_tokens="2000",
            stop_reason="tool_use",
            ttft_ms="917",
            success="True",
        ),
        _claude("t1", "tool", parent="i1", tool_name="Read", tool_use_id=_TOOL_USE_ID),
        _claude("d1", "tool.blocked_on_user", parent="t1", decision="accept", source="config"),
        _claude("e1", "tool.execution", parent="t1", success="True"),
    )

    by_id = {span.otel_span_id: span for span in write.spans}
    turn, llm, tool = by_id["i1"], by_id["l1"], by_id["t1"]
    assert (turn.kind, turn.opens_turn) == ("step", True)
    assert (llm.kind, llm.parent_span_id, llm.model, llm.provider) == (
        "llm",
        turn.span_id,
        "claude-haiku-4-5-20251001",
        "anthropic",
    )
    # Claude Code counts the cache outside its input tokens; a span counts it in.
    assert (llm.input_tokens, llm.output_tokens, llm.cache_read_tokens, llm.cache_write_tokens) == (
        20_167,
        300,
        18_157,
        2_000,
    )
    assert dict(llm.attributes) == {"gen_ai.response.finish_reasons": "tool_use", "otari.llm.ttft_ms": 917}
    assert (tool.kind, tool.tool_name, tool.tool_type, tool.tool_call_id, tool.outcome) == (
        "tool",
        "Read",
        "client",
        call_id_digest(_WORKSPACE, _TOOL_USE_ID),
        "ok",
    )
    assert dict(tool.attributes) == {"otari.tool.decision": "accept", "otari.tool.decision_source": "config"}
    assert write.harness == "claude-code"


def test_a_tool_a_hook_refused_is_blocked_not_failed() -> None:
    [write] = _project(
        _claude("i1", "interaction"),
        _claude("t1", "tool", parent="i1", tool_name="Edit", tool_use_id=_TOOL_USE_ID),
        _claude("d1", "tool.blocked_on_user", parent="t1", decision="reject", source="hook"),
    )

    by_id = {span.otel_span_id: span for span in write.spans}
    assert (by_id["t1"].outcome, by_id["t1"].error_class) == ("blocked", None)
    assert dict(by_id["t1"].attributes) == {"otari.tool.decision": "reject", "otari.tool.decision_source": "hook"}
    assert dict(by_id["d1"].attributes) == dict(by_id["t1"].attributes)
    assert by_id["d1"].outcome == "ok"


def test_a_claude_code_tool_that_returned_an_error_ran_and_is_not_a_session_error() -> None:
    [write] = _project(
        _claude("t1", "tool", parent="i1", tool_name="Bash", tool_use_id=_TOOL_USE_ID),
        _claude("e1", "tool.execution", parent="t1", success="False"),
    )

    tool = next(span for span in write.spans if span.otel_span_id == "t1")
    assert (tool.outcome, tool.error_class) == ("ok", None)
    assert tool.attributes["otari.tool.succeeded"] is False


def test_a_claude_code_sessions_turns_share_one_trace_across_otel_traces() -> None:
    writes = _project(
        _claude("i1", "interaction", trace_id="1" * 32),
        _claude("i2", "interaction", trace_id="2" * 32),
    )

    assert [len(write.spans) for write in writes] == [2]
    assert all(span.opens_turn for span in writes[0].spans)


def test_session_id_groups_only_claude_codes_own_spans() -> None:
    writes = _project(
        _span("a", trace_id="1" * 32, attrs={"session.id": "s-1"}),
        _span("b", trace_id="2" * 32, attrs={"session.id": "s-1"}),
    )

    assert [len(write.spans) for write in writes] == [1, 1]


def test_a_nested_claude_code_interaction_is_an_agent_not_a_new_turn() -> None:
    [write] = _project(
        _claude("i1", "interaction"),
        _claude("t1", "tool", parent="i1", tool_name="Task"),
        _claude("i2", "interaction", parent="t1"),
    )

    nested = next(span for span in write.spans if span.otel_span_id == "i2")
    assert (nested.kind, nested.opens_turn) == ("agent", False)


def test_a_call_id_is_kept_as_a_digest_never_as_sent() -> None:
    [write] = _project(_claude("t1", "tool", tool_name="Bash", tool_use_id="hf_AbCdEfGhIjKlMnOpQrStUvWx0123456789"))

    stored = write.spans[0].tool_call_id
    assert stored is not None and stored.startswith("tc-") and "AbCd" not in stored


def test_a_claude_code_tool_that_returned_an_error_is_no_error_at_all() -> None:
    [write] = _project(
        _claude("t1", "tool", tool_name="Bash", tool_use_id=_TOOL_USE_ID),
        _claude("e1", "tool.execution", parent="t1", success="False"),
    )

    outcomes = {span.otel_span_id: span.outcome for span in write.spans}
    assert outcomes == {"t1": "ok", "e1": "ok"}
    run = next(span for span in write.spans if span.otel_span_id == "e1")
    assert run.attributes["otari.tool.succeeded"] is False


def _gate(span_id: str, outcome: str, *, enforcement: str = "required", **attrs: Any) -> OtlpSpan:
    return _span(
        span_id,
        trace_id=attrs.pop("trace_id", "e" * 32),
        name="otari.gate",
        attrs={
            "session.id": "77192d4d-b969-44eb-bb05-e7ca84f6dfab",
            "otari.harness": "claude-code",
            "otari.gate.id": "no-changelog",
            "otari.gate.type": "path",
            "otari.gate.origin": "repo",
            "otari.gate.enforcement": enforcement,
            "otari.gate.outcome": outcome,
            "otari.gate.blocked": outcome == "fail" and enforcement == "required",
            "otari.gate.moment": "pre_tool_use.edit_target",
            "otari.policy.digest": "0123456789abcdef",
            "gen_ai.tool.call.id": _TOOL_USE_ID,
            **attrs,
        },
        scope="otari.hook",
    )


def test_a_gate_the_hook_reported_is_a_guardrail_span_named_for_the_gate() -> None:
    [write] = _project(_gate("g1", "fail"))

    [gate] = write.spans
    assert (gate.kind, gate.name, gate.outcome, gate.error_class, gate.tool_call_id) == (
        "guardrail",
        "gate:no-changelog",
        "blocked",
        None,
        call_id_digest(_WORKSPACE, _TOOL_USE_ID),
    )
    assert gate.attributes["otari.guardrail.profile"] == "no-changelog"
    assert gate.attributes["otari.guardrail.mode"] == "block"
    assert gate.attributes["otari.gate.blocked"] is True
    assert gate.attributes["otari.policy.digest"] == "0123456789abcdef"
    assert write.harness == "claude-code"


def test_a_gate_id_a_guardrail_accepts_survives_even_where_it_is_not_an_identifier() -> None:
    # `api-key-` reads as a credential to the identifier rule, but it is a legal gate id.
    [write] = _project(_gate("g1", "fail", **{"otari.gate.id": "api-key-in-diff"}))

    [gate] = write.spans
    assert gate.attributes["otari.gate.id"] == "api-key-in-diff"
    assert "otari.guardrail.profile" not in gate.attributes
    [summary] = _gate_summaries(list(write.spans))
    assert (summary.gate_id, summary.blocked) == ("api-key-in-diff", 1)


def test_a_judge_gate_keeps_the_model_its_calls_ran_on_and_the_tokens_they_used() -> None:
    [write] = _project(
        _gate(
            "g1",
            "pass",
            enforcement="advisory",
            **{
                "otari.gate.type": "judge",
                "gen_ai.provider.name": "anthropic",
                "gen_ai.request.model": "claude-haiku-4-5-20251001",
                "gen_ai.usage.input_tokens": "9",
                "gen_ai.usage.output_tokens": "111",
                "gen_ai.usage.cache_read_tokens": "0",
                "gen_ai.usage.cache_write_tokens": "6684",
            },
        )
    )

    [gate] = write.spans
    assert (gate.kind, gate.model, gate.provider) == ("guardrail", "claude-haiku-4-5-20251001", "anthropic")
    # The hook counts the cache outside its input tokens, as Claude does; a span counts it in.
    assert (gate.input_tokens, gate.output_tokens, gate.cache_read_tokens, gate.cache_write_tokens) == (
        6_693,
        111,
        0,
        6_684,
    )


def test_a_judges_usage_on_a_further_model_is_priced_not_read_as_a_gate() -> None:
    side = _span(
        "u1",
        trace_id="e" * 32,
        parent="g1",
        name="otari.gate.usage",
        attrs={"session.id": "77192d4d-b969-44eb-bb05-e7ca84f6dfab", "gen_ai.request.model": "claude-sonnet-4-5"},
        scope="otari.hook",
    )

    [write] = _project(_gate("g1", "pass"), side)

    assert [span.name for span in write.spans] == ["gate:no-changelog"]


def test_a_gate_that_called_no_model_has_no_usage() -> None:
    [write] = _project(_gate("g1", "pass"))

    [gate] = write.spans
    assert (gate.model, gate.input_tokens, gate.cache_read_tokens, gate.cache_write_tokens) == (None, None, None, None)


@pytest.mark.parametrize(
    ("outcome", "enforcement", "expected"),
    [
        ("pass", "required", ("ok", None, "block")),
        ("fail", "advisory", ("ok", None, "monitor")),
        ("not_run", "advisory", ("error", "guardrail_unavailable", "monitor")),
        ("unknown", "required", ("error", "guardrail_unavailable", "block")),
    ],
)
def test_only_a_gate_that_could_not_decide_is_an_error(
    outcome: str, enforcement: str, expected: tuple[Any, ...]
) -> None:
    [write] = _project(_gate("g1", outcome, enforcement=enforcement))

    [gate] = write.spans
    assert (gate.outcome, gate.error_class, gate.attributes["otari.guardrail.mode"]) == expected


def test_a_gate_lands_on_the_same_session_as_the_agents_own_spans() -> None:
    writes = _project(
        _claude("i1", "interaction"),
        _claude("t1", "tool", parent="i1", tool_name="Edit", tool_use_id=_TOOL_USE_ID),
        _gate("g1", "fail"),
    )

    assert [len(write.spans) for write in writes] == [3]


def test_text_in_a_span_name_or_attribute_never_reaches_the_document() -> None:
    [write] = _project(
        _span(
            "a",
            name="summarize: the merger memo",
            failed=True,
            attrs={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "read file",
                "error.type": "file not found",
            },
        )
    )

    [span] = write.spans
    assert (span.name, span.tool_name, span.outcome, span.error_class) == ("execute_tool", None, "error", "error")
    assert "merger" not in repr(span)


@pytest.mark.parametrize(
    ("reasons", "stored"),
    [(["stop"], "stop"), (("tool_calls",), "tool_calls"), (["stop", "length"], "stop,length"), ("stop", "stop")],
)
def test_finish_reasons_sent_as_an_array_are_kept(reasons: Any, stored: str) -> None:
    [write] = _project(
        _span("c1", attrs={"gen_ai.operation.name": "chat", "gen_ai.response.finish_reasons": reasons}),
    )

    [span] = write.spans
    assert span.attributes == {"gen_ai.response.finish_reasons": stored}


@pytest.mark.parametrize(
    "reasons",
    [
        ["stop", "the model ran out of room"],
        ["stop", 1],
        ["stop", None],
        [],
        ["x" * 100, "y" * 100],
        {"first": "stop"},
        ["stop,length"],
        "stop,length",
    ],
)
def test_finish_reasons_that_are_not_all_identifiers_are_dropped_and_the_span_kept(reasons: Any) -> None:
    [write] = _project(
        _span("c1", attrs={"gen_ai.operation.name": "chat", "gen_ai.response.finish_reasons": reasons}),
    )

    [span] = write.spans
    assert span.attributes == {}


@pytest.mark.parametrize(("sent", "stored"), [(1024, 1024), ("1024", 1024), (10**12, _INT32_MAX)])
def test_the_requested_max_tokens_is_kept(sent: Any, stored: int) -> None:
    [write] = _project(_span("c1", attrs={"gen_ai.operation.name": "chat", "gen_ai.request.max_tokens": sent}))

    [span] = write.spans
    assert span.attributes == {"gen_ai.request.max_tokens": stored}


@pytest.mark.parametrize("sent", [-1, 1.5, True, "lots"])
def test_a_max_tokens_that_is_not_a_count_is_dropped_and_the_span_kept(sent: Any) -> None:
    [write] = _project(_span("c1", attrs={"gen_ai.operation.name": "chat", "gen_ai.request.max_tokens": sent}))

    [span] = write.spans
    assert span.attributes == {}


def test_an_array_attribute_is_read_only_where_a_trace_asks_for_it() -> None:
    reasons = AnyValue(array_value=ArrayValue(values=[AnyValue(string_value="stop")]))
    received = [KeyValue(key="gen_ai.response.finish_reasons", value=reasons)]

    assert _attributes(received) == {"gen_ai.response.finish_reasons": None}
    assert _attributes(received, arrays=True) == {"gen_ai.response.finish_reasons": ("stop",)}


def test_a_span_with_no_ids_is_skipped() -> None:
    assert _project(_span("", trace_id="")) == []


def test_equal_span_ids_from_two_otel_traces_of_one_conversation_stay_apart() -> None:
    conversation = {"gen_ai.conversation.id": "conv-7"}
    [write] = _project(
        _span("root", trace_id="1" * 32, attrs=conversation),
        _span("leaf", trace_id="1" * 32, parent="root", attrs=conversation),
        _span("root", trace_id="2" * 32, attrs=conversation),
        _span("leaf", trace_id="2" * 32, parent="root", attrs=conversation),
    )

    assert len({span.span_id for span in write.spans}) == 4
    by_id = {span.span_id: span for span in write.spans}
    for span in write.spans:
        if span.parent_span_id is not None:
            assert by_id[span.parent_span_id].otel_trace_id == span.otel_trace_id


def test_a_span_id_is_the_same_on_every_export() -> None:
    [first] = _project(_span("a"))
    [again] = _project(_span("a"))

    assert first.spans[0].span_id == again.spans[0].span_id


def test_keys_with_no_user_do_not_share_a_conversation() -> None:
    conversation = {"gen_ai.conversation.id": "conv-7"}
    [one] = _project(_span("a", attrs=conversation), user_id=None, api_key_id="k1")
    [other] = _project(_span("a", attrs=conversation), user_id=None, api_key_id="k2")
    [first_key] = _project(_span("a", attrs=conversation), user_id="alice", api_key_id="k1")
    [second_key] = _project(_span("a", attrs=conversation), user_id="alice", api_key_id="k2")

    assert one.trace_id != other.trace_id
    assert first_key.trace_id == second_key.trace_id


def test_token_counts_are_clamped_and_malformed_ones_dropped() -> None:
    def tokens(value: Any) -> int | None:
        [write] = _project(_span("a", attrs={"gen_ai.usage.input_tokens": value}))
        count: int | None = write.spans[0].input_tokens
        return count

    assert tokens("120") == 120
    assert tokens("\u00b2") is None
    assert tokens("9" * 5_000) is None
    assert tokens(-5) is None
    assert tokens("-5") is None
    assert tokens(True) is None
    assert tokens(2**62) == _INT32_MAX
    assert tokens(str(2**40)) == _INT32_MAX


@pytest.mark.parametrize(
    ("provider", "input_tokens", "cache_read_key", "cache_write_key"),
    [
        pytest.param(
            "anthropic",
            48_210,
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cache_creation.input_tokens",
            id="inside-as-the-conventions-count-them",
        ),
        pytest.param(
            "openai",
            48_210,
            "gen_ai.usage.cached_tokens",
            "gen_ai.usage.cache_write_tokens",
            id="inside-under-older-names",
        ),
        pytest.param(
            "anthropic",
            300,
            "gen_ai.usage.cache_read_tokens",
            "gen_ai.usage.cache_write_tokens",
            id="on-top-when-they-outnumber-it",
        ),
    ],
)
def test_an_llm_span_counts_the_whole_prompt_whichever_way_the_exporter_reported_its_cache(
    provider: str, input_tokens: int, cache_read_key: str, cache_write_key: str
) -> None:
    attrs = {
        "gen_ai.provider.name": provider,
        "gen_ai.usage.input_tokens": input_tokens,
        cache_read_key: 41_900,
        cache_write_key: 6_010,
        "gen_ai.usage.reasoning.output_tokens": 12,
    }
    [write] = _project(_span("a", attrs=attrs))

    [span] = write.spans
    assert (span.input_tokens, span.cache_read_tokens, span.cache_write_tokens) == (48_210, 41_900, 6_010)
    assert span.reasoning_tokens == 12


def test_a_reported_zero_cache_count_is_kept_not_stored_as_unknown() -> None:
    attrs = {
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.cache_creation.input_tokens": 0,
    }
    [write] = _project(_span("a", attrs=attrs))

    [span] = write.spans
    assert (span.input_tokens, span.cache_read_tokens, span.cache_write_tokens) == (100, 0, 0)


def test_a_span_that_reports_no_cache_count_leaves_it_unknown() -> None:
    [write] = _project(_span("a", attrs={"gen_ai.usage.input_tokens": 100}))

    [span] = write.spans
    assert (span.cache_read_tokens, span.cache_write_tokens) == (None, None)


def test_cached_tokens_never_lift_the_whole_prompt_past_what_the_store_holds() -> None:
    attrs = {
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.cache_read_tokens": _INT32_MAX,
    }
    [write] = _project(_span("a", attrs=attrs))

    assert write.spans[0].input_tokens == _INT32_MAX


def test_span_times_are_clamped_to_the_window_a_trace_can_live_in() -> None:
    [write] = _project(
        _span("old", start=datetime(1970, 1, 1, tzinfo=UTC), end=datetime(1970, 1, 1, 0, 1, tzinfo=UTC)),
        _span("future", start=_NOW, end=datetime(2999, 1, 1, tzinfo=UTC)),
        _span("backwards", start=_T0, end=_T0 - timedelta(hours=1)),
    )

    spans = {span.otel_span_id: span for span in write.spans}
    assert spans["old"].start_time == spans["old"].end_time == _NOW - _RETENTION
    assert spans["future"].end_time == _NOW + timedelta(minutes=5)
    assert spans["future"].duration_ms == 5 * 60 * 1000
    assert spans["backwards"].end_time == _T0
    assert spans["backwards"].duration_ms == 0


def test_a_duration_never_exceeds_what_the_store_holds() -> None:
    [write] = project_otlp_spans(
        [_span("a", start=_NOW - timedelta(days=365), end=_NOW)],
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=timedelta(days=3650),
        now=_NOW,
    ).writes

    assert write.spans[0].duration_ms == _INT32_MAX


def test_spans_past_the_sessions_an_export_may_write_are_dropped_and_counted() -> None:
    projection = project_otlp_spans(
        [_span("a", trace_id="1" * 32), _span("b", trace_id="2" * 32), _span("c", trace_id="3" * 32)],
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=_RETENTION,
        now=_NOW,
        max_traces=2,
    )

    assert len(projection.writes) == 2
    assert projection.dropped == 1


class _ExplodingAttributes(dict[str, Any]):
    def get(self, key: str, default: Any = None) -> Any:
        raise RuntimeError("the merger memo")


def test_a_span_that_cannot_be_projected_is_dropped_never_raised() -> None:
    projection = project_otlp_spans(
        [_span("good"), _span("bad", attrs=_ExplodingAttributes(x=1)), _span("nameless", name=None)],  # type: ignore[arg-type]
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=_RETENTION,
        now=_NOW,
    )

    [write] = projection.writes
    assert [span.otel_span_id for span in write.spans] == ["good"]
    assert projection.dropped == 2


def test_one_export_can_never_fill_the_trace_queue() -> None:
    """The writer's queue is shared by every request, so one export keeps only a share of it."""
    assert MAX_SPANS_PER_EXPORT < GatewayConfig().trace_queue_max_spans


def test_the_session_cap_binds_below_the_span_cap() -> None:
    """A cap on sessions at or above the span cap could never take effect."""
    assert MAX_TRACES_PER_EXPORT < MAX_SPANS_PER_EXPORT


def test_content_is_kept_only_where_the_deployment_keeps_it() -> None:
    spans = (
        _claude("i1", "interaction", user_prompt="Fix the flaky test"),
        _claude("t1", "tool", parent="i1", tool_name="Bash", tool_use_id=_TOOL_USE_ID, full_command="pytest -x"),
        _claude("t2", "tool", parent="i1", tool_name="Edit", file_path="src/app.py"),
        _gate("g1", "fail", **{"otari.gate.evidence": "CHANGELOG.md", "otari.gate.detail": "(CHANGELOG.md)"}),
    )

    [kept] = project_otlp_spans(
        list(spans),
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=_RETENTION,
        now=_NOW,
        keep_content=True,
    ).writes
    [dropped] = _project(*spans)

    by_id = {span.otel_span_id: dict(span.content) for span in kept.spans}
    assert by_id == {
        "i1": {"prompt": "Fix the flaky test"},
        "t1": {"command": "pytest -x"},
        "t2": {"file_path": "src/app.py"},
        "g1": {"gate.evidence": "CHANGELOG.md", "gate.detail": "(CHANGELOG.md)"},
    }
    assert all(not span.content for span in dropped.spans)


def test_a_prompt_claude_code_redacted_is_not_content() -> None:
    [write] = project_otlp_spans(
        [_claude("i1", "interaction", user_prompt="<REDACTED>")],
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=_RETENTION,
        now=_NOW,
        keep_content=True,
    ).writes

    assert not write.spans[0].content


def test_a_claude_code_model_request_keeps_its_request_id_though_it_looks_like_a_credential() -> None:
    [write] = _project(_claude("l1", "llm_request", request_id="req_011CgBfeQi6uhNCE3aH5jmr3"))

    assert write.spans[0].request_id == "req_011CgBfeQi6uhNCE3aH5jmr3"


class _Writer:
    def __init__(self) -> None:
        self.writes: list[TraceWrite] = []

    def submit(self, write: TraceWrite) -> None:
        self.writes.append(write)


@pytest.mark.parametrize(("budget_exempt", "written"), [(True, 1), (False, 0)])
def test_a_priced_request_reaches_its_session_only_for_a_key_that_may_import(budget_exempt: bool, written: int) -> None:
    writer = _Writer()

    record_priced_requests(
        cast(TraceWriter, writer),
        [PricedRequest("req_1", "session-1", datetime.now(UTC), Decimal("0.01"), "claude_code")],
        budget_exempt=budget_exempt,
        workspace_id=_WORKSPACE,
        user_id=None,
        api_key_id=None,
        retention=timedelta(days=30),
    )

    assert len(writer.writes) == written


@pytest.mark.parametrize(("source", "written"), [("claude_code", 1), ("otari-hook", 1), ("codex", 0), ("otel", 0)])
def test_a_priced_request_reaches_a_session_only_from_a_source_the_trace_store_has(source: str, written: int) -> None:
    writer = _Writer()

    record_priced_requests(
        cast(TraceWriter, writer),
        [PricedRequest("req_1", "session-1", datetime.now(UTC), Decimal("0.01"), source)],
        budget_exempt=True,
        workspace_id=_WORKSPACE,
        user_id=None,
        api_key_id=None,
        retention=timedelta(days=30),
    )

    assert len(writer.writes) == written


def test_a_priced_request_whose_id_a_span_cannot_carry_is_skipped_and_the_rest_recorded() -> None:
    writer = _Writer()

    record_priced_requests(
        cast(TraceWriter, writer),
        [
            PricedRequest("batch/req_1", "session-1", datetime.now(UTC), Decimal("0.01"), "claude_code"),
            PricedRequest("req_2", "session-2", datetime.now(UTC), Decimal("0.02"), "claude_code"),
        ],
        budget_exempt=True,
        workspace_id=_WORKSPACE,
        user_id=None,
        api_key_id=None,
        retention=timedelta(days=30),
    )

    [write] = writer.writes
    assert [span.request_id for span in write.spans] == ["req_2"]
