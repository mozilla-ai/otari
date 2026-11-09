"""Spans an instrumented agent exports become trace documents, projected, never passed through."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue

from gateway.api.routes.otlp import _attributes
from gateway.core.config import GatewayConfig
from gateway.services.external_usage_service import (
    CLAUDE_SOURCE,
    ExternalUsageEvent,
    PricedEvent,
    ReadUsage,
    one_booking_per_response,
)
from gateway.services.traces import (
    MAX_SPANS_PER_EXPORT,
    OtlpSpan,
    SessionRef,
    TraceWriter,
    project_otlp_spans,
    record_otlp_export,
)
from gateway.services.traces._collector import call_id_digest
from gateway.services.traces._otlp import (
    MAX_TRACES_PER_EXPORT,
    PricedRequest,
    otlp_sessions,
    priced_requests,
    record_priced_requests,
)
from gateway.services.traces._reading import _gate_summaries
from gateway.services.traces._service import _growth
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


def test_a_conversation_named_on_the_root_alone_holds_its_whole_otel_trace() -> None:
    writes = _project(
        _span("root", attrs={"gen_ai.conversation.id": "conv-1"}),
        _span("chat", parent="root", attrs={"gen_ai.operation.name": "chat"}),
        _span("tool", parent="chat", attrs={"gen_ai.operation.name": "execute_tool"}),
    )

    [write] = writes
    assert len(write.spans) == 3


def test_a_conversation_named_below_the_root_holds_the_spans_around_it() -> None:
    writes = _project(
        _span("root"),
        _span("agent", parent="root", attrs={"gen_ai.conversation.id": "conv-1"}),
        _span("sibling", parent="root"),
    )

    [write] = writes
    assert len(write.spans) == 3


def test_a_session_label_an_agent_names_for_its_user_holds_the_spans_under_it() -> None:
    sessions = otlp_sessions(
        [
            _span("root", attrs={"otari.user_session_label": "project:otari", "otari.session_label": "other"}),
            _span("chat", parent="root", attrs={"gen_ai.operation.name": "chat"}),
        ]
    )

    assert set(sessions.values()) == {SessionRef(source="otlp", key="project:otari")}


def test_a_long_chain_of_spans_naming_nothing_is_in_the_session_its_root_names() -> None:
    depth = 20_000
    chain = [_span("s0", attrs={"gen_ai.conversation.id": "conv-1"})]
    chain += [_span(f"s{i}", parent=f"s{i - 1}") for i in range(1, depth)]

    sessions = otlp_sessions(reversed(chain))

    assert set(sessions.values()) == {SessionRef(source="otlp", key="conv-1")}
    assert len(sessions) == depth


def test_two_conversations_in_one_otel_trace_each_keep_the_spans_under_them() -> None:
    writes = _project(
        _span("a", attrs={"gen_ai.conversation.id": "conv-a"}),
        _span("a1", parent="a"),
        _span("b", attrs={"gen_ai.conversation.id": "conv-b"}),
        _span("b1", parent="b"),
    )

    assert sorted(sorted(span.otel_span_id or "" for span in write.spans) for write in writes) == [
        ["a", "a1"],
        ["b", "b1"],
    ]


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
    ("provider", "input_tokens", "cache_read_key", "cache_write_key", "scope"),
    [
        pytest.param(
            "anthropic",
            48_210,
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cache_write.input_tokens",
            "",
            id="inside-as-the-conventions-count-them",
        ),
        pytest.param(
            "openai",
            48_210,
            "gen_ai.usage.cached_tokens",
            "gen_ai.usage.cache_write_tokens",
            "",
            id="inside-under-older-names",
        ),
        pytest.param(
            "anthropic",
            300,
            "gen_ai.usage.cache_read_tokens",
            "gen_ai.usage.cache_write_tokens",
            "",
            id="on-top-when-they-outnumber-it",
        ),
        pytest.param(
            "anthropic",
            48_210,
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cache_creation.input_tokens",
            "opentelemetry.instrumentation.anthropic",
            id="inside-from-openllmetry",
        ),
        pytest.param(
            "anthropic",
            300,
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cache_creation.input_tokens",
            "openlit.instrumentation.anthropic",
            id="on-top-from-openlit",
        ),
    ],
)
def test_an_llm_span_counts_the_whole_prompt_whichever_way_the_exporter_reported_its_cache(
    provider: str, input_tokens: int, cache_read_key: str, cache_write_key: str, scope: str
) -> None:
    attrs = {
        "gen_ai.provider.name": provider,
        "gen_ai.usage.input_tokens": input_tokens,
        cache_read_key: 41_900,
        cache_write_key: 6_010,
        "gen_ai.usage.reasoning.output_tokens": 12,
    }
    [write] = _project(_span("a", attrs=attrs, scope=scope))

    [span] = write.spans
    assert (span.input_tokens, span.cache_read_tokens, span.cache_write_tokens) == (48_210, 41_900, 6_010)
    assert span.reasoning_tokens == 12


def test_a_known_emitters_cache_is_counted_its_way_even_where_fresh_input_outnumbers_it() -> None:
    attrs = {
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": 50_000,
        "gen_ai.usage.cache_read.input_tokens": 300,
        "gen_ai.usage.cache_creation.input_tokens": 150,
    }
    [write] = _project(_span("a", attrs=attrs, scope="openlit.instrumentation.anthropic"))

    assert write.spans[0].input_tokens == 50_450


@pytest.mark.parametrize(
    ("attrs", "request_id"),
    [({"gen_ai.response.id": "msg_1"}, "msg_1"), ({}, "a"), ({"gen_ai.response.id": "not an id"}, None)],
)
def test_an_llm_span_carries_the_id_the_ledger_priced_its_call_by(
    attrs: dict[str, str], request_id: str | None
) -> None:
    [write] = _project(_span("a", attrs={"gen_ai.operation.name": "chat", **attrs}))

    assert write.spans[0].request_id == request_id


def test_a_managed_agents_turn_carries_the_id_its_usage_was_booked_by() -> None:
    attrs = {"gen_ai.operation.name": "chat", "anthropic.session.id": "sesn_1", "gen_ai.response.id": "msg_1"}
    [write] = _project(_span("a", name="anthropic.model_request", attrs=attrs, scope="com.anthropic.claude-platform"))

    assert write.spans[0].request_id == f"{'t' * 32}:a"


def test_a_cache_write_reported_under_two_names_is_counted_as_the_larger() -> None:
    attrs = {
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.cache_write.input_tokens": 0,
        "gen_ai.usage.cache_creation.input_tokens": 160,
    }
    [write] = _project(_span("a", attrs=attrs, scope="openlit.instrumentation.anthropic"))

    [span] = write.spans
    assert (span.input_tokens, span.cache_write_tokens) == (260, 160)


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


def test_a_priced_request_read_off_its_exported_span_reaches_that_session_from_any_source() -> None:
    writer = _Writer()

    record_priced_requests(
        cast(TraceWriter, writer),
        [PricedRequest("msg_1", "conv-1", datetime.now(UTC), Decimal("0.01"), "otel", traced=True, harness=None)],
        budget_exempt=True,
        workspace_id=_WORKSPACE,
        user_id=None,
        api_key_id=None,
        retention=timedelta(days=30),
    )

    [write] = writer.writes
    assert write.harness is None
    assert write.trace_id == SessionRef(source="otlp", key="conv-1").trace_id(_WORKSPACE, None, None)
    assert [(span.request_id, span.cost_snapshot) for span in write.spans] == [("msg_1", Decimal("0.01"))]


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


def test_an_exports_spans_are_taken_in_when_it_was_admitted_not_when_it_finished() -> None:
    """An upload that outlasts an erasure of its user is judged from when it was let in."""
    admitted = _NOW - timedelta(minutes=11)

    [write] = project_otlp_spans(
        [_claude("i1", "interaction", user_prompt="Fix the flaky test")],
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=_RETENTION,
        now=_NOW,
        keep_content=True,
        content_generation=0,
        captured_at=admitted,
    ).writes

    assert write.captured_at == admitted


def test_a_call_two_instrumentations_both_exported_counts_its_tokens_once_in_the_session() -> None:
    call = {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 500, "gen_ai.usage.output_tokens": 20}
    [write] = _project(
        _span("sdk", attrs={**call, "gen_ai.response.id": "msg_1"}),
        _span("openllmetry", attrs={**call, "gen_ai.response.id": "msg_1"}),
        _span("next", attrs={**call, "gen_ai.response.id": "msg_2"}),
    )

    growth = _growth((_WORKSPACE, write.trace_id), list(write.spans), _NOW)

    # Both spans of the call are kept.
    assert len(write.spans) == 3
    assert (growth.input_tokens, growth.output_tokens) == (1000, 40)


def test_a_call_two_instrumentations_exported_counts_the_tokens_of_the_one_reporting_them() -> None:
    call = {"gen_ai.operation.name": "chat", "gen_ai.response.id": "msg_1"}
    [write] = _project(
        _span("wrapper", attrs=call),
        _span("openllmetry", attrs={**call, "gen_ai.usage.input_tokens": 500, "gen_ai.usage.output_tokens": 20}),
    )

    growth = _growth((_WORKSPACE, write.trace_id), list(write.spans), _NOW)

    assert (growth.input_tokens, growth.output_tokens) == (500, 20)


def _usage(
    event_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> ExternalUsageEvent:
    return ExternalUsageEvent(
        source_event_id=event_id,
        timestamp=_T0,
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def test_a_response_the_claude_sdk_booked_is_priced_on_its_span_even_where_another_span_came_first() -> None:
    pairs, event_spans = one_booking_per_response(
        [
            ReadUsage("otel", _usage("msg_1", input_tokens=1300), ("t", "openllmetry")),
            ReadUsage(CLAUDE_SOURCE, _usage("msg_1", input_tokens=1000), ("t", "sdk")),
        ]
    )

    assert [source for source, _ in pairs] == [CLAUDE_SOURCE]
    assert event_spans == {(CLAUDE_SOURCE, "msg_1"): ("t", "sdk")}


def test_a_response_two_libraries_booked_keeps_the_booking_reporting_most_tokens() -> None:
    pairs, event_spans = one_booking_per_response(
        [
            ReadUsage("agent-a", _usage("msg_1", output_tokens=20), ("t", "wrapper")),
            ReadUsage("agent-b", _usage("msg_1", input_tokens=500, output_tokens=20), ("t", "full")),
            ReadUsage("agent-a", _usage("msg_2", input_tokens=10), ("t", "other")),
        ]
    )

    assert [(source, event.source_event_id) for source, event in pairs] == [("agent-b", "msg_1"), ("agent-a", "msg_2")]
    assert event_spans == {("agent-b", "msg_1"): ("t", "full"), ("agent-a", "msg_2"): ("t", "other")}


def test_a_response_whose_cache_one_library_counts_on_top_keeps_the_booking_with_its_cache_writes() -> None:
    pairs, _ = one_booking_per_response(
        [
            ReadUsage("agent-a", _usage("msg_1", input_tokens=50, output_tokens=20), ("t", "a")),
            ReadUsage(
                "agent-b", _usage("msg_1", input_tokens=50, output_tokens=20, cache_write_tokens=160), ("t", "b")
            ),
        ]
    )

    assert [(source, event.cache_write_tokens) for source, event in pairs] == [("agent-b", 160)]


def test_two_calls_whose_ids_are_their_span_ids_are_never_merged() -> None:
    pairs, event_spans = one_booking_per_response(
        [
            ReadUsage("agent-a", _usage("00000000000000a1", input_tokens=5), ("t1", "00000000000000a1"), False),
            ReadUsage("agent-b", _usage("00000000000000a1", input_tokens=9), ("t2", "00000000000000a1"), False),
        ]
    )

    assert [source for source, _ in pairs] == ["agent-a", "agent-b"]
    # Each booking's cost reaches the span it was read from.
    assert event_spans == {
        ("agent-a", "00000000000000a1"): ("t1", "00000000000000a1"),
        ("agent-b", "00000000000000a1"): ("t2", "00000000000000a1"),
    }


class _CappedWriter:
    def __init__(self) -> None:
        self.writes: list[TraceWrite] = []

    def submit(self, write: TraceWrite, *, truncated: int = 0) -> None:
        self.writes.append(write)


def test_a_span_kept_under_the_cap_is_in_the_session_a_span_past_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.services.traces._otlp.MAX_SPANS_PER_EXPORT", 1)
    writer = _CappedWriter()

    record_otlp_export(
        cast(TraceWriter, writer),
        [_span("call", parent="root"), _span("root", attrs={"gen_ai.conversation.id": "conv-1"})],
        total=2,
        budget_exempt=True,
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="k",
        retention=_RETENTION,
    )

    [write] = writer.writes
    assert [span.otel_span_id for span in write.spans] == ["call"]
    assert write.trace_id == SessionRef(source="otlp", key="conv-1").trace_id(_WORKSPACE, "alice", "k")


def test_a_session_whose_calls_were_exported_is_never_the_one_the_session_cap_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gateway.services.traces._otlp.MAX_TRACES_PER_EXPORT", 1)
    writer = _Writer()
    at = datetime.now(UTC)

    record_priced_requests(
        cast(TraceWriter, writer),
        [
            PricedRequest("req_1", "labeled", at, Decimal("0.01"), "claude_code"),
            PricedRequest("msg_1", "conv-1", at, Decimal("0.02"), "otel", traced=True),
        ],
        budget_exempt=True,
        workspace_id=_WORKSPACE,
        user_id=None,
        api_key_id="k",
        retention=_RETENTION,
    )

    [write] = writer.writes
    assert write.trace_id == SessionRef(source="otlp", key="conv-1").trace_id(_WORKSPACE, None, "k")


def test_a_response_whose_input_counts_its_cache_is_ranked_by_its_input_alone() -> None:
    inclusive = _usage("msg_1", input_tokens=100, output_tokens=10, cache_read_tokens=80)
    inclusive = inclusive.model_copy(update={"cache_tokens_in_prompt": True})
    pairs, _ = one_booking_per_response(
        [
            ReadUsage("agent-a", inclusive, ("t", "a")),
            ReadUsage("agent-b", _usage("msg_1", input_tokens=150, output_tokens=10), ("t", "b")),
        ]
    )

    assert [source for source, _ in pairs] == ["agent-b"]


def test_a_claude_response_reported_twice_keeps_the_report_that_splits_its_1h_cache_writes() -> None:
    streamed = _usage("msg_1", input_tokens=1000, cache_write_tokens=100)
    served = streamed.model_copy(update={"cache_write_1h_tokens": 40})
    pairs, event_spans = one_booking_per_response(
        [
            ReadUsage(CLAUDE_SOURCE, streamed, ("t", "sdk")),
            ReadUsage(CLAUDE_SOURCE, served, ("t", "api")),
        ]
    )

    assert [event.cache_write_1h_tokens for _, event in pairs] == [40]
    # The cost still reaches the span the response was first read from.
    assert event_spans == {(CLAUDE_SOURCE, "msg_1"): ("t", "sdk")}


def test_a_priced_call_on_a_stored_span_comes_before_one_past_the_export_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gateway.services.traces._otlp.MAX_SPANS_PER_EXPORT", 1)
    at = datetime.now(UTC)

    def priced(event_id: str) -> PricedEvent:
        return PricedEvent(event_id, None, at, Decimal("0.01"), "otel", {})

    requests = priced_requests(
        [priced("msg_late"), priced("msg_kept")],
        [_span("kept", attrs={"gen_ai.conversation.id": "c"}), _span("late", attrs={"gen_ai.conversation.id": "c"})],
        {("otel", "msg_kept"): ("t" * 32, "kept"), ("otel", "msg_late"): ("t" * 32, "late")},
    )

    assert [request.request_id for request in requests] == ["msg_kept", "msg_late"]
