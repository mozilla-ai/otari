"""Spans an instrumented agent exports become trace documents, projected, never passed through."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from gateway.ports.trace_storage_port import TraceWrite, is_identifier
from gateway.services.traces import OtlpSpan, project_otlp_spans

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
        return write.spans[0].input_tokens

    assert tokens("120") == 120
    assert tokens("\u00b2") is None
    assert tokens("9" * 5_000) is None
    assert tokens(-5) is None
    assert tokens("-5") is None
    assert tokens(True) is None
    assert tokens(2**62) == _INT32_MAX
    assert tokens(str(2**40)) == _INT32_MAX


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
