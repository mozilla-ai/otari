"""Recording a request's trace: the collector, the bounded writer, the usage-row tap and the middleware.

The writer's limits are what make tracing best-effort rather than a risk to the
request path, so each one is asserted on what it drops and counts, not only on
what it writes.
"""

import asyncio
import uuid
from collections.abc import MutableMapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError
from starlette.types import Receive, Send

from gateway.api.trace_capture import TRACE_SCOPE_KEY, TraceCaptureMiddleware
from gateway.models.usage import UsageLog
from gateway.services.log_writer import TracingLogWriter, _llm_call
from gateway.services.traces import RequestTrace, RequestTraces, TraceWriter
from gateway.services.traces._writer import SPANS_DROPPED
from gateway.services.web_fetch_service import WebFetchResult, WebFetchService
from gateway.services.web_retrieval_backend import WEB_FETCH_TOOL_NAME, WebRetrievalBackend
from gateway.services.web_retrieval_policy import DomainPolicy, canonicalize_web_url
from gateway.types.traces import SpanRecord, TraceWrite, WriteResult

_WORKSPACE = uuid.uuid4()
_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)


def _trace(request_id: str = "req-1", *, max_spans: int = 256) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="key-1",
        endpoint="/v1/chat/completions",
        started_at=_T0,
        max_spans=max_spans,
    )


def _usage_row(**overrides: Any) -> UsageLog:
    fields: dict[str, Any] = {
        "id": "row-1",
        "timestamp": _T0 + timedelta(seconds=3),
        "latency_ms": 2900,
        "model": "gpt-5",
        "provider": "openai",
        "status": "success",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost": Decimal("0.0049"),
        "request_group_id": "req-1",
        "policy_name": None,
        "selection_reason": None,
        "attempt_position": None,
        "status_code": None,
    }
    return UsageLog(**(fields | overrides))


def _write(spans: int, request_id: str = "req-1", user_id: str = "alice") -> TraceWrite:
    step = SpanRecord(span_id=request_id, kind="step", origin="gateway", name="request", outcome="ok")
    extra = tuple(
        SpanRecord(span_id=f"{request_id}-{i}", kind="tool", origin="gateway", name="tool", outcome="ok")
        for i in range(spans - 1)
    )
    return TraceWrite(
        workspace_id=_WORKSPACE,
        trace_id=request_id,
        user_id=user_id,
        api_key_id=None,
        session_source="none",
        spans=(step, *extra),
    )


def _dropped(reason: str) -> float:
    value: float = SPANS_DROPPED.labels(reason=reason)._value.get()
    return value


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


def test_a_usage_row_becomes_an_llm_span_with_what_billing_recorded() -> None:
    trace = _trace()

    trace.record_llm_call(_llm_call(_usage_row()))

    [span] = trace.spans
    assert (span.kind, span.outcome, span.parent_span_id) == ("llm", "ok", "req-1")
    assert (span.model, span.provider) == ("gpt-5", "openai")
    assert (span.input_tokens, span.output_tokens, span.cost_snapshot) == (100, 20, Decimal("0.0049"))
    assert span.start_time == _T0 + timedelta(milliseconds=100)


def test_an_absorbed_attempt_is_a_recovered_error() -> None:
    """A failover the next candidate made up for is visible, and does not fail the request."""
    trace = _trace()

    trace.record_llm_call(
        _llm_call(_usage_row(status="absorbed", status_code=429, attempt_position=1, policy_name="cheap-first"))
    )

    [span] = trace.spans
    assert (span.outcome, span.recovered, span.error_class) == ("error", True, "status_429")
    assert span.attributes == {
        "otari.routing.policy": "cheap-first",
        "otari.routing.attempt_position": 1,
        "http.response.status_code": 429,
    }


def test_client_text_in_a_routing_field_is_dropped_not_stored() -> None:
    trace = _trace()

    trace.record_llm_call(_llm_call(_usage_row(policy_name="the policy Alice wrote for the Q3 launch")))

    assert "otari.routing.policy" not in trace.spans[0].attributes


def test_a_span_that_cannot_be_built_is_dropped_never_raised() -> None:
    """Recording runs inside the request path, so a bad value costs the span, not the request."""
    trace = _trace()

    trace.record_llm_call(_llm_call(_usage_row(id="not an identifier")))

    assert (trace.spans, trace.dropped) == ([], 1)


def test_a_tool_name_that_is_not_an_identifier_is_kept_out_of_the_span() -> None:
    trace = _trace()

    trace.record_tool_call(tool_name="delete all the files", tool_type="mcp", started=_T0, ok=True)

    [span] = trace.spans
    assert (span.name, span.tool_name) == ("execute_tool", None)


def test_spans_past_the_cap_are_counted_not_kept() -> None:
    trace = _trace(max_spans=2)

    for _ in range(5):
        trace.record_tool_call(tool_name="Bash", tool_type="mcp", started=_T0, ok=True)

    assert (len(trace.spans), trace.dropped) == (2, 3)


@pytest.mark.parametrize(
    ("status_code", "completed", "outcome", "error_class"),
    [
        (200, True, "ok", None),
        (502, True, "error", "status_502"),
        (200, False, "error", "incomplete"),
        (None, False, "error", "incomplete"),
    ],
)
def test_the_step_ends_with_what_the_client_got(
    status_code: int | None, completed: bool, outcome: str, error_class: str | None
) -> None:
    write = _trace().finish(status_code=status_code, completed=completed)

    step = write.spans[0]
    assert (step.kind, step.span_id, step.outcome, step.error_class) == ("step", "req-1", outcome, error_class)
    assert (write.trace_id, write.session_source) == ("req-1", "none")


# ---------------------------------------------------------------------------
# The usage-row tap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tap_records_a_row_on_its_requests_trace_and_still_writes_it() -> None:
    traces = RequestTraces()
    trace = _trace()
    traces.begin(trace)
    inner = AsyncMock()

    await TracingLogWriter(inner, traces).put(_usage_row())

    assert [span.kind for span in trace.spans] == ["llm"]
    inner.put.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_row_of_a_request_with_no_open_trace_is_only_written() -> None:
    inner = AsyncMock()

    await TracingLogWriter(inner, RequestTraces()).put(_usage_row(request_group_id="someone-else"))

    inner.put.assert_awaited_once()


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def _writer(store: Any, **overrides: Any) -> TraceWriter:
    settings: dict[str, Any] = {
        "max_queued_spans": 10,
        "batch_spans": 4,
        "interval_s": 0.01,
        "write_timeout_s": 1.0,
        "shutdown_s": 1.0,
    }
    return TraceWriter(store.write, **(settings | overrides))


@pytest.mark.asyncio
async def test_queued_traces_are_written_in_batches() -> None:
    store = AsyncMock()
    store.write.return_value = WriteResult(accepted=3)
    writer = _writer(store)
    await writer.start()

    writer.submit(_write(3, "a"))
    writer.submit(_write(3, "b"))
    await asyncio.sleep(0.1)
    await writer.stop()

    written = [trace.trace_id for call in store.write.await_args_list for trace in call.args[0]]
    assert written == ["a", "b"]
    assert all(sum(len(t.spans) for t in call.args[0]) <= 4 for call in store.write.await_args_list)


@pytest.mark.asyncio
async def test_an_erased_users_queued_traces_are_never_written() -> None:
    """Erasure deletes what is stored; what is still queued must not bring it back."""
    store = AsyncMock()
    store.write.return_value = WriteResult()
    writer = _writer(store, interval_s=60.0, batch_spans=100)
    writer.submit(_write(2, "theirs", user_id="bob"))
    writer.submit(_write(3, "mine"))

    assert writer.discard_user("bob") == 2
    await writer.start()
    await writer.stop()

    assert [t.trace_id for call in store.write.await_args_list for t in call.args[0]] == ["mine"]


@pytest.mark.asyncio
async def test_a_request_that_does_not_fit_is_dropped_whole() -> None:
    store = AsyncMock()
    store.write.return_value = WriteResult()
    writer = _writer(store, max_queued_spans=5)
    before = _dropped("queue_full")

    writer.submit(_write(4, "fits"))
    writer.submit(_write(3, "does-not-fit"))

    assert _dropped("queue_full") - before == 3
    await writer.stop()
    assert [t.trace_id for call in store.write.await_args_list for t in call.args[0]] == ["fits"]


@pytest.mark.asyncio
async def test_a_store_that_hangs_costs_one_timeout_then_the_writer_moves_on() -> None:
    hung = asyncio.Event()

    async def hang(traces: tuple[TraceWrite, ...]) -> WriteResult:
        await hung.wait()
        return WriteResult()

    store = AsyncMock()
    store.write.side_effect = hang
    writer = _writer(store, write_timeout_s=0.05)
    before = _dropped("timeout")
    await writer.start()

    writer.submit(_write(2))
    await asyncio.sleep(0.2)
    await writer.stop()

    assert _dropped("timeout") - before == 2


@pytest.mark.asyncio
async def test_a_store_error_drops_the_batch_and_keeps_the_writer_alive() -> None:
    store = AsyncMock()
    store.write.side_effect = [RuntimeError("store down"), WriteResult(accepted=2)]
    writer = _writer(store, batch_spans=2)
    before = _dropped("store_error")
    await writer.start()

    writer.submit(_write(2, "lost"))
    await asyncio.sleep(0.05)
    writer.submit(_write(2, "kept"))
    await asyncio.sleep(0.05)
    await writer.stop()

    assert _dropped("store_error") - before == 2
    assert store.write.await_count == 2


@pytest.mark.asyncio
async def test_one_trace_the_store_refuses_does_not_sink_the_rest_of_its_batch() -> None:
    """A batch is one transaction; a trace whose key was deleted must not cost every tenant's spans."""

    async def refuse_bad(traces: tuple[TraceWrite, ...]) -> WriteResult:
        if any(trace.trace_id == "bad" for trace in traces):
            raise IntegrityError("INSERT", {}, Exception("foreign key"))
        return WriteResult(accepted=sum(len(trace.spans) for trace in traces))

    store = AsyncMock()
    store.write.side_effect = refuse_bad
    writer = _writer(store, interval_s=60.0, batch_spans=100)
    before = _dropped("store_error")
    writer.submit(_write(2, "good"))
    writer.submit(_write(3, "bad"))
    writer.submit(_write(2, "also-good"))
    await writer.start()
    await writer.stop()

    stored = [call.args[0][0].trace_id for call in store.write.await_args_list if len(call.args[0]) == 1]
    assert sorted(stored) == ["also-good", "bad", "good"]
    assert _dropped("store_error") - before == 3


@pytest.mark.asyncio
async def test_an_unreachable_database_drops_the_batch_without_retrying_each_trace() -> None:
    """Retrying a trace at a time only helps when one trace's data is bad; an outage fails every retry."""
    store = AsyncMock()
    store.write.side_effect = OperationalError("INSERT", {}, Exception("connection refused"))
    writer = _writer(store, interval_s=60.0, batch_spans=100)
    before = _dropped("store_error")
    for name in ("a", "b", "c"):
        writer.submit(_write(2, name))
    await writer.start()
    await writer.stop()

    assert store.write.await_count == 1
    assert _dropped("store_error") - before == 6


@pytest.mark.asyncio
async def test_shutdown_flushes_within_its_limit_and_drops_the_rest() -> None:
    async def slow(traces: tuple[TraceWrite, ...]) -> WriteResult:
        await asyncio.sleep(0.2)
        return WriteResult()

    store = AsyncMock()
    store.write.side_effect = slow
    writer = _writer(store, batch_spans=2, shutdown_s=0.05, write_timeout_s=1.0)
    before = _dropped("shutdown")
    writer.submit(_write(2, "a"))
    writer.submit(_write(2, "b"))

    started = asyncio.get_running_loop().time()
    await writer.stop()

    assert asyncio.get_running_loop().time() - started < 0.5
    assert _dropped("shutdown") - before > 0


def test_submitting_never_waits_on_the_store() -> None:
    """``submit`` is a plain function: it cannot await anything a store does."""
    assert not asyncio.iscoroutinefunction(TraceWriter.submit)


# ---------------------------------------------------------------------------
# The middleware
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, writer: Any) -> None:
        self.trace_writer = writer


class _App:
    def __init__(self, writer: Any) -> None:
        self.state = _State(writer)


async def _serve(traces: RequestTraces, writer: Any, *, more_body_last: bool, raise_after_start: bool = False) -> None:
    async def downstream(scope: MutableMapping[str, Any], receive: Receive, send: Send) -> None:
        traces.begin(_trace())
        scope[TRACE_SCOPE_KEY] = "req-1"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        if raise_after_start:
            raise ConnectionResetError
        await send({"type": "http.response.body", "body": b"x", "more_body": more_body_last})

    middleware = TraceCaptureMiddleware(downstream, traces=traces)
    scope = {"type": "http", "app": _App(writer)}

    async def send(message: MutableMapping[str, Any]) -> None:
        return None

    await middleware(scope, AsyncMock(), send)


@pytest.mark.asyncio
async def test_the_middleware_submits_the_trace_once_the_body_is_sent() -> None:
    traces = RequestTraces()
    writer = AsyncMock()
    writer.submit = lambda trace, truncated=0: submitted.append(trace)
    submitted: list[TraceWrite] = []

    await _serve(traces, writer, more_body_last=False)

    [write] = submitted
    assert write.spans[0].outcome == "ok"
    assert len(traces) == 0, "the trace leaves the registry with its response"


@pytest.mark.asyncio
async def test_a_response_that_breaks_off_is_an_incomplete_step() -> None:
    traces = RequestTraces()
    writer = AsyncMock()
    submitted: list[TraceWrite] = []
    writer.submit = lambda trace, truncated=0: submitted.append(trace)

    with pytest.raises(ConnectionResetError):
        await _serve(traces, writer, more_body_last=True, raise_after_start=True)

    assert submitted[0].spans[0].error_class == "incomplete"


# ---------------------------------------------------------------------------
# A gateway-run tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_web_fetch_records_a_timed_tool_span() -> None:
    service = AsyncMock(spec=WebFetchService)
    service.fetch.return_value = WebFetchResult(
        text="Hello",
        content_type="text/html",
        content_kind="html",
        requested_url=canonicalize_web_url("https://example.com"),
        final_url=canonicalize_web_url("https://example.com"),
        redirect_count=0,
    )
    trace = _trace()
    backend = WebRetrievalBackend(
        enable_search=False, enable_fetch=True, retrieval_service=service, fetch_policy=DomainPolicy(), trace=trace
    )
    async with backend:
        await backend.call_tool(WEB_FETCH_TOOL_NAME, {"url": "https://example.com"})

    [span] = trace.spans
    assert (span.kind, span.tool_name, span.tool_type, span.outcome) == (
        "tool",
        WEB_FETCH_TOOL_NAME,
        "otari_web_fetch",
        "ok",
    )
    assert span.duration_ms is not None
