"""Settlement-parity tests for the shared request pipeline.

The pipeline consolidation (issues #100 / #101) requires that the streaming
reservation-settlement callbacks are defined in exactly one place and wired
identically for every format and for both the single-attempt and
platform-fallback streaming paths. These tests pin that contract:

* every adapter (chat / messages / responses) gets all four settlement
  callbacks on both the standalone and platform header shapes;
* the first-chunk fallback timeout is read from the same config keys (with the
  tool-loop-aware variant) for every format;
* the callbacks settle the budget reservation correctly (reconcile on usage,
  policy ladder on missing usage, refund on error and on client disconnect);
* a pre-stream dispatch failure refunds the reservation for every format.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi import HTTPException, Response

import gateway.api.routes._pipeline as pipeline
from gateway.api.routes import chat, messages, responses
from gateway.api.routes._pipeline import (
    RequestContext,
    ToolContext,
    build_streaming_response,
    prepare_gateway_tools,
    run_single_attempt_stream,
    run_standalone_non_stream,
    stream_first_chunk_timeout_seconds,
)
from gateway.core.config import GatewayConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.budget_service import ReservationHandle

ADAPTERS = [
    pytest.param(chat._ADAPTER, id="chat"),
    pytest.param(messages._ADAPTER, id="messages"),
    pytest.param(responses._ADAPTER, id="responses"),
]


def _tool_ctx(**overrides: Any) -> ToolContext:
    defaults: dict[str, Any] = {
        "config": GatewayConfig(),
        "mcp_server_configs": None,
        "use_sandbox": False,
        "sandbox_tool_entry": None,
        "sandbox_url": None,
        "sandbox_auth_token": None,
        "use_web_search": False,
        "web_search_tool_entry": None,
        "web_search_url": None,
        "web_search_auth_token": None,
        "remaining_user_tools": None,
        "max_tool_iterations": 10,
        "tools_header": None,
    }
    defaults.update(overrides)
    return ToolContext(**defaults)


def _ctx(
    config: GatewayConfig,
    *,
    db: Any = None,
    log_writer: Any = None,
    reservation: ReservationHandle | None = None,
    rate_limit_info: RateLimitInfo | None = None,
) -> RequestContext:
    return RequestContext(
        config=config,
        db=db,
        log_writer=log_writer,
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=rate_limit_info,
        reservation=reservation,
        started_at=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Callback wiring parity across formats and paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("hybrid_path", [False, True], ids=["standalone", "hybrid"])
def test_all_settlement_callbacks_wired_for_every_format_and_path(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    hybrid_path: bool,
) -> None:
    """Every streaming response, regardless of format and of single-attempt vs
    platform-fallback path, must wire on_complete / on_error / on_no_usage /
    on_incomplete. Both paths build through ``build_streaming_response``, so
    asserting here covers them uniformly.
    """
    captured: dict[str, Any] = {}

    def fake_streaming_generator(**kwargs: Any) -> AsyncIterator[str]:
        captured.update(kwargs)

        async def _gen() -> AsyncIterator[str]:
            yield "data: {}\n\n"

        return _gen()

    monkeypatch.setattr(pipeline, "streaming_generator", fake_streaming_generator)

    async def _empty() -> AsyncIterator[Any]:
        return
        yield  # unreachable; makes this a generator

    response = build_streaming_response(
        adapter=adapter,
        stream=_empty(),
        provider=LLMProvider.OPENAI,
        model="m",
        config=GatewayConfig(),
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
        platform_correlation_id="corr-1" if hybrid_path else None,
        platform_request_id="req-1" if hybrid_path else None,
    )

    for callback_name in ("on_complete", "on_error", "on_no_usage", "on_incomplete"):
        assert callable(captured.get(callback_name)), f"{callback_name} not wired"
    assert captured["fmt"] is adapter.stream_format
    if hybrid_path:
        assert response.headers["X-Correlation-ID"] == "corr-1"
        assert response.headers["X-Otari-Request-ID"] == "req-1"
    else:
        assert "X-Correlation-ID" not in response.headers


# ---------------------------------------------------------------------------
# First-chunk timeout parity
# ---------------------------------------------------------------------------


def test_first_chunk_timeout_defaults() -> None:
    config = GatewayConfig()
    assert stream_first_chunk_timeout_seconds(config, tool_mode=False) == 2.0
    assert stream_first_chunk_timeout_seconds(config, tool_mode=True) == 30.0


def test_first_chunk_timeout_reads_shared_config_keys() -> None:
    config = GatewayConfig(
        platform={
            "streaming_first_chunk_timeout_ms": 500,
            "streaming_first_chunk_timeout_ms_tool_loop": 7000,
        }
    )
    assert stream_first_chunk_timeout_seconds(config, tool_mode=False) == 0.5
    assert stream_first_chunk_timeout_seconds(config, tool_mode=True) == 7.0


# ---------------------------------------------------------------------------
# Settlement behavior (shared callback bodies, exercised via the chat format)
# ---------------------------------------------------------------------------


class _Settlement:
    """Records which settlement primitives the callbacks invoked."""

    def __init__(self) -> None:
        self.reconciled: list[float] = []
        self.refunded = 0
        self.logged: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_log_usage(**kwargs: Any) -> float | None:
            self.logged.append(kwargs)
            usage = kwargs.get("usage_override")
            if kwargs.get("cost_override") is not None:
                return float(kwargs["cost_override"])
            return 0.25 if usage else None

        async def fake_reconcile(db: Any, handle: Any, actual_cost: float) -> None:
            self.reconciled.append(actual_cost)

        async def fake_refund(db: Any, handle: Any) -> None:
            self.refunded += 1

        monkeypatch.setattr(pipeline, "log_usage", fake_log_usage)
        monkeypatch.setattr(pipeline, "reconcile_reservation", fake_reconcile)
        monkeypatch.setattr(pipeline, "refund_reservation", fake_refund)


def _chunk(usage: CompletionUsage | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c1",
        choices=[],
        created=0,
        model="m",
        object="chat.completion.chunk",
        usage=usage,
    )


def _reservation(estimate: float = 0.5) -> ReservationHandle:
    return ReservationHandle(user_id="user-1", estimate=estimate, reserved=True, strategy="for_update")


def _build(stream: AsyncIterator[ChatCompletionChunk], config: GatewayConfig) -> Any:
    return build_streaming_response(
        adapter=chat._ADAPTER,
        stream=stream,
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        config=config,
        db=cast(Any, object()),
        log_writer=cast(Any, object()),
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        reservation=_reservation(),
    )


async def _drain(response: Any) -> list[str]:
    return [chunk async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_stream_with_usage_reconciles_actual_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    await _drain(_build(stream(), GatewayConfig()))

    assert settlement.reconciled == [0.25]
    assert settlement.refunded == 0


@pytest.mark.asyncio
async def test_stream_without_usage_allow_free_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk()

    await _drain(_build(stream(), GatewayConfig(stream_missing_usage_policy="allow_free")))

    assert settlement.refunded == 1
    assert settlement.reconciled == []


@pytest.mark.asyncio
async def test_stream_without_usage_estimate_policy_charges_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk()

    await _drain(_build(stream(), GatewayConfig(stream_missing_usage_policy="estimate")))

    assert settlement.reconciled == [0.5]
    assert settlement.refunded == 0


@pytest.mark.asyncio
async def test_stream_error_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk()
        raise RuntimeError("upstream broke")

    await _drain(_build(stream(), GatewayConfig()))

    assert settlement.refunded == 1
    assert settlement.reconciled == []


@pytest.mark.asyncio
async def test_client_disconnect_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk()
        yield _chunk()

    response = _build(stream(), GatewayConfig())
    iterator = response.body_iterator
    await iterator.__anext__()
    # Closing the generator mid-stream simulates a client disconnect.
    await iterator.aclose()

    assert settlement.refunded == 1
    assert settlement.reconciled == []


# ---------------------------------------------------------------------------
# Platform usage reports scheduled from streaming callbacks are not lost
# ---------------------------------------------------------------------------


def _build_platform(stream: AsyncIterator[ChatCompletionChunk]) -> Any:
    return build_streaming_response(
        adapter=chat._ADAPTER,
        stream=stream,
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        config=GatewayConfig(),
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
        platform_correlation_id="corr-1",
        platform_request_id="req-1",
    )


@pytest.mark.parametrize("yields_chunk", [True, False], ids=["no-usage-chunk", "zero-chunk"])
@pytest.mark.asyncio
async def test_platform_stream_without_usage_reports_final_success(
    monkeypatch: pytest.MonkeyPatch,
    yields_chunk: bool,
) -> None:
    reports: list[dict[str, Any]] = []

    async def completed_report() -> None:
        return None

    def fake_report(**kwargs: Any) -> Any:
        reports.append(kwargs)
        return completed_report()

    def fake_schedule(report: Any, correlation_id: str) -> None:
        assert correlation_id == "corr-1"
        report.close()

    monkeypatch.setattr(pipeline, "_report_platform_usage", fake_report)
    monkeypatch.setattr(pipeline, "_schedule_usage_report", fake_schedule)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        if yields_chunk:
            yield _chunk()

    await _drain(_build_platform(stream()))

    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report.pop("config"), GatewayConfig)
    assert report == {
        "correlation_id": "corr-1",
        "outcome": "success",
        "usage": None,
        "session_label": None,
        "is_final_attempt": True,
    }


@pytest.mark.asyncio
async def test_platform_usage_report_task_is_tracked_until_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled report task must be strongly referenced while in flight
    (a bare fire-and-forget task can be garbage collected mid-run) and
    discarded once it completes.
    """
    tracked: set[asyncio.Task[None]] = set()
    monkeypatch.setattr(pipeline, "_USAGE_REPORT_TASKS", tracked)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_report(**kwargs: Any) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(pipeline, "_report_platform_usage", slow_report)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    await _drain(_build_platform(stream()))
    await asyncio.wait_for(started.wait(), timeout=2)

    assert len(tracked) == 1
    task = next(iter(tracked))
    release.set()
    await asyncio.wait_for(task, timeout=2)
    for _ in range(10):
        if not tracked:
            break
        await asyncio.sleep(0)
    assert not tracked


@pytest.mark.asyncio
async def test_failed_platform_usage_report_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report that raises must surface in the logs instead of vanishing
    with the unreferenced task.
    """
    monkeypatch.setattr(pipeline, "_USAGE_REPORT_TASKS", set(), raising=False)
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    class _LoggerRecorder:
        def warning(self, msg: str, *args: Any) -> None:
            warnings.append((msg, args))

        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    monkeypatch.setattr(pipeline, "logger", _LoggerRecorder())

    async def failing_report(**kwargs: Any) -> None:
        raise RuntimeError("usage endpoint down")

    monkeypatch.setattr(pipeline, "_report_platform_usage", failing_report)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    await _drain(_build_platform(stream()))
    for _ in range(20):
        if warnings:
            break
        await asyncio.sleep(0)

    assert warnings, "failed usage report was not logged"
    assert "corr-1" in warnings[0][1]


# ---------------------------------------------------------------------------
# Rejections after the budget pre-debit release the reservation
# ---------------------------------------------------------------------------


async def _call_prepare_gateway_tools(ctx: RequestContext, **overrides: Any) -> ToolContext:
    from fastapi import Response

    kwargs: dict[str, Any] = {
        "adapter": chat._ADAPTER,
        "ctx": ctx,
        "response": Response(),
        "guardrails": None,
        "guardrail_text": "",
        "tools": None,
        "mcp_servers": None,
        "mcp_server_ids": None,
        "max_tool_iterations": None,
        "tools_header": None,
    }
    kwargs.update(overrides)
    return await prepare_gateway_tools(**kwargs)


@pytest.mark.asyncio
async def test_tool_misconfiguration_400_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation())
    with pytest.raises(HTTPException) as exc_info:
        await _call_prepare_gateway_tools(ctx, tools=[{"type": "otari_code_execution"}])

    assert exc_info.value.status_code == 400
    assert settlement.refunded == 1


@pytest.mark.asyncio
async def test_mcp_server_ids_in_standalone_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation())
    with pytest.raises(HTTPException) as exc_info:
        await _call_prepare_gateway_tools(
            ctx, mcp_server_ids=[cast(Any, "11111111-1111-1111-1111-111111111111")]
        )

    assert exc_info.value.status_code == 400
    assert settlement.refunded == 1


@pytest.mark.asyncio
async def test_guardrail_block_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def blocking_guardrails(*args: Any, **kwargs: Any) -> None:
        raise HTTPException(status_code=403, detail="blocked")

    monkeypatch.setattr(pipeline, "apply_input_guardrails", blocking_guardrails)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation())
    with pytest.raises(HTTPException) as exc_info:
        await _call_prepare_gateway_tools(ctx)

    assert exc_info.value.status_code == 403
    assert settlement.refunded == 1


# ---------------------------------------------------------------------------
# Pre-stream dispatch failures refund the reservation for every format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "module", "provider_fn"),
    [
        pytest.param(chat._ADAPTER, chat, "acompletion", id="chat"),
        pytest.param(messages._ADAPTER, messages, "amessages", id="messages"),
        pytest.param(responses._ADAPTER, responses, "aresponses", id="responses"),
    ],
)
@pytest.mark.asyncio
async def test_pre_stream_failure_refunds_reservation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    module: Any,
    provider_fn: str,
) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def failing_provider(**kwargs: Any) -> Any:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(module, provider_fn, failing_provider)

    ctx = _ctx(
        GatewayConfig(),
        db=object(),
        log_writer=object(),
        reservation=_reservation(),
    )

    with pytest.raises(HTTPException):
        await run_single_attempt_stream(
            adapter=adapter,
            ctx=ctx,
            tool_ctx=_tool_ctx(),
            call_kwargs={"model": "openai:gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            provider=LLMProvider.OPENAI,
            model="gpt-4",
        )

    assert settlement.refunded == 1
    assert settlement.reconciled == []


# ---------------------------------------------------------------------------
# Non-streaming settlement (characterization, issue #463 P1a)
#
# `run_standalone_non_stream` and `run_platform_non_stream` are the pair that
# the routing-policy work merges into one multi-attempt executor. Neither had a
# single direct test before this block: both were exercised only through
# route-level HTTP tests, and the standalone half only in the Postgres-backed
# integration tier. So these pin today's behavior in the fast tier, and the
# same assertions must hold after the merge.
#
# The asymmetry being pinned, and the whole reason the merge is not a move:
# the standalone path reserves budget, logs usage, reconciles, and relabels the
# model; `run_platform_non_stream` takes no `ctx` at all, so it does none of
# those four things. Whatever executes both must acquire local settlement
# without changing either side's observable behavior.
# ---------------------------------------------------------------------------


def _completion(model: str = "gpt-4", usage: CompletionUsage | None = None) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
            )
        ],
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )


def _usage() -> CompletionUsage:
    return CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


async def _run_standalone(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: ChatCompletion | None = None,
    error: BaseException | None = None,
    reservation: ReservationHandle | None = None,
    rate_limit_info: RateLimitInfo | None = None,
    db: Any = None,
    display_model: str | None = None,
) -> tuple[Any, Response]:
    """Drive the standalone non-streaming path with a faked provider call."""

    async def fake_acompletion(**kwargs: Any) -> Any:
        if error is not None:
            raise error
        return result if result is not None else _completion()

    monkeypatch.setattr(chat, "acompletion", fake_acompletion)

    ctx = _ctx(
        GatewayConfig(),
        db=db if db is not None else object(),
        log_writer=object(),
        reservation=reservation,
        rate_limit_info=rate_limit_info,
    )
    response = Response()
    returned = await run_standalone_non_stream(
        adapter=chat._ADAPTER,
        ctx=ctx,
        tool_ctx=_tool_ctx(),
        call_kwargs={"model": "openai:gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        response=response,
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        display_model=display_model,
    )
    return returned, response


@pytest.mark.asyncio
async def test_standalone_non_stream_success_logs_once_and_reconciles(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    result, _ = await _run_standalone(
        monkeypatch, result=_completion(usage=_usage()), reservation=_reservation()
    )

    assert result.usage is not None
    assert len(settlement.logged) == 1
    assert settlement.reconciled == [0.25]
    assert settlement.refunded == 0


@pytest.mark.asyncio
async def test_standalone_non_stream_logs_success_without_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat sets ``log_success_without_usage``, so a usage-less result still
    writes a row and still reconciles (at cost 0.0, since the fake returns None).
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    await _run_standalone(monkeypatch, result=_completion(usage=None), reservation=_reservation())

    assert len(settlement.logged) == 1
    assert settlement.reconciled == [0.0]
    assert settlement.refunded == 0


@pytest.mark.asyncio
async def test_standalone_non_stream_keys_usage_on_target_and_relabels_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias relabels the response ``model`` while usage keys on the target.

    This is the invariant the routing-policy work must preserve for a policy
    name: what the caller sees and what gets billed are deliberately different.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    result, _ = await _run_standalone(
        monkeypatch,
        result=_completion(usage=_usage()),
        reservation=_reservation(),
        display_model="fast",
    )

    assert result.model == "fast"
    assert settlement.logged[0]["model"] == "gpt-4"
    assert settlement.logged[0]["provider"] is LLMProvider.OPENAI


@pytest.mark.asyncio
async def test_standalone_non_stream_applies_rate_limit_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    _, response = await _run_standalone(
        monkeypatch,
        result=_completion(usage=_usage()),
        reservation=_reservation(),
        rate_limit_info=RateLimitInfo(limit=60, remaining=59, reset=time.time() + 60),
    )

    assert response.headers["X-RateLimit-Limit"] == "60"
    assert response.headers["X-RateLimit-Remaining"] == "59"


@pytest.mark.asyncio
async def test_standalone_non_stream_provider_failure_logs_error_and_refunds_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    with pytest.raises(HTTPException):
        await _run_standalone(monkeypatch, error=RuntimeError("connection refused"), reservation=_reservation())

    assert settlement.refunded == 1
    assert settlement.reconciled == []
    assert len(settlement.logged) == 1
    assert settlement.logged[0]["error"] == "connection refused"
    assert settlement.logged[0]["status_code"] is not None


@pytest.mark.asyncio
async def test_standalone_non_stream_http_exception_refunds_without_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-mapped HTTPException refunds but writes no usage row.

    Gateway-side rejections log through ``log_gateway_rejection`` at their own
    call sites, so logging here would double-count them.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    with pytest.raises(HTTPException):
        await _run_standalone(
            monkeypatch,
            error=HTTPException(status_code=403, detail="nope"),
            reservation=_reservation(),
        )

    assert settlement.refunded == 1
    assert settlement.logged == []


@pytest.mark.asyncio
async def test_standalone_non_stream_without_reservation_settles_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Master-key callers reserve nothing, so there is nothing to reconcile."""
    settlement = _Settlement()
    settlement.install(monkeypatch)

    await _run_standalone(monkeypatch, result=_completion(usage=_usage()), reservation=None)

    assert len(settlement.logged) == 1
    assert settlement.reconciled == []
    assert settlement.refunded == 0


@pytest.mark.asyncio
async def test_standalone_non_stream_marks_budget_exempt_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``exclude_from_budget`` key holds an unreserved handle: the row is
    still written, flagged as not counting toward the budget.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    exempt = ReservationHandle(
        user_id="user-1", estimate=0.0, reserved=False, strategy="for_update", counts_toward_budget=False
    )
    await _run_standalone(monkeypatch, result=_completion(usage=_usage()), reservation=exempt)

    assert settlement.logged[0]["counts_toward_budget"] is False


@pytest.mark.asyncio
async def test_hybrid_shaped_context_settles_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``db=None`` (the hybrid shape) the standalone path writes and
    settles nothing, which is exactly what ``run_platform_non_stream`` does
    today by having no ``ctx`` at all. The merged executor must keep this
    difference driven by the presence of a local session, not by the mode flag.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    ctx = _ctx(GatewayConfig(), db=None, log_writer=None, reservation=None)

    async def fake_acompletion(**kwargs: Any) -> Any:
        return _completion(usage=_usage())

    monkeypatch.setattr(chat, "acompletion", fake_acompletion)

    await run_standalone_non_stream(
        adapter=chat._ADAPTER,
        ctx=ctx,
        tool_ctx=_tool_ctx(),
        call_kwargs={"model": "openai:gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        response=Response(),
        provider=LLMProvider.OPENAI,
        model="gpt-4",
    )

    assert settlement.logged == []
    assert settlement.reconciled == []
    assert settlement.refunded == 0


@pytest.mark.asyncio
async def test_standalone_stream_has_no_first_chunk_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow first token must not be cut off on the standalone streaming path.

    The first-chunk deadline exists so a hybrid-mode attempt can fail over to
    the next entry in the routing policy, and it is read from
    ``config.platform`` (empty in standalone), defaulting to 2000ms. Standalone
    has no next entry, so ``run_single_attempt_stream`` deliberately never
    applies a deadline at all.

    This pins that. The config below sets a 10ms deadline and the first chunk
    takes ~50ms: if the standalone path ever starts consulting the cap (say by
    being routed through the multi-attempt walker), this test fails instead of
    turning slow-but-valid responses into 504s in production, which is what
    issue #237 was.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def slow_first_chunk(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def gen() -> AsyncIterator[ChatCompletionChunk]:
            await asyncio.sleep(0.05)
            yield _chunk(_usage())

        return gen()

    monkeypatch.setattr(chat, "acompletion", slow_first_chunk)

    config = GatewayConfig(platform={"streaming_first_chunk_timeout_ms": 10})
    assert stream_first_chunk_timeout_seconds(config, tool_mode=False) == 0.01

    ctx = _ctx(config, db=object(), log_writer=object(), reservation=_reservation())
    response = await run_single_attempt_stream(
        adapter=chat._ADAPTER,
        ctx=ctx,
        tool_ctx=_tool_ctx(config=config),
        call_kwargs={"model": "openai:gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        provider=LLMProvider.OPENAI,
        model="gpt-4",
    )
    chunks = await _drain(response)

    assert chunks, "the slow first chunk was dropped: a first-chunk deadline is being applied"
    assert settlement.reconciled == [0.25]
