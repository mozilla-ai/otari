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
import re
import time
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm import LLMProvider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from fastapi import BackgroundTasks, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError

import gateway.api.routes._pipeline as pipeline
import gateway.streaming as streaming
from gateway.api.routes import chat, messages, responses
from gateway.api.routes._pipeline import (
    RequestContext,
    ToolContext,
    build_streaming_response,
    log_usage,
    prepare_gateway_tools,
    run_platform_non_stream,
    run_single_attempt_stream,
    run_standalone_non_stream,
    stream_first_chunk_timeout_seconds,
)
from gateway.api.routes._platform import ResolvedAttempt, ResolvedRoute, SettledCost
from gateway.core.config import GatewayConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.budget_service import ReservationHandle
from gateway.services.tenancy.errors import WorkspaceMcpServerNotFoundError

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


# The preamble resolves the organization from the workspace on every standalone
# request, so a context without one is a shape production never builds; the
# organization-guardrail resolve refuses it (fail closed) rather than skipping
# the mandates. Defaulted here so these tests keep exercising the refusals they
# are about instead of that one.
_ORGANIZATION_ID = uuid.UUID("99999999-9999-9999-9999-999999999999")


def _ctx(
    config: GatewayConfig,
    *,
    db: Any = None,
    log_writer: Any = None,
    reservation: ReservationHandle | None = None,
    rate_limit_info: RateLimitInfo | None = None,
    workspace_id: uuid.UUID | None = None,
    organization_id: uuid.UUID | None = _ORGANIZATION_ID,
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
        workspace_id=workspace_id,
        organization_id=organization_id,
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


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    ("interval_ms", "expected_seconds"),
    [(15000, 15.0), (2500, 2.5), (0, 0.0)],
    ids=["default", "custom", "disabled"],
)
def test_keepalive_interval_wired_from_config_for_every_format(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    interval_ms: int,
    expected_seconds: float,
) -> None:
    """The transport keepalive interval reaches the generator for every format.

    Without this the config field is inert and a slow time-to-first-token still
    writes nothing downstream until an intermediary severs the connection.
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

    build_streaming_response(
        adapter=adapter,
        stream=_empty(),
        provider=LLMProvider.OPENAI,
        model="m",
        config=GatewayConfig(streaming_keepalive_interval_ms=interval_ms),
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
    )

    assert captured["keepalive_interval_seconds"] == expected_seconds


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


def _reservation(estimate: Decimal = Decimal("0.5")) -> ReservationHandle:
    return ReservationHandle(user_id="user-1", estimate=estimate, reserved=True, strategy="for_update")


def _build(
    stream: AsyncIterator[ChatCompletionChunk],
    config: GatewayConfig,
    *,
    workspace_id: uuid.UUID | None = None,
) -> Any:
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
        workspace_id=workspace_id,
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


class _FakeLogWriter:
    def __init__(self) -> None:
        self.put_rows: list[Any] = []

    async def put(self, row: Any) -> None:
        self.put_rows.append(row)


@pytest.mark.asyncio
async def test_log_usage_skips_the_workspace_lookup_when_given_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of threading ``workspace_id`` through: a caller that
    already resolved it must not pay ``workspace_for_key_id``'s own lookup a
    second time. For a master-key request (``api_key_id=None``) that lookup is
    the un-memoized ``default_workspace_id`` query the preamble already paid.
    """
    lookups = 0

    async def counting_workspace_for_key_id(db: Any, api_key_id: str | None) -> uuid.UUID:
        nonlocal lookups
        lookups += 1
        return uuid.uuid4()

    monkeypatch.setattr(pipeline, "workspace_for_key_id", counting_workspace_for_key_id)
    workspace_id = uuid.uuid4()

    await log_usage(
        db=cast(Any, object()),
        log_writer=cast(Any, _FakeLogWriter()),
        api_key_id=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        workspace_id=workspace_id,
    )

    assert lookups == 0


@pytest.mark.asyncio
async def test_log_usage_still_resolves_the_workspace_when_not_given_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback stays intact for the callers that have no context to draw
    ``workspace_id`` from (a rejection logged before one exists, the vision
    side-call)."""
    lookups = 0
    resolved = uuid.uuid4()

    async def counting_workspace_for_key_id(db: Any, api_key_id: str | None) -> uuid.UUID:
        nonlocal lookups
        lookups += 1
        return resolved

    monkeypatch.setattr(pipeline, "workspace_for_key_id", counting_workspace_for_key_id)
    log_writer = _FakeLogWriter()

    await log_usage(
        db=cast(Any, object()),
        log_writer=cast(Any, log_writer),
        api_key_id=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
    )

    assert lookups == 1
    assert log_writer.put_rows[0].workspace_id == resolved


@pytest.mark.asyncio
async def test_stream_settlement_forwards_the_contexts_workspace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ctx.workspace_id``, already resolved once in the preamble, reaches
    ``log_usage`` rather than being silently dropped and re-derived there
    (otari#643 follow-up: a master-key request must not pay the un-memoized
    default-workspace lookup twice for one request).
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)
    workspace_id = uuid.uuid4()

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    await _drain(_build(stream(), GatewayConfig(), workspace_id=workspace_id))

    assert settlement.logged[0]["workspace_id"] == workspace_id


@pytest.mark.asyncio
async def test_standalone_non_stream_forwards_the_contexts_workspace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)
    workspace_id = uuid.uuid4()

    await _run_standalone(
        monkeypatch,
        result=_completion(usage=_usage()),
        reservation=_reservation(),
        workspace_id=workspace_id,
    )

    assert settlement.logged[0]["workspace_id"] == workspace_id


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


@pytest.mark.asyncio
async def test_platform_stream_holds_terminal_usage_until_cost_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter_started = asyncio.Event()
    reporter_release = asyncio.Event()

    async def fake_report(**kwargs: Any) -> SettledCost:
        reporter_started.set()
        await reporter_release.wait()
        return SettledCost(cost_usd="0.012345", pricing_source="managed")

    monkeypatch.setattr(pipeline, "_report_platform_usage", fake_report)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="content-1",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="hello", role="assistant"),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=0,
            model="gpt-4",
            object="chat.completion.chunk",
            usage=None,
        )
        yield _chunk(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    response = build_streaming_response(
        adapter=chat._ADAPTER,
        stream=stream(),
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        config=GatewayConfig(platform={"usage_inline_timeout_ms": 1000}),
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
        platform_correlation_id="corr-1",
    )
    iterator = cast(AsyncIterator[Any], response.body_iterator)

    first = await iterator.__anext__()
    assert "hello" in first
    pending_terminal: asyncio.Future[Any] = asyncio.ensure_future(iterator.__anext__())
    await asyncio.wait_for(reporter_started.wait(), timeout=1)
    assert not pending_terminal.done()

    reporter_release.set()
    terminal = await asyncio.wait_for(pending_terminal, timeout=1)
    assert '"cost_usd":"0.012345"' in terminal
    assert '"pricing_source":"managed"' in terminal


@pytest.mark.asyncio
async def test_inline_settlement_timeout_leaves_accounting_report_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked: set[asyncio.Task[SettledCost | None]] = set()
    monkeypatch.setattr(pipeline, "_USAGE_REPORT_TASKS", tracked)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_report() -> SettledCost:
        started.set()
        await release.wait()
        return SettledCost(cost_usd="0.012345", pricing_source="managed")

    result = await pipeline._await_usage_report(
        slow_report(),
        "corr-timeout",
        GatewayConfig(platform={"usage_inline_timeout_ms": 10}),
    )

    assert result is None
    assert started.is_set()
    assert len(tracked) == 1
    task = next(iter(tracked))
    assert not task.cancelled()

    release.set()
    assert await asyncio.wait_for(task, timeout=1) == SettledCost(
        cost_usd="0.012345",
        pricing_source="managed",
    )


@pytest.mark.asyncio
async def test_platform_non_stream_awaits_and_attaches_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter_finished = False

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return _completion(usage=_usage())

    async def fake_report(**kwargs: Any) -> SettledCost:
        nonlocal reporter_finished
        await asyncio.sleep(0)
        reporter_finished = True
        return SettledCost(cost_usd="0.012345", pricing_source="managed")

    monkeypatch.setattr(chat, "acompletion", fake_acompletion)
    monkeypatch.setattr(pipeline, "_report_platform_usage", fake_report)
    attempt = ResolvedAttempt(
        attempt_id="3f1b6a1e-0000-4000-8000-000000000002",
        position=0,
        provider="openai",
        model="gpt-4",
        api_key="sk-test",
        managed=True,
    )

    result = await run_platform_non_stream(
        adapter=chat._ADAPTER,
        route=ResolvedRoute(request_id="req-1", fallback_enabled=False, attempts=[attempt]),
        base_request_fields={"messages": [{"role": "user", "content": "hi"}]},
        tool_ctx=_tool_ctx(),
        response=Response(),
        background_tasks=BackgroundTasks(),
        config=GatewayConfig(platform={"usage_inline_timeout_ms": 1000}),
        rate_limit_info=None,
    )

    assert reporter_finished is True
    assert result.usage is not None
    assert getattr(result.usage, "cost_usd", None) == "0.012345"
    assert getattr(result.usage, "pricing_source", None) == "managed"


# ---------------------------------------------------------------------------
# Terminal cost buffering: only the last carrier is held
# ---------------------------------------------------------------------------


def _content_chunk(chunk_id: str, text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=chunk_id,
        choices=[ChunkChoice(delta=ChoiceDelta(content=text, role="assistant"), finish_reason=None, index=0)],
        created=0,
        model="gpt-4",
        object="chat.completion.chunk",
        usage=None,
    )


def _usage_chunk(chunk_id: str, prompt_tokens: int) -> ChatCompletionChunk:
    """An OpenAI ``include_usage`` chunk: usage, no choices."""
    return ChatCompletionChunk(
        id=chunk_id,
        choices=[],
        created=0,
        model="gpt-4",
        object="chat.completion.chunk",
        usage=CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1),
    )


class _BlockedReporter:
    """A usage reporter that parks until released, so ordering is observable."""

    def __init__(self, settlement: SettledCost | None = None) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._settlement = settlement or SettledCost(cost_usd="0.012345", pricing_source="managed")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_report(**kwargs: Any) -> SettledCost | None:
            self.started.set()
            await self.release.wait()
            return self._settlement

        monkeypatch.setattr(pipeline, "_report_platform_usage", fake_report)


def _build_platform_stream(
    stream: AsyncIterator[ChatCompletionChunk],
    *,
    config: GatewayConfig | None = None,
    display_model: str | None = None,
) -> AsyncIterator[str]:
    response = build_streaming_response(
        adapter=chat._ADAPTER,
        stream=stream,
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        config=config or GatewayConfig(platform={"usage_inline_timeout_ms": 5000}),
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
        display_model=display_model,
        platform_correlation_id="corr-1",
    )
    return cast(AsyncIterator[str], response.body_iterator)


@pytest.mark.asyncio
async def test_intermediate_usage_chunk_does_not_hold_the_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat tool loop forwards one ``include_usage`` chunk per iteration.

    The first one matches the carrier predicate but is not terminal, so holding
    it would park the next iteration's answer behind settlement and leave cost on
    a chunk the client has already passed.
    """
    reporter = _BlockedReporter()
    reporter.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _content_chunk("c0", "thinking")
        yield _usage_chunk("u1", 10)  # iteration 1's usage chunk, not terminal
        yield _content_chunk("c1", "final")
        yield _content_chunk("c2", "answer")
        yield _usage_chunk("u2", 20)  # the real terminal carrier

    iterator = _build_platform_stream(stream())

    # Everything up to the real terminal carrier streams without waiting.
    streamed = [await asyncio.wait_for(iterator.__anext__(), timeout=1) for _ in range(4)]
    assert [_chunk_id(part) for part in streamed] == ["c0", "u1", "c1", "c2"]
    assert not reporter.started.is_set()

    pending_terminal = asyncio.ensure_future(iterator.__anext__())
    await asyncio.wait_for(reporter.started.wait(), timeout=1)
    assert not pending_terminal.done()

    reporter.release.set()
    terminal = await asyncio.wait_for(pending_terminal, timeout=1)
    assert _chunk_id(terminal) == "u2"
    assert '"cost_usd":"0.012345"' in terminal
    assert not any("cost_usd" in part for part in streamed)


@pytest.mark.asyncio
async def test_carrier_overflow_resumes_streaming_and_settles_on_a_later_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More chunks than the cap after a carrier means it was not terminal.

    The held chunks are released in order and a later carrier still takes the
    cost, so a long final answer neither stalls nor loses inline cost.
    """
    reporter = _BlockedReporter()
    reporter.install(monkeypatch)
    reporter.release.set()
    debug_messages: list[str] = []

    class _LoggerRecorder:
        def debug(self, msg: str, *args: Any) -> None:
            debug_messages.append(msg % args if args else msg)

        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    monkeypatch.setattr(streaming, "logger", _LoggerRecorder())

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _usage_chunk("u1", 10)
        for index in range(6):
            yield _content_chunk(f"c{index}", f"part-{index}")
        yield _usage_chunk("u2", 20)

    emitted = [part async for part in _build_platform_stream(stream())]

    assert [_chunk_id(part) for part in emitted if _chunk_id(part)] == [
        "u1",
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "u2",
    ]
    terminal = next(part for part in emitted if _chunk_id(part) == "u2")
    assert '"cost_usd":"0.012345"' in terminal
    assert '"cost_usd"' not in next(part for part in emitted if _chunk_id(part) == "u1")
    assert emitted[-1] == "data: [DONE]\n\n"
    assert sum("was not terminal" in message for message in debug_messages) == 1


@pytest.mark.asyncio
async def test_buffered_terminal_carrier_is_relabeled_to_the_display_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The buffered path must relabel exactly like the unbuffered one.

    Hybrid passes no ``display_model`` today; this keeps the buffer from becoming
    the reason a real provider model leaks past an alias later.
    """
    reporter = _BlockedReporter()
    reporter.install(monkeypatch)
    reporter.release.set()

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _content_chunk("c0", "hi")
        yield _usage_chunk("u1", 10)

    emitted = [part async for part in _build_platform_stream(stream(), display_model="my-alias")]

    assert all("gpt-4" not in part for part in emitted)
    terminal = next(part for part in emitted if _chunk_id(part) == "u1")
    assert '"model":"my-alias"' in terminal
    assert '"cost_usd":"0.012345"' in terminal


@pytest.mark.asyncio
async def test_mid_stream_error_emits_the_buffered_carrier_before_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _BlockedReporter()
    reporter.install(monkeypatch)
    reporter.release.set()

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _content_chunk("c0", "hi")
        yield _usage_chunk("u1", 10)
        raise RuntimeError("upstream died")

    emitted = [part async for part in _build_platform_stream(stream())]

    assert [_chunk_id(part) for part in emitted if _chunk_id(part)] == ["c0", "u1"]
    assert "cost_usd" not in "".join(emitted)
    assert "An error occurred during streaming" in emitted[-2]
    assert emitted[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_keepalives_continue_while_terminal_settlement_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _BlockedReporter()
    reporter.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _usage_chunk("u1", 10)

    iterator = _build_platform_stream(
        stream(),
        config=GatewayConfig(
            streaming_keepalive_interval_ms=10,
            platform={"usage_inline_timeout_ms": 5000},
        ),
    )

    first = await asyncio.wait_for(iterator.__anext__(), timeout=1)
    assert first == ": keepalive\n\n"

    reporter.release.set()
    rest = [part async for part in iterator]
    assert _chunk_id(rest[0]) == "u1"
    assert '"cost_usd":"0.012345"' in rest[0]


@pytest.mark.asyncio
async def test_cancellation_while_settling_leaves_the_report_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked: set[asyncio.Task[SettledCost | None]] = set()
    monkeypatch.setattr(pipeline, "_USAGE_REPORT_TASKS", tracked)
    reporter = _BlockedReporter()
    reporter.install(monkeypatch)

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield _usage_chunk("u1", 10)

    iterator = _build_platform_stream(stream())
    pending = asyncio.ensure_future(iterator.__anext__())
    await asyncio.wait_for(reporter.started.wait(), timeout=1)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await cast(Any, iterator).aclose()

    assert len(tracked) == 1
    report_task = next(iter(tracked))
    assert not report_task.cancelled()

    reporter.release.set()
    assert await asyncio.wait_for(report_task, timeout=1) == SettledCost(
        cost_usd="0.012345",
        pricing_source="managed",
    )


def _chunk_id(part: str) -> str | None:
    """The ``id`` of a chunk in an SSE data line, or None for a non-data frame."""
    match = re.search(r'"id":"([^"]+)"', part)
    return match.group(1) if match else None


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
    # Stubbed rather than fed a session: every case here is about a *different*
    # admission refusal, and the organization plane resolves before all of them.
    with patch("gateway.api.routes._pipeline.resolve_organization_guardrails", new=AsyncMock(return_value=[])):
        return await prepare_gateway_tools(**kwargs)


@pytest.mark.asyncio
async def test_tool_misconfiguration_400_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation(), workspace_id=uuid.uuid4())
    with pytest.raises(HTTPException) as exc_info:
        await _call_prepare_gateway_tools(ctx, tools=[{"type": "otari_code_execution"}])

    assert exc_info.value.status_code == 400
    assert settlement.refunded == 1


@pytest.mark.asyncio
async def test_a_request_without_a_workspace_is_refused_before_any_tool_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone tool resolution needs a workspace; with none the request is refused, not served.

    Every gate in this block fails closed on a context carrying no workspace,
    and the organization-guardrail resolve is the first of them (otari#654), so
    it is the one that answers: a 500, because an unresolvable tenancy is a
    server invariant rather than something the caller sent wrong. What the case
    is really about is that the request is refused rather than served with its
    tool configuration silently dropped, and that the hold does not survive it.
    `_resolve_mcp_server_ids` keeps its own guard on the same condition; it is
    simply no longer the first to run.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation())
    with pytest.raises(HTTPException) as exc_info:
        await _call_prepare_gateway_tools(
            ctx, mcp_server_ids=[cast(Any, "11111111-1111-1111-1111-111111111111")]
        )

    assert exc_info.value.status_code == 500
    assert settlement.refunded == 1


@pytest.mark.asyncio
async def test_unknown_mcp_server_id_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An id naming no server in the request's workspace is a 404, the same answer hybrid mode gives."""
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def missing(*args: Any, **kwargs: Any) -> list[Any]:
        raise WorkspaceMcpServerNotFoundError("11111111-1111-1111-1111-111111111111")

    monkeypatch.setattr(pipeline, "resolve_workspace_mcp_servers", missing)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation(), workspace_id=uuid.uuid4())
    with pytest.raises(HTTPException) as exc_info:
        await _call_prepare_gateway_tools(
            ctx, mcp_server_ids=[cast(Any, "11111111-1111-1111-1111-111111111111")]
        )

    assert exc_info.value.status_code == 404
    assert settlement.refunded == 1


@pytest.mark.asyncio
async def test_a_database_failure_releases_the_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `SQLAlchemyError` is not an `HTTPException`, and must still not leave the hold behind.

    Two reads in `prepare_gateway_tools` touch the database, so a failure in
    either would otherwise escape the release and hold the estimate against
    `users.reserved` until the budget's next reset.
    """
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def failing(*args: Any, **kwargs: Any) -> list[Any]:
        raise SQLAlchemyError("connection reset")

    monkeypatch.setattr(pipeline, "resolve_workspace_mcp_servers", failing)

    db = AsyncMock()
    ctx = _ctx(GatewayConfig(), db=cast(Any, db), reservation=_reservation(), workspace_id=uuid.uuid4())
    with pytest.raises(SQLAlchemyError):
        await _call_prepare_gateway_tools(
            ctx, mcp_server_ids=[cast(Any, "11111111-1111-1111-1111-111111111111")]
        )

    assert settlement.refunded == 1
    assert db.rollback.await_count == 1, "the session is rolled back first, or the release cannot run"


@pytest.mark.asyncio
async def test_a_release_that_also_fails_reraises_the_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database still refusing work must surface the first failure, not a second one."""
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def failing(*args: Any, **kwargs: Any) -> list[Any]:
        raise SQLAlchemyError("connection reset")

    async def failing_release(*args: Any, **kwargs: Any) -> None:
        raise SQLAlchemyError("still down")

    monkeypatch.setattr(pipeline, "resolve_workspace_mcp_servers", failing)
    monkeypatch.setattr(pipeline, "release_reservation", failing_release)

    db = AsyncMock()
    db.rollback.side_effect = SQLAlchemyError("still down")
    ctx = _ctx(GatewayConfig(), db=cast(Any, db), reservation=_reservation(), workspace_id=uuid.uuid4())
    with pytest.raises(SQLAlchemyError, match="connection reset"):
        await _call_prepare_gateway_tools(
            ctx, mcp_server_ids=[cast(Any, "11111111-1111-1111-1111-111111111111")]
        )


@pytest.mark.asyncio
async def test_guardrail_block_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    settlement = _Settlement()
    settlement.install(monkeypatch)

    async def blocking_guardrails(*args: Any, **kwargs: Any) -> None:
        raise HTTPException(status_code=403, detail="blocked")

    monkeypatch.setattr(pipeline, "apply_input_guardrails", blocking_guardrails)

    ctx = _ctx(GatewayConfig(), db=cast(Any, object()), reservation=_reservation(), workspace_id=uuid.uuid4())
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
    workspace_id: uuid.UUID | None = None,
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
        workspace_id=workspace_id,
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
        user_id="user-1", estimate=Decimal(0), reserved=False, strategy="for_update", counts_toward_budget=False
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
