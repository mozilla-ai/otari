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

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from any_llm.types.completion import ChatCompletionChunk, CompletionUsage
from fastapi import HTTPException

import gateway.api.routes._pipeline as pipeline
from gateway.api.routes import chat, messages, responses
from gateway.api.routes._pipeline import (
    RequestContext,
    ToolContext,
    build_streaming_response,
    prepare_gateway_tools,
    run_single_attempt_stream,
    stream_first_chunk_timeout_seconds,
)
from gateway.core.config import GatewayConfig
from gateway.services.budget_service import ReservationHandle

ADAPTERS = [
    pytest.param(chat._ADAPTER, id="chat"),
    pytest.param(messages._ADAPTER, id="messages"),
    pytest.param(responses._ADAPTER, id="responses"),
]


def _tool_ctx(**overrides: Any) -> ToolContext:
    defaults: dict[str, Any] = {
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
        rate_limit_info=None,
        reservation=reservation,
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
    monkeypatch.delenv("GATEWAY_SANDBOX_URL", raising=False)

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
