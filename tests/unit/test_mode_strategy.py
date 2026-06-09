"""Unit tests for the request mode-strategy seam.

These exercise each strategy / settlement in isolation — selection, the
platform usage-report scheduling, and the standalone log + reconcile/refund
behavior — without spinning up a full request. The end-to-end behavior is
covered by the standalone and platform-mode integration suites.
"""

from __future__ import annotations

from typing import Any

import pytest
from any_llm import LLMProvider
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException, Response
from starlette.requests import Request

from gateway.api.routes import _mode_strategy
from gateway.api.routes._mode_strategy import (
    PlatformSettlement,
    PlatformStrategy,
    RequestSettlement,
    ResolveErrors,
    ResolveSpec,
    StandaloneSettlement,
    StandaloneStrategy,
    select_request_mode_strategy,
)
from gateway.core.config import GatewayConfig
from gateway.services.budget_service import ReservationHandle

_PLATFORM_TOKEN_ENV_VARS = ("OTARI_AI_TOKEN", "OTARI_PLATFORM_TOKEN", "ANY_LLM_PLATFORM_TOKEN")


def _clear_platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PLATFORM_TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": raw_headers})


def _spec(**overrides: Any) -> ResolveSpec:
    err = HTTPException(status_code=400, detail="boom")
    defaults: dict[str, Any] = {
        "model_selector": "openai:gpt-4o-mini",
        "user_id_from_request": None,
        "prompt_chars": 10,
        "max_output_tokens": None,
        "errors": ResolveErrors(
            db_unavailable=HTTPException(status_code=500, detail="Database session unavailable"),
            master_key_user_required=err,
            api_key_validation_failed=err,
            no_user=err,
            forbidden_user=err,
            no_pricing=HTTPException(status_code=402, detail="no pricing"),
        ),
    }
    defaults.update(overrides)
    return ResolveSpec(**defaults)


def _reservation() -> ReservationHandle:
    return ReservationHandle(user_id="u1", estimate=0.5, reserved=True, strategy="for_update")


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def test_select_strategy_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_platform_env(monkeypatch)
    config = GatewayConfig()
    strategy = select_request_mode_strategy(config, db=None, log_writer=object())  # type: ignore[arg-type]
    assert isinstance(strategy, StandaloneStrategy)
    assert strategy.is_platform is False


def test_select_strategy_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    config = GatewayConfig(mode="platform", platform={"base_url": "http://platform.test/api/v1"})
    strategy = select_request_mode_strategy(config, db=None, log_writer=object())  # type: ignore[arg-type]
    assert isinstance(strategy, PlatformStrategy)
    assert strategy.is_platform is True


# ---------------------------------------------------------------------------
# PlatformSettlement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_settlement_success_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, CompletionUsage | None]] = []

    async def _fake_report(
        *, config: Any, correlation_id: str, outcome: str, usage: Any, error_class: Any = None
    ) -> None:
        calls.append((correlation_id, outcome, usage))

    monkeypatch.setattr(_mode_strategy, "_report_platform_usage", _fake_report)
    settlement: RequestSettlement = PlatformSettlement(config=GatewayConfig(), correlation_id="corr-1")
    usage = CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)

    await settlement.on_success(usage)
    await _drain_tasks()

    assert calls == [("corr-1", "success", usage)]


@pytest.mark.asyncio
async def test_platform_settlement_error_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, Any]] = []

    async def _fake_report(
        *, config: Any, correlation_id: str, outcome: str, usage: Any, error_class: Any = None
    ) -> None:
        calls.append((correlation_id, outcome, usage))

    monkeypatch.setattr(_mode_strategy, "_report_platform_usage", _fake_report)
    settlement = PlatformSettlement(config=GatewayConfig(), correlation_id="corr-2")

    await settlement.on_error("boom")
    await _drain_tasks()

    assert calls == [("corr-2", "error", None)]


@pytest.mark.asyncio
async def test_platform_settlement_no_usage_and_incomplete_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    async def _fake_report(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(_mode_strategy, "_report_platform_usage", _fake_report)
    settlement = PlatformSettlement(config=GatewayConfig(), correlation_id="corr-3")

    await settlement.on_no_usage()
    await settlement.on_incomplete()
    await settlement.on_provider_error_precommit("boom")
    await _drain_tasks()

    assert calls == []


# ---------------------------------------------------------------------------
# StandaloneSettlement
# ---------------------------------------------------------------------------


def _patch_budget(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    recorded: dict[str, list[Any]] = {"log": [], "reconcile": [], "refund": []}

    async def _fake_log_usage(**kwargs: Any) -> float | None:
        recorded["log"].append(kwargs)
        return 0.5

    async def _fake_reconcile(db: Any, reservation: Any, amount: float) -> None:
        recorded["reconcile"].append((reservation, amount))

    async def _fake_refund(db: Any, reservation: Any) -> None:
        recorded["refund"].append(reservation)

    monkeypatch.setattr("gateway.api.routes.chat.log_usage", _fake_log_usage)
    monkeypatch.setattr(_mode_strategy, "reconcile_reservation", _fake_reconcile)
    monkeypatch.setattr(_mode_strategy, "refund_reservation", _fake_refund)
    return recorded


def _standalone_settlement(config: GatewayConfig, reservation: ReservationHandle | None) -> StandaloneSettlement:
    return StandaloneSettlement(
        db=object(),  # type: ignore[arg-type]
        log_writer=object(),  # type: ignore[arg-type]
        api_key_id="key-1",
        user_id="u1",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        endpoint="/v1/chat/completions",
        reservation=reservation,
        config=config,
    )


@pytest.mark.asyncio
async def test_standalone_success_logs_and_reconciles(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    settlement = _standalone_settlement(GatewayConfig(), reservation)
    usage = CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)

    await settlement.on_success(usage)

    assert len(recorded["log"]) == 1
    assert recorded["log"][0]["usage_override"] is usage
    assert recorded["reconcile"] == [(reservation, 0.5)]
    assert recorded["refund"] == []


@pytest.mark.asyncio
async def test_standalone_error_logs_and_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    settlement = _standalone_settlement(GatewayConfig(), reservation)

    await settlement.on_error("boom")

    assert recorded["log"][0]["error"] == "boom"
    assert recorded["refund"] == [reservation]
    assert recorded["reconcile"] == []


@pytest.mark.asyncio
async def test_standalone_incomplete_refunds_without_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    settlement = _standalone_settlement(GatewayConfig(), reservation)

    await settlement.on_incomplete()

    assert recorded["log"] == []
    assert recorded["refund"] == [reservation]


@pytest.mark.asyncio
async def test_standalone_precommit_logs_without_refund(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    settlement = _standalone_settlement(GatewayConfig(), reservation)

    await settlement.on_provider_error_precommit("boom")

    assert recorded["log"][0]["error"] == "boom"
    assert recorded["refund"] == []
    assert recorded["reconcile"] == []


@pytest.mark.asyncio
async def test_standalone_no_usage_allow_free_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    config = GatewayConfig(stream_missing_usage_policy="allow_free")
    settlement = _standalone_settlement(config, reservation)

    await settlement.on_no_usage()

    assert len(recorded["log"]) == 1
    assert recorded["log"][0].get("error") is None
    assert recorded["log"][0].get("cost_override") is None
    assert recorded["refund"] == [reservation]
    assert recorded["reconcile"] == []


@pytest.mark.asyncio
async def test_standalone_no_usage_estimate_charges_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    config = GatewayConfig(stream_missing_usage_policy="estimate")
    settlement = _standalone_settlement(config, reservation)

    await settlement.on_no_usage()

    assert recorded["log"][0]["cost_override"] == reservation.estimate
    assert recorded["log"][0].get("error") is None
    assert recorded["reconcile"] == [(reservation, reservation.estimate)]
    assert recorded["refund"] == []


@pytest.mark.asyncio
async def test_standalone_no_usage_fail_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    reservation = _reservation()
    config = GatewayConfig(stream_missing_usage_policy="fail")
    settlement = _standalone_settlement(config, reservation)

    await settlement.on_no_usage()

    assert recorded["log"][0]["error"] == "stream completed without usage data"
    assert recorded["log"][0]["cost_override"] == reservation.estimate
    assert recorded["reconcile"] == [(reservation, reservation.estimate)]


@pytest.mark.asyncio
async def test_standalone_settlement_noop_without_db(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _patch_budget(monkeypatch)
    settlement = StandaloneSettlement(
        db=None,
        log_writer=object(),  # type: ignore[arg-type]
        api_key_id=None,
        user_id=None,
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        endpoint="/v1/chat/completions",
        reservation=_reservation(),
        config=GatewayConfig(),
    )

    await settlement.on_success(None)
    await settlement.on_error("boom")
    await settlement.on_incomplete()

    assert recorded == {"log": [], "reconcile": [], "refund": []}


# ---------------------------------------------------------------------------
# resolve() reject paths that need no DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_resolve_missing_bearer_raises_401() -> None:
    strategy = PlatformStrategy(GatewayConfig(platform={"base_url": "http://platform.test"}))
    request = _make_request()  # no Authorization header
    with pytest.raises(HTTPException) as ei:
        await strategy.resolve(raw_request=request, response=Response(), spec=_spec())
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_standalone_resolve_db_none_raises_spec_error() -> None:
    strategy = StandaloneStrategy(GatewayConfig(), db=None, log_writer=object())  # type: ignore[arg-type]
    request = _make_request()
    with pytest.raises(HTTPException) as ei:
        await strategy.resolve(raw_request=request, response=Response(), spec=_spec())
    assert ei.value.status_code == 500
    assert ei.value.detail == "Database session unavailable"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _drain_tasks() -> None:
    """Yield control so any ``asyncio.create_task`` scheduled by a settlement runs."""
    import asyncio

    for _ in range(3):
        await asyncio.sleep(0)
