"""Unit tests for the vision side-call billing order in ``resolve_request_context``.

The vision describe call happens inside ``normalize_messages``, so by the time
the reservation top-up runs its cost is already incurred. These tests pin the
guarantee documented on ``_bill_vision_side_call``: the side-call is billed
even when the top-up rejects with 402, while the main reservation is still
refunded.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.api.routes._pipeline as pipeline
from gateway.api.routes import chat
from gateway.core.config import GatewayConfig
from gateway.services.budget_service import ReservationHandle

_VISION_USAGE = CompletionUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)


class _Recorder:
    """Captures the billing and settlement primitives the pipeline invoked."""

    def __init__(self) -> None:
        self.usage_logged: list[dict[str, Any]] = []
        self.reconciled: list[float] = []
        self.refunded = 0

    def install(self, monkeypatch: pytest.MonkeyPatch, *, topup_status: int | None) -> None:
        async def fake_verify(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
            return SimpleNamespace(id="key-1", user_id="user-1"), False

        async def fake_find_pricing(*args: Any, **kwargs: Any) -> None:
            return None

        async def fake_reserve(*args: Any, **kwargs: Any) -> ReservationHandle:
            return ReservationHandle(user_id="user-1", estimate=0.0, reserved=True, strategy="for_update")

        async def fake_increase(*args: Any, **kwargs: Any) -> None:
            if topup_status is not None:
                raise HTTPException(status_code=topup_status, detail="budget exceeded")

        async def fake_log_usage(**kwargs: Any) -> float:
            self.usage_logged.append(kwargs)
            return 0.01

        async def fake_reconcile(db: Any, handle: Any, actual_cost: float) -> None:
            self.reconciled.append(actual_cost)

        async def fake_refund(db: Any, handle: Any) -> None:
            self.refunded += 1

        monkeypatch.setattr(pipeline, "verify_api_key_or_master_key", fake_verify)
        monkeypatch.setattr(pipeline, "check_rate_limit", lambda request, user_id: None)
        monkeypatch.setattr(pipeline, "find_model_pricing", fake_find_pricing)
        monkeypatch.setattr(pipeline, "reserve_budget", fake_reserve)
        monkeypatch.setattr(pipeline, "increase_reservation", fake_increase)
        monkeypatch.setattr(pipeline, "log_usage", fake_log_usage)
        monkeypatch.setattr(pipeline, "reconcile_reservation", fake_reconcile)
        monkeypatch.setattr(pipeline, "refund_reservation", fake_refund)


async def _normalize_with_vision(
    user_id: str, provider: LLMProvider | None, model: str, instance: str | None
) -> tuple[int, CompletionUsage | None]:
    return 5000, _VISION_USAGE


async def _resolve(config: GatewayConfig) -> pipeline.RequestContext:
    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})
    return await pipeline.resolve_request_context(
        adapter=chat._ADAPTER,
        raw_request=request,
        response=Response(),
        db=cast(AsyncSession, object()),
        config=config,
        log_writer=cast(Any, object()),
        model="openai:gpt-4",
        user_id_from_request=None,
        estimate_prompt_chars=100,
        estimate_max_output_tokens=None,
        master_key_user_required_detail="user required",
        user_forbidden_detail="forbidden",
        normalize_messages=_normalize_with_vision,
    )


def _config() -> GatewayConfig:
    return GatewayConfig(require_pricing=False, vision_describe_model="openai:gpt-4o-mini")


@pytest.mark.asyncio
async def test_vision_side_call_billed_even_when_topup_rejects_402(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vision cost is already incurred, so a 402 from the reservation
    top-up must not skip its usage-log row or its committed spend; the main
    reservation is still refunded.
    """
    recorder = _Recorder()
    recorder.install(monkeypatch, topup_status=402)

    with pytest.raises(HTTPException) as exc_info:
        await _resolve(_config())

    assert exc_info.value.status_code == 402
    assert len(recorder.usage_logged) == 1
    logged = recorder.usage_logged[0]
    assert logged["model"] == "gpt-4o-mini"
    assert logged["provider"] == "openai"
    assert logged["usage_override"] is _VISION_USAGE
    assert recorder.reconciled == [0.01]
    assert recorder.refunded == 1


@pytest.mark.asyncio
async def test_vision_side_call_billed_on_successful_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch, topup_status=None)

    ctx = await _resolve(_config())

    assert ctx.user_id == "user-1"
    assert len(recorder.usage_logged) == 1
    assert recorder.reconciled == [0.01]
    assert recorder.refunded == 0
