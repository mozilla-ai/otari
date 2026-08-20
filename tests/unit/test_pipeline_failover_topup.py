"""Unit tests for the mid-failover reservation top-up (issue #463).

``top_up_reservation_for_attempt`` is the only thing stopping a fallover to a
pricier candidate from spending past a cap the budget gate already approved. The
gate runs once, against the head candidate, so without this a policy whose
``on_failure`` chain climbs in price would serve the expensive model on the cheap
model's hold.

It also carries the ``require_pricing`` gate for fallback candidates, for the
same reason: the admission gate prices only the head, so an unpriced model that
would 402 when named directly would otherwise serve, and log ``cost=null``,
purely by being reached as a fallback.

These are pure unit tests. ``find_model_pricing`` and ``increase_reservation``
are stubbed, so what is pinned is the decision logic: when the hold grows, by how
much, when it does not, and which failure the caller sees.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.api.routes._pipeline as pipeline
from gateway.core.config import GatewayConfig
from gateway.services.budget_service import ReservationHandle
from gateway.types.attempt import Attempt

# Cost the stubbed estimator returns per (instance, model), standing in for the
# real pricing lookup plus `estimate_cost`.
_PRICES = {
    ("openai", "cheap"): 1.0,
    ("openai", "same"): 5.0,
    ("anthropic", "pricey"): 9.0,
}


def _attempt(position: int, instance: str, model: str) -> Attempt:
    return Attempt(
        position=position,
        instance=instance,
        provider=LLMProvider.OPENAI,
        model=model,
        kwargs={"api_key": "sk-test"},
    )


def _ctx(
    *,
    estimate: float = 5.0,
    counts_toward_budget: bool = True,
    require_pricing: bool = False,
    db: object | None = object(),
    reservation: ReservationHandle | None = None,
    estimate_inputs: object | None = None,
) -> Any:
    """A context carrying only what the top-up reads.

    Deliberately a SimpleNamespace rather than a real ``RequestContext``: the
    function touches six attributes, and building the real thing would drag in
    auth, rate limiting, and the allow-list resolver for no added coverage.
    """
    return SimpleNamespace(
        db=cast(AsyncSession, db) if db is not None else None,
        # Resolved once in the preamble on a real request. None is "this
        # organization has no rate override", so the top-up prices against the
        # deployment list, which is what ``_PRICES`` below stands in for.
        organization_id=None,
        reservation=reservation
        or ReservationHandle(
            user_id="user-1",
            estimate=estimate,
            reserved=True,
            strategy="for_update",
            counts_toward_budget=counts_toward_budget,
        ),
        estimate_inputs=(
            estimate_inputs
            if estimate_inputs is not None
            else pipeline.EstimateInputs(prompt_chars=400, max_output_tokens=100, default_output_tokens=100)
        ),
        config=GatewayConfig(require_pricing=require_pricing),
    )


@pytest.fixture
def increases(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every top-up delta, with pricing stubbed from ``_PRICES``."""
    recorded: list[float] = []

    async def fake_find_pricing(
        _db: Any, instance: str, model: str, *, organization_id: Any = None, **_kwargs: Any
    ) -> Any:
        # ``organization_id`` is named rather than swallowed by ``**_kwargs`` so
        # this fake keeps failing if the real signature stops passing it.
        price = _PRICES.get((instance, model))
        return None if price is None else SimpleNamespace(price=price)

    def fake_estimate_cost(pricing: Any, **_kwargs: Any) -> float:
        return 0.0 if pricing is None else float(pricing.price)

    async def fake_increase(_db: Any, _handle: Any, delta: float, **_kwargs: Any) -> None:
        recorded.append(delta)

    monkeypatch.setattr(pipeline, "find_model_pricing", fake_find_pricing)
    monkeypatch.setattr(pipeline, "estimate_cost", fake_estimate_cost)
    monkeypatch.setattr(pipeline, "increase_reservation", fake_increase)
    return recorded


@pytest.mark.asyncio
async def test_a_pricier_candidate_grows_the_hold_before_dispatch(increases: list[float]) -> None:
    """The delta, not the full price: the original hold already covers the head."""
    await pipeline.top_up_reservation_for_attempt(_ctx(estimate=5.0), _attempt(2, "anthropic", "pricey"))
    assert increases == [4.0]


@pytest.mark.asyncio
async def test_a_cheaper_candidate_is_a_no_op(increases: list[float]) -> None:
    """The hold only ever grows toward the candidate that serves. Shrinking it
    would release budget another request could take, on a chain that has not
    finished spending.
    """
    await pipeline.top_up_reservation_for_attempt(_ctx(estimate=5.0), _attempt(2, "openai", "cheap"))
    assert increases == []


@pytest.mark.asyncio
async def test_an_equally_priced_candidate_is_a_no_op(increases: list[float]) -> None:
    await pipeline.top_up_reservation_for_attempt(_ctx(estimate=5.0), _attempt(2, "openai", "same"))
    assert increases == []


@pytest.mark.asyncio
async def test_a_refused_top_up_stops_the_chain_with_the_failover_detail(
    monkeypatch: pytest.MonkeyPatch, increases: list[float]
) -> None:
    """The refusal is reported as its own condition rather than as a generic
    budget rejection: the caller was admitted, so "you are out of budget" alone
    would look like the gate contradicting itself.
    """

    async def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise HTTPException(status_code=402, detail="budget exceeded")

    monkeypatch.setattr(pipeline, "increase_reservation", refuse)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline.top_up_reservation_for_attempt(_ctx(estimate=5.0), _attempt(2, "anthropic", "pricey"))

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == pipeline.budget_exhausted_mid_failover_detail()
    assert "failing over" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_an_unpriced_fallback_is_refused_under_require_pricing(increases: list[float]) -> None:
    """Otherwise a model that 402s when named directly serves for free by being
    reached as a fallback, and logs cost=null.
    """
    with pytest.raises(HTTPException) as exc_info:
        await pipeline.top_up_reservation_for_attempt(
            _ctx(require_pricing=True), _attempt(2, "openai", "no-such-model")
        )

    assert exc_info.value.status_code == 402
    assert increases == []


@pytest.mark.asyncio
async def test_an_unpriced_fallback_is_allowed_when_require_pricing_is_off(increases: list[float]) -> None:
    await pipeline.top_up_reservation_for_attempt(_ctx(require_pricing=False), _attempt(2, "openai", "no-such-model"))
    assert increases == []


@pytest.mark.asyncio
async def test_a_budget_exempt_request_skips_the_pricing_gate(increases: list[float]) -> None:
    """A request that is never debited cannot overshoot a cap, so the gate does
    not apply to it. Matches the admission-time rule.
    """
    await pipeline.top_up_reservation_for_attempt(
        _ctx(require_pricing=True, counts_toward_budget=False), _attempt(2, "openai", "no-such-model")
    )
    assert increases == []


@pytest.mark.asyncio
async def test_no_reservation_means_nothing_to_top_up(increases: list[float]) -> None:
    """Budget enforcement disabled, or a user with no budget: there is no hold to
    grow, so the chain proceeds rather than failing on a missing handle.
    """
    ctx = _ctx()
    ctx.reservation = None
    await pipeline.top_up_reservation_for_attempt(ctx, _attempt(2, "anthropic", "pricey"))
    assert increases == []


@pytest.mark.asyncio
async def test_no_estimate_inputs_means_nothing_to_reprice(increases: list[float]) -> None:
    ctx = _ctx()
    ctx.estimate_inputs = None
    await pipeline.top_up_reservation_for_attempt(ctx, _attempt(2, "anthropic", "pricey"))
    assert increases == []
