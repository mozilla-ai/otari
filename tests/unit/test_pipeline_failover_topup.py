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

These are pure unit tests. ``resolve_model_pricing`` and ``increase_reservation``
are stubbed, so what is pinned is the decision logic: when the hold grows, by how
much, when it does not, and which failure the caller sees.
"""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.api.routes._pipeline as pipeline
from gateway.api.routes._attempts import CandidateCannotServe
from gateway.core.config import GatewayConfig
from gateway.core.error_codes import BUDGET_EXCEEDED, error_headers
from gateway.exceptions.budget_exceptions import ProviderSpendCeilingExceededError, SpendCeilingExceededError
from gateway.services import pricing_service
from gateway.services.budgets import EstimateInputs, ReservationHandle
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
    estimate: Decimal = Decimal(5),
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
        uow=object(),
        rate_limit_grant=None,
        # Resolved once in the preamble on a real request. None is "this
        # organization has no rate override", so the top-up prices against the
        # deployment list, which is what ``_PRICES`` below stands in for.
        organization_id=None,
        # No memo, as for a context the preamble priced nothing on: the top-up resolves its own.
        prices=None,
        reservation=reservation
        or ReservationHandle(
            user_id="user-1",
            estimate=estimate,
            reserved=True,
            strategy="for_update",
            counts_toward_budget=counts_toward_budget,
            scope_ceilings=(),
        ),
        estimate_inputs=(
            estimate_inputs
            if estimate_inputs is not None
            else EstimateInputs(prompt_chars=400, max_output_tokens=100, default_output_tokens=100)
        ),
        config=GatewayConfig(require_pricing=require_pricing),
    )


@pytest.fixture
def increases(monkeypatch: pytest.MonkeyPatch) -> list[Decimal]:
    """Record every size a top-up grows the reservation to, with pricing stubbed from ``_PRICES``."""
    recorded: list[Decimal] = []

    async def fake_find_pricing(_db: Any, instance: str, model: str, *, organization_id: Any, **_kwargs: Any) -> Any:
        # ``organization_id`` is named, and deliberately has no default, so this
        # fake stops binding if the real call site drops the keyword. With a
        # default it would bind cleanly and the regression would pass silently,
        # which is the opposite of what naming it here is for.
        price = _PRICES.get((instance, model))
        if price is None:
            return None
        rate = SimpleNamespace(price=price, input_price_per_million=price, output_price_per_million=price)
        return SimpleNamespace(pricing=rate)

    def fake_estimate_cost(pricing: Any, **_kwargs: Any) -> Decimal:
        return Decimal(0) if pricing is None else Decimal(str(pricing.price))

    async def fake_increase(_db: Any, _handle: Any, *, estimate: Decimal, **_kwargs: Any) -> None:
        recorded.append(estimate)

    # The top-up prices through the request's price memo, which resolves here.
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", fake_find_pricing)
    monkeypatch.setattr(pipeline, "estimate_cost", fake_estimate_cost)
    monkeypatch.setattr(pipeline, "increase_reservation", fake_increase)
    return recorded


@pytest.mark.asyncio
async def test_a_fallback_grows_the_hold_to_its_own_estimate(increases: list[Decimal]) -> None:
    """The candidate's whole estimate, because each leg grows only by what it lacks of it."""
    await pipeline.top_up_reservation_for_attempt(_ctx(estimate=Decimal(5)), _attempt(2, "anthropic", "pricey"))
    assert increases == [Decimal(9)]


@pytest.mark.asyncio
async def test_a_refused_top_up_stops_the_chain_with_the_failover_detail(
    monkeypatch: pytest.MonkeyPatch, increases: list[Decimal]
) -> None:
    """The refusal is reported as its own condition rather than as a generic
    budget rejection: the caller was admitted, so "you are out of budget" alone
    would look like the gate contradicting itself.
    """

    async def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise HTTPException(status_code=402, detail="budget exceeded")

    monkeypatch.setattr(pipeline, "increase_reservation", refuse)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline.top_up_reservation_for_attempt(_ctx(estimate=Decimal(5)), _attempt(2, "anthropic", "pricey"))

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == pipeline.budget_exhausted_mid_failover_detail()
    assert "failing over" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_a_refused_top_up_keeps_the_budget_code(
    monkeypatch: pytest.MonkeyPatch, increases: list[Decimal]
) -> None:
    """The detail is replaced, but the code and scope a client maps the refusal by are not."""
    refusal_headers = error_headers(BUDGET_EXCEEDED, budget_scope="user")

    async def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise HTTPException(status_code=403, detail="budget exceeded", headers=refusal_headers)

    monkeypatch.setattr(pipeline, "increase_reservation", refuse)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline.top_up_reservation_for_attempt(_ctx(estimate=Decimal(5)), _attempt(2, "anthropic", "pricey"))

    assert exc_info.value.detail == pipeline.budget_exhausted_mid_failover_detail()
    assert exc_info.value.headers == refusal_headers


@pytest.mark.asyncio
async def test_an_unpriced_fallback_is_refused_under_require_pricing(increases: list[Decimal]) -> None:
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
async def test_an_unpriced_fallback_is_allowed_when_require_pricing_is_off(increases: list[Decimal]) -> None:
    await pipeline.top_up_reservation_for_attempt(_ctx(require_pricing=False), _attempt(2, "openai", "no-such-model"))
    assert increases == [Decimal(0)]


@pytest.mark.asyncio
async def test_a_budget_exempt_request_skips_the_pricing_gate(increases: list[Decimal]) -> None:
    """A request that is never debited cannot overshoot a cap, so the gate does
    not apply to it. Matches the admission-time rule.
    """
    await pipeline.top_up_reservation_for_attempt(
        _ctx(require_pricing=True, counts_toward_budget=False), _attempt(2, "openai", "no-such-model")
    )
    assert increases == [Decimal(0)]


@pytest.mark.asyncio
async def test_no_reservation_means_nothing_to_top_up(increases: list[Decimal]) -> None:
    """Budget enforcement disabled, or a user with no budget: there is no hold to
    grow, so the chain proceeds rather than failing on a missing handle.
    """
    ctx = _ctx()
    ctx.reservation = None
    await pipeline.top_up_reservation_for_attempt(ctx, _attempt(2, "anthropic", "pricey"))
    assert increases == []


@pytest.mark.asyncio
async def test_no_estimate_inputs_means_nothing_to_reprice(increases: list[Decimal]) -> None:
    ctx = _ctx()
    ctx.estimate_inputs = None
    await pipeline.top_up_reservation_for_attempt(ctx, _attempt(2, "anthropic", "pricey"))
    assert increases == []


class _Budgets:
    """Stands in for the budget service a move goes through, recording the provider of each move."""

    def __init__(self) -> None:
        self.moves: list[str] = []
        self.error: BaseException | None = None

    async def move_reservation(self, _handle: Any, attempt: Attempt, **_kwargs: Any) -> None:
        self.moves.append(attempt.instance)
        if self.error is not None:
            raise self.error


@pytest.fixture
def budgets(monkeypatch: pytest.MonkeyPatch) -> _Budgets:
    fake = _Budgets()
    monkeypatch.setattr(pipeline, "get_budget_service", lambda _uow, _db: fake)
    return fake


def _refused_by(provider_key_id: str | None) -> SpendCeilingExceededError:
    error = ProviderSpendCeilingExceededError if provider_key_id is not None else SpendCeilingExceededError
    return error("Workspace has exceeded budget limit", scope_type="workspace")


@pytest.mark.asyncio
async def test_a_fallback_is_moved_onto_its_providers_budgets(budgets: _Budgets) -> None:
    await pipeline.move_reservation_for_attempt(_ctx(), _attempt(2, "anthropic", "pricey"))

    assert budgets.moves == ["anthropic"]


@pytest.mark.asyncio
async def test_every_candidate_is_moved_when_the_walk_admits_it(budgets: _Budgets) -> None:
    admit = pipeline._candidate_admission(_ctx())

    await admit(_attempt(1, "openai", "cheap"))
    await admit(_attempt(2, "anthropic", "pricey"))

    assert budgets.moves == ["openai", "anthropic"]


@pytest.mark.asyncio
async def test_a_fallback_whose_providers_budget_is_full_is_skipped(budgets: _Budgets) -> None:
    """A full budget on the fallback's provider says nothing about the next candidate, so the walk moves on."""
    budgets.error = _refused_by("anthropic")

    with pytest.raises(CandidateCannotServe) as exc_info:
        await pipeline.move_reservation_for_attempt(_ctx(), _attempt(2, "anthropic", "pricey"))

    refusal = exc_info.value.refusal
    assert (refusal.status_code, refusal.detail) == (403, "Workspace has exceeded budget limit")
    assert refusal.headers == error_headers(BUDGET_EXCEEDED, budget_scope="workspace")


@pytest.mark.asyncio
async def test_a_full_budget_on_every_provider_stops_the_chain(budgets: _Budgets) -> None:
    budgets.error = _refused_by(None)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline.move_reservation_for_attempt(_ctx(), _attempt(2, "anthropic", "pricey"))

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == pipeline.budget_exhausted_mid_failover_detail()
    assert exc_info.value.headers == error_headers(BUDGET_EXCEEDED, budget_scope="workspace")


@pytest.mark.asyncio
async def test_a_database_failure_during_a_move_stops_the_chain(budgets: _Budgets) -> None:
    """A database failure is not the provider's, so the walk must not try the next candidate as if it were."""
    budgets.error = OperationalError("UPDATE scoped_budgets", {}, Exception("connection lost"))

    with pytest.raises(HTTPException) as exc_info:
        await pipeline.move_reservation_for_attempt(_ctx(), _attempt(2, "anthropic", "pricey"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == pipeline.BUDGETS_UNAVAILABLE_DETAIL
