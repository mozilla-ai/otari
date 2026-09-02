"""Tests for budget enforcement behavior.

The counters are compared exactly rather than with ``pytest.approx``. They are
``NUMERIC(18, 6)`` as of mozilla-ai/otari#691, so an exact comparison is the
honest one, and it is also the only one that would notice a counter regressing
to binary floating point: the drift is far too small for ``approx`` to see.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from gateway.models.entities import MAX_COUNT_LIMIT, Budget, ModelPricing, User
from gateway.repositories.users_repository import get_active_user
from gateway.services.budget_service import (
    estimate_cost,
    increase_reservation,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)


@pytest.mark.asyncio
async def test_reserve_budget_reads_user_without_locking(
    async_db: Any,
) -> None:
    """reserve_budget should read the user without a row lock (no for_update)."""
    budget = Budget(
        budget_id="race-budget",
        max_budget=10.0,
    )
    async_db.add(budget)

    user = User(
        user_id="race-user",
        spend=9.0,
        budget_id="race-budget",
    )
    async_db.add(user)
    await async_db.commit()

    with patch(
        "gateway.services.budget_service.get_active_user",
        wraps=get_active_user,
    ) as mock_get_active_user:
        handle = await reserve_budget(async_db, "race-user", 0.5, strategy="cas")

    assert handle.reserved
    assert mock_get_active_user.call_args.kwargs.get("for_update", False) is False


@pytest.mark.asyncio
async def test_reserve_budget_rejects_at_limit(
    async_db: Any,
) -> None:
    """A user already at the budget limit is rejected, even for a zero-cost request."""
    budget = Budget(
        budget_id="full-budget",
        max_budget=10.0,
    )
    async_db.add(budget)

    user = User(
        user_id="full-user",
        spend=10.0,
        budget_id="full-budget",
    )
    async_db.add(user)
    await async_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, "full-user", 0.0)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_reservation_accumulates_to_prevent_overspend(async_db: Any) -> None:
    """Two reservations evaluated against the same committed spend cannot both
    pass — the second sees the first's hold (F2 TOCTOU fix).

    This is the core of the overspend-race fix: before, concurrent requests all
    read stale ``spend`` and each added its cost afterward, blowing the cap. Now
    each reservation atomically commits its estimate to ``reserved``, so the
    second request is rejected even though it was checked against the same
    starting spend.
    """
    async_db.add(Budget(budget_id="resv-budget", max_budget=10.0))
    async_db.add(User(user_id="resv-user", spend=9.0, budget_id="resv-budget"))
    await async_db.commit()

    # First reservation fits: 9.0 + 0.8 <= 10.
    handle = await reserve_budget(async_db, "resv-user", 0.8)
    assert handle.reserved

    # A naive check against committed spend (9.0) would pass the second request
    # too; it is rejected because reserved is now held.
    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, "resv-user", 0.8)
    assert exc_info.value.status_code == 403

    # expire_all() forces fresh reads — the reservation UPDATEs use
    # synchronize_session=False and the test session has expire_on_commit=False.
    async_db.expire_all()
    user = await get_active_user(async_db, "resv-user")
    assert user is not None
    assert user.reserved == Decimal("0.8")
    assert user.spend == Decimal("9.0")


@pytest.mark.asyncio
async def test_reconcile_records_actual_cost_and_releases_hold(async_db: Any) -> None:
    """reconcile_reservation adds the actual cost to spend and frees the estimate."""
    async_db.add(Budget(budget_id="rec-budget", max_budget=100.0))
    async_db.add(User(user_id="rec-user", spend=10.0, budget_id="rec-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "rec-user", 5.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "rec-user")
    assert user is not None and user.reserved == Decimal("5.0")

    await reconcile_reservation(async_db, handle, 3.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "rec-user")
    assert user is not None
    assert user.spend == Decimal("13.0")  # 10 + actual 3 (not the 5 estimate)
    assert user.reserved == Decimal("0.0")


@pytest.mark.asyncio
async def test_refund_releases_hold_without_charging(async_db: Any) -> None:
    """refund_reservation releases the estimate without recording any spend."""
    async_db.add(Budget(budget_id="ref-budget", max_budget=100.0))
    async_db.add(User(user_id="ref-user", spend=10.0, budget_id="ref-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "ref-user", 5.0)
    await refund_reservation(async_db, handle)

    async_db.expire_all()
    user = await get_active_user(async_db, "ref-user")
    assert user is not None
    assert user.spend == Decimal("10.0")  # unchanged
    assert user.reserved == Decimal("0.0")


def test_estimate_cost_clamps_negative_output_tokens() -> None:
    """A hostile negative max_output_tokens must not produce a negative estimate."""
    pricing = ModelPricing(model_key="openai:gpt-4o", input_price_per_million=2.5, output_price_per_million=10.0)
    est = estimate_cost(pricing, prompt_chars=400, max_output_tokens=-1_000_000, default_output_tokens=1024)
    # Output term clamped to 0 → only the prompt contributes; never negative.
    assert est >= 0.0
    assert est == Decimal("0.000250")


@pytest.mark.asyncio
async def test_reserve_budget_clamps_negative_estimate(async_db: Any) -> None:
    """A negative estimate must not reduce users.reserved (budget-gate bypass)."""
    async_db.add(Budget(budget_id="neg-budget", max_budget=100.0))
    async_db.add(User(user_id="neg-user", spend=10.0, reserved=4.0, budget_id="neg-budget"))
    await async_db.commit()

    await reserve_budget(async_db, "neg-user", -50.0)

    async_db.expire_all()
    user = await get_active_user(async_db, "neg-user")
    assert user is not None
    assert user.reserved == Decimal("4.0")  # unchanged: negative clamped to 0
    assert user.spend == Decimal("10.0")


@pytest.mark.asyncio
async def test_increase_reservation_grows_hold_and_folds_handle(async_db: Any) -> None:
    """A fitting top-up adds to `reserved` and folds the delta into the handle."""
    async_db.add(Budget(budget_id="inc-budget", max_budget=100.0))
    async_db.add(User(user_id="inc-user", spend=10.0, budget_id="inc-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "inc-user", 5.0)
    await increase_reservation(async_db, handle, 7.0)

    # Delta folded into the handle so the single reconcile/refund covers it all.
    assert handle.estimate == Decimal("12.0")
    async_db.expire_all()
    user = await get_active_user(async_db, "inc-user")
    assert user is not None and user.reserved == Decimal("12.0")

    # Reconcile releases the full held amount.
    await reconcile_reservation(async_db, handle, 9.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "inc-user")
    assert user is not None
    assert user.reserved == Decimal("0.0")
    assert user.spend == Decimal("19.0")  # 10 + actual 9


@pytest.mark.asyncio
async def test_increase_reservation_rejects_without_touching_original(async_db: Any) -> None:
    """An over-budget top-up raises and leaves the original hold for the caller.

    Like ``reserve_budget``, ``increase_reservation`` does not self-refund — the
    request routes own refunding on failure. The rejected delta must not have
    been added to ``reserved`` (the atomic UPDATE either fully applies or not).
    """
    async_db.add(Budget(budget_id="incr-budget", max_budget=10.0))
    async_db.add(User(user_id="incr-user", spend=8.0, budget_id="incr-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "incr-user", 1.0)  # 8 + 1 <= 10, fits
    # Topping up by 5 would need 8 + 1 + 5 = 14 > 10 → rejected.
    with pytest.raises(HTTPException) as exc_info:
        await increase_reservation(async_db, handle, 5.0)
    assert exc_info.value.status_code == 403

    # Only the original 1.0 hold remains; the delta was not applied. The caller
    # (a request route) is responsible for refunding the original on failure.
    async_db.expire_all()
    user = await get_active_user(async_db, "incr-user")
    assert user is not None
    assert user.reserved == Decimal("1.0")
    assert user.spend == Decimal("8.0")

    await refund_reservation(async_db, handle)
    async_db.expire_all()
    user = await get_active_user(async_db, "incr-user")
    assert user is not None and user.reserved == Decimal("0.0")


@pytest.mark.asyncio
async def test_increase_reservation_noop_for_nonpositive_delta(async_db: Any) -> None:
    """A zero/negative delta leaves the reservation untouched."""
    async_db.add(Budget(budget_id="incn-budget", max_budget=100.0))
    async_db.add(User(user_id="incn-user", spend=0.0, budget_id="incn-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "incn-user", 5.0)
    await increase_reservation(async_db, handle, 0.0)
    await increase_reservation(async_db, handle, -3.0)

    assert handle.estimate == Decimal("5.0")
    async_db.expire_all()
    user = await get_active_user(async_db, "incn-user")
    assert user is not None and user.reserved == Decimal("5.0")


@pytest.mark.asyncio
async def test_reconcile_clamps_negative_cost(async_db: Any) -> None:
    """A negative actual_cost must not reduce users.spend."""
    async_db.add(Budget(budget_id="negc-budget", max_budget=100.0))
    async_db.add(User(user_id="negc-user", spend=10.0, budget_id="negc-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "negc-user", 5.0)
    await reconcile_reservation(async_db, handle, -3.0)

    async_db.expire_all()
    user = await get_active_user(async_db, "negc-user")
    assert user is not None
    assert user.spend == Decimal("10.0")  # not reduced by the negative cost
    assert user.reserved == Decimal("0.0")  # hold released


@pytest.mark.asyncio
async def test_reserve_budget_enforces_a_token_only_budget(async_db: Any) -> None:
    """A budget with no dollar cap still binds through its token cap, and settles at the actual."""
    async_db.add(Budget(budget_id="token-budget", max_budget=None, token_limit=5_000))
    async_db.add(User(user_id="token-user", budget_id="token-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "token-user", 0.0, estimated_tokens=1_500)

    assert handle.reserved
    user = await get_active_user(async_db, "token-user")
    assert user is not None
    assert (user.current_tokens, user.reserved_tokens) == (0, 1_500)

    await reconcile_reservation(async_db, handle, 0.0, actual_tokens=900)

    await async_db.refresh(user)
    assert (user.current_tokens, user.reserved_tokens) == (900, 0)


@pytest.mark.asyncio
async def test_reserve_budget_rejects_at_the_token_limit(async_db: Any) -> None:
    """A user whose window has reached its token cap is rejected, estimate or no estimate."""
    async_db.add(Budget(budget_id="spent-tokens", max_budget=None, token_limit=1_000))
    async_db.add(User(user_id="spent-token-user", budget_id="spent-tokens", current_tokens=1_000))
    await async_db.commit()

    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(async_db, "spent-token-user", 0.0, estimated_tokens=0)

    assert refusal.value.status_code == 403


@pytest.mark.asyncio
async def test_reserve_budget_rejects_over_the_request_limit(async_db: Any) -> None:
    """A request cap admits exactly its count and refuses the next one."""
    async_db.add(Budget(budget_id="two-requests", max_budget=None, request_limit=2))
    async_db.add(User(user_id="counted-user", budget_id="two-requests"))
    await async_db.commit()

    for _ in range(2):
        handle = await reserve_budget(async_db, "counted-user", 0.0)
        await reconcile_reservation(async_db, handle, 0.0)

    user = await get_active_user(async_db, "counted-user")
    assert user is not None
    assert (user.current_requests, user.reserved_requests) == (2, 0)

    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(async_db, "counted-user", 0.0)

    assert refusal.value.status_code == 403


@pytest.mark.asyncio
async def test_a_top_up_grows_the_token_hold_without_counting_a_second_request(async_db: Any) -> None:
    """Attachments expanding the prompt grow the token hold; the request stays one request."""
    async_db.add(Budget(budget_id="topped-up", max_budget=10.0, token_limit=10_000, request_limit=1))
    async_db.add(User(user_id="topped-up-user", budget_id="topped-up"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "topped-up-user", 1.0, estimated_tokens=1_000)
    await increase_reservation(async_db, handle, Decimal("0.5"), additional_tokens=500)

    user = await get_active_user(async_db, "topped-up-user")
    assert user is not None
    assert user.reserved_tokens == 1_500
    # A request cap of one would refuse the top-up if it took a second hold.
    assert user.reserved_requests == 1
    assert handle.token_estimate == 1_500

    await refund_reservation(async_db, handle)

    await async_db.refresh(user)
    assert (user.reserved_tokens, user.reserved_requests) == (0, 0)


@pytest.mark.asyncio
async def test_a_free_model_still_spends_a_token_budget(async_db: Any) -> None:
    """A model priced at zero costs no dollars, but it still consumes tokens.

    The free-model shortcut used to return a reservation holding nothing, which
    let a client pass a token or request cap by naming a free-priced model.
    """
    async_db.add(ModelPricing(model_key="openai:free-model", input_price_per_million=0.0, output_price_per_million=0.0))
    async_db.add(Budget(budget_id="free-tokens", max_budget=None, token_limit=1_000))
    async_db.add(User(user_id="free-model-user", budget_id="free-tokens"))
    await async_db.commit()

    handle = await reserve_budget(
        async_db, "free-model-user", 5.0, model="openai:free-model", estimated_tokens=400
    )

    # The dollar axis is not held: the request cannot spend.
    assert handle.estimate == Decimal(0)
    assert handle.token_estimate == 400

    await reconcile_reservation(async_db, handle, 0.0, actual_tokens=400)

    user = await get_active_user(async_db, "free-model-user")
    assert user is not None
    assert (user.current_tokens, user.reserved_tokens) == (400, 0)
    assert user.spend == Decimal(0)

    # And the cap binds: three more of these do not fit under 1000.
    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(
            async_db, "free-model-user", 5.0, model="openai:free-model", estimated_tokens=700
        )

    assert refusal.value.status_code == 403


@pytest.mark.asyncio
async def test_a_free_model_on_a_dollars_only_budget_still_reserves_nothing(async_db: Any) -> None:
    """The hot path is unchanged where no count is capped: no hold, no ledger row."""
    async_db.add(ModelPricing(model_key="openai:free-two", input_price_per_million=0.0, output_price_per_million=0.0))
    async_db.add(Budget(budget_id="dollars-only", max_budget=10.0))
    async_db.add(User(user_id="dollars-only-user", budget_id="dollars-only"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "dollars-only-user", 5.0, model="openai:free-two", estimated_tokens=400)

    assert not handle.reserved
    assert handle.reservation_id is None
    assert (handle.estimate, handle.token_estimate) == (Decimal(0), 0)


@pytest.mark.asyncio
async def test_a_token_only_top_up_is_held(async_db: Any) -> None:
    """Attachments can expand the prompt without moving the price, and that still holds.

    ``increase_reservation`` used to return on the dollar delta alone, so the
    extra tokens were settled against a hold that had never grown to cover them.
    """
    async_db.add(Budget(budget_id="token-topup", max_budget=10.0, token_limit=10_000))
    async_db.add(User(user_id="token-topup-user", budget_id="token-topup"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "token-topup-user", 1.0, estimated_tokens=1_000)
    await increase_reservation(async_db, handle, Decimal("0"), additional_tokens=500)

    user = await get_active_user(async_db, "token-topup-user")
    assert user is not None
    assert user.reserved_tokens == 1_500
    assert handle.token_estimate == 1_500
    # The dollar hold is untouched by a delta of zero.
    assert user.reserved == Decimal("1.0")

    await refund_reservation(async_db, handle)

    await async_db.refresh(user)
    assert (user.reserved_tokens, user.reserved) == (0, Decimal(0))


@pytest.mark.asyncio
async def test_a_free_model_refusal_names_the_axis_that_gated_it(async_db: Any) -> None:
    """The dollar axis is not asked of a free request, so it cannot be the answer.

    A zero-priced model reserves no dollars, so the gate never tests that cap.
    Reporting it anyway named the one cap with room as the one that refused,
    which is worse than the unqualified word the axis naming replaced.
    """
    async_db.add(
        ModelPricing(model_key="openai:free-axis", input_price_per_million=0.0, output_price_per_million=0.0)
    )
    # Both caps are spent, so a paid request would legitimately report either.
    async_db.add(Budget(budget_id="both-spent", max_budget=10.0, token_limit=1_000))
    async_db.add(
        User(user_id="free-axis-user", budget_id="both-spent", spend=Decimal("10.0"), current_tokens=1_000)
    )
    await async_db.commit()

    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(
            async_db, "free-axis-user", 5.0, model="openai:free-axis", estimated_tokens=1
        )

    assert "token limit" in str(refusal.value.detail)


@pytest.mark.asyncio
async def test_a_token_top_up_is_clamped_before_any_hold_is_taken(async_db: Any) -> None:
    """The ceilings are held first, so a delta bounded after them is bounded too late.

    The estimate derives from a client-supplied output bound that nothing on the
    wire limits, and every hold is added to a BIGINT server-side: an unclamped
    delta answers with an overflow where a refusal was owed.
    """
    async_db.add(Budget(budget_id="clamped", max_budget=10.0))
    async_db.add(User(user_id="clamped-user", budget_id="clamped"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "clamped-user", 1.0, estimated_tokens=10)
    await increase_reservation(async_db, handle, Decimal("0"), additional_tokens=10**18)

    user = await get_active_user(async_db, "clamped-user")
    assert user is not None
    assert handle.token_estimate == 10 + MAX_COUNT_LIMIT
    assert user.reserved_tokens == 10 + MAX_COUNT_LIMIT
