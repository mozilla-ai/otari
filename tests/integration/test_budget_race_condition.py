"""Tests for budget enforcement behavior."""

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from gateway.models.entities import Budget, ModelPricing, User
from gateway.repositories.users_repository import get_active_user
from gateway.services.budget_service import (
    estimate_cost,
    increase_reservation,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
    validate_user_budget,
)


@pytest.mark.asyncio
async def test_validate_user_budget_reads_user_without_locking(
    async_db: Any,
) -> None:
    """validate_user_budget should allow under-limit users without lock contention."""
    budget = Budget(
        budget_id="race-budget",
        max_budget=10.0,
    )
    async_db.add(budget)

    user = User(
        user_id="race-user",
        spend=9.99,
        budget_id="race-budget",
    )
    async_db.add(user)
    await async_db.commit()

    with patch(
        "gateway.services.budget_service.get_active_user",
        wraps=get_active_user,
    ) as mock_get_active_user:
        result = await validate_user_budget(async_db, "race-user", strategy="cas")

    assert result.user_id == "race-user"
    assert mock_get_active_user.call_args.kwargs.get("for_update", False) is False


@pytest.mark.asyncio
async def test_budget_check_rejects_at_limit(
    async_db: Any,
) -> None:
    """Test that a user at or over budget limit is rejected."""
    from fastapi import HTTPException

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
        await validate_user_budget(async_db, "full-user")
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
    assert user.reserved == pytest.approx(0.8)
    assert user.spend == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_reconcile_records_actual_cost_and_releases_hold(async_db: Any) -> None:
    """reconcile_reservation adds the actual cost to spend and frees the estimate."""
    async_db.add(Budget(budget_id="rec-budget", max_budget=100.0))
    async_db.add(User(user_id="rec-user", spend=10.0, budget_id="rec-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "rec-user", 5.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "rec-user")
    assert user is not None and user.reserved == pytest.approx(5.0)

    await reconcile_reservation(async_db, handle, 3.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "rec-user")
    assert user is not None
    assert user.spend == pytest.approx(13.0)  # 10 + actual 3 (not the 5 estimate)
    assert user.reserved == pytest.approx(0.0)


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
    assert user.spend == pytest.approx(10.0)  # unchanged
    assert user.reserved == pytest.approx(0.0)


def test_estimate_cost_clamps_negative_output_tokens() -> None:
    """A hostile negative max_output_tokens must not produce a negative estimate."""
    pricing = ModelPricing(model_key="openai:gpt-4o", input_price_per_million=2.5, output_price_per_million=10.0)
    est = estimate_cost(pricing, prompt_chars=400, max_output_tokens=-1_000_000, default_output_tokens=1024)
    # Output term clamped to 0 → only the prompt contributes; never negative.
    assert est >= 0.0
    assert est == pytest.approx((400 / 4 / 1_000_000) * 2.5)


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
    assert user.reserved == pytest.approx(4.0)  # unchanged — negative clamped to 0
    assert user.spend == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_increase_reservation_grows_hold_and_folds_handle(async_db: Any) -> None:
    """A fitting top-up adds to `reserved` and folds the delta into the handle."""
    async_db.add(Budget(budget_id="inc-budget", max_budget=100.0))
    async_db.add(User(user_id="inc-user", spend=10.0, budget_id="inc-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "inc-user", 5.0)
    await increase_reservation(async_db, handle, 7.0)

    # Delta folded into the handle so the single reconcile/refund covers it all.
    assert handle.estimate == pytest.approx(12.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "inc-user")
    assert user is not None and user.reserved == pytest.approx(12.0)

    # Reconcile releases the full held amount.
    await reconcile_reservation(async_db, handle, 9.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "inc-user")
    assert user is not None
    assert user.reserved == pytest.approx(0.0)
    assert user.spend == pytest.approx(19.0)  # 10 + actual 9


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
    assert user.reserved == pytest.approx(1.0)
    assert user.spend == pytest.approx(8.0)

    await refund_reservation(async_db, handle)
    async_db.expire_all()
    user = await get_active_user(async_db, "incr-user")
    assert user is not None and user.reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_increase_reservation_noop_for_nonpositive_delta(async_db: Any) -> None:
    """A zero/negative delta leaves the reservation untouched."""
    async_db.add(Budget(budget_id="incn-budget", max_budget=100.0))
    async_db.add(User(user_id="incn-user", spend=0.0, budget_id="incn-budget"))
    await async_db.commit()

    handle = await reserve_budget(async_db, "incn-user", 5.0)
    await increase_reservation(async_db, handle, 0.0)
    await increase_reservation(async_db, handle, -3.0)

    assert handle.estimate == pytest.approx(5.0)
    async_db.expire_all()
    user = await get_active_user(async_db, "incn-user")
    assert user is not None and user.reserved == pytest.approx(5.0)


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
    assert user.spend == pytest.approx(10.0)  # not reduced by the negative cost
    assert user.reserved == pytest.approx(0.0)  # hold released
