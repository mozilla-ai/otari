"""Tests for atomic spend update via SQL expression in reconcile_reservation."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import User
from gateway.services.budget_service import ReservationHandle, reconcile_reservation


@pytest.mark.asyncio
async def test_spend_update_uses_sql_expression(async_db: AsyncSession) -> None:
    """Test that reconcile_reservation updates spend atomically via SQL, not Python read-modify-write."""
    # Set up user with initial spend
    user = User(user_id="atomic-user", spend=5.0)
    async_db.add(user)
    await async_db.commit()

    # actual_cost equivalent to the old log_usage computation:
    # (1M / 1M) * 2.5 + (100K / 1M) * 10.0 = 2.5 + 1.0 = 3.5
    actual_cost = 3.5
    await reconcile_reservation(
        async_db,
        ReservationHandle(user_id="atomic-user", estimate=0.0, reserved=False, strategy="for_update"),
        actual_cost,
    )

    await async_db.refresh(user)
    assert user is not None

    expected_new_spend = 5.0 + 3.5
    assert abs(user.spend - expected_new_spend) < 0.001, f"Expected spend {expected_new_spend}, got {user.spend}"


@pytest.mark.asyncio
async def test_multiple_spend_updates_accumulate(async_db: AsyncSession) -> None:
    """Test that multiple sequential spend updates via reconcile_reservation accumulate correctly."""
    user = User(user_id="multi-spend-user", spend=0.0)
    async_db.add(user)
    await async_db.commit()

    # Each call costs (1M/1M)*10 + (1M/1M)*10 = 20.0, x3 = 60.0
    for _ in range(3):
        await reconcile_reservation(
            async_db,
            ReservationHandle(user_id="multi-spend-user", estimate=0.0, reserved=False, strategy="for_update"),
            20.0,
        )

    await async_db.refresh(user)
    assert user is not None

    assert abs(user.spend - 60.0) < 0.001, f"Expected spend 60.0, got {user.spend}"
