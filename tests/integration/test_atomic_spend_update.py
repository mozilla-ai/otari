"""Tests for atomic spend update via SQL expression in reconcile_reservation.

Compared exactly rather than within a tolerance: ``users.spend`` is
``NUMERIC(18, 6)`` as of mozilla-ai/otari#691, so the sum of the settled amounts
is the value, and a tolerance would keep passing on a counter that had gone back
to binary floating point.
"""

from decimal import Decimal

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
    # (1M / 1M) * 2.5 + (100K / 1M) * 10.0 = 2.5 + 1.0 = 3.5, plus a
    # micro-dollar tail so the assertion below is about the column's full scale.
    actual_cost = Decimal("3.500001")
    await reconcile_reservation(
        async_db,
        ReservationHandle(user_id="atomic-user", estimate=Decimal(0), reserved=False, strategy="for_update"),
        actual_cost,
    )

    await async_db.refresh(user)
    assert user is not None

    assert user.spend == Decimal("8.500001")


@pytest.mark.asyncio
async def test_multiple_spend_updates_accumulate(async_db: AsyncSession) -> None:
    """Test that multiple sequential spend updates via reconcile_reservation accumulate correctly."""
    user = User(user_id="multi-spend-user", spend=0.0)
    async_db.add(user)
    await async_db.commit()

    # Each call costs (1M/1M)*10 + (1M/1M)*10 = 20.0, plus a micro-dollar tail so
    # three of them accumulate at the column's full scale rather than at a value
    # a float would have carried unharmed. A float caller still reaches this
    # function; ``test_budget_race_condition.py`` is where that path is covered.
    for _ in range(3):
        await reconcile_reservation(
            async_db,
            ReservationHandle(user_id="multi-spend-user", estimate=Decimal(0), reserved=False, strategy="for_update"),
            Decimal("20.000001"),
        )

    await async_db.refresh(user)
    assert user is not None

    assert user.spend == Decimal("60.000003")
