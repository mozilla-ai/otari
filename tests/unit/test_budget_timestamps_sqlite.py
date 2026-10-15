"""The budget gate's timestamps stay UTC-aware on SQLite, the OSS edition's default engine.

SQLite has no timestamp type, so a plain ``DateTime(timezone=True)`` column
reads back naive, and comparing it to the aware ``now`` the gate computes raises
``TypeError``. Every integration test runs on PostgreSQL, which hands back an
aware value, so the engine where this fails had no coverage: a user on any budget
with a reset period got a 500 on every request.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TypeVar

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.budgets import Budget, BudgetReservation, BudgetResetLog, ScopedBudget
from gateway.models.users import User
from gateway.services.budgets import reserve_budget

T = TypeVar("T")


def _run(scenario: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Create the schema on a fresh in-memory SQLite database and run one scenario."""

    async def main() -> T:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                return await scenario(session)
        finally:
            await engine.dispose()

    return asyncio.run(main())


async def _seed_user(session: AsyncSession, *, next_reset_at: datetime, alignment: str | None) -> None:
    session.add(
        Budget(
            budget_id="daily",
            max_budget=Decimal(10),
            reset_alignment=alignment,
            budget_duration_sec=None if alignment else 3600,
        )
    )
    session.add(
        User(
            user_id="alice",
            budget_id="daily",
            spend=Decimal(3),
            budget_started_at=next_reset_at - timedelta(days=1),
            next_budget_reset_at=next_reset_at,
        )
    )
    await session.commit()
    session.expunge_all()


@pytest.mark.parametrize("alignment", ["calendar_day", None])
def test_the_gate_admits_a_user_whose_period_has_not_ended(alignment: str | None) -> None:
    async def scenario(session: AsyncSession) -> User:
        await _seed_user(session, next_reset_at=datetime.now(UTC) + timedelta(hours=1), alignment=alignment)
        await reserve_budget(session, "alice", Decimal("0.01"))
        session.expunge_all()
        return (await session.execute(select(User))).scalar_one()

    user = _run(scenario)

    assert user.spend == Decimal(3), "a period that has not ended keeps its spend"


@pytest.mark.parametrize("alignment", ["calendar_day", None])
def test_the_gate_rolls_a_period_that_has_ended(alignment: str | None) -> None:
    async def scenario(session: AsyncSession) -> tuple[User, BudgetResetLog]:
        await _seed_user(session, next_reset_at=datetime.now(UTC) - timedelta(minutes=1), alignment=alignment)
        await reserve_budget(session, "alice", Decimal("0.01"))
        session.expunge_all()
        user = (await session.execute(select(User))).scalar_one()
        log = (await session.execute(select(BudgetResetLog))).scalar_one()
        return user, log

    user, log = _run(scenario)

    assert user.spend == 0
    assert user.next_budget_reset_at is not None
    assert user.next_budget_reset_at > datetime.now(UTC)
    assert log.reset_at.utcoffset() == timedelta(0)


def test_the_gate_timestamps_read_back_utc_aware() -> None:
    """Every column the gate compares against ``now``, read off the database rather than the identity map."""
    moment = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)

    async def scenario(session: AsyncSession) -> list[datetime | None]:
        await _seed_user(session, next_reset_at=moment, alignment="calendar_day")
        session.add(BudgetResetLog(budget_id="daily", user_id="alice", previous_spend=Decimal(0), next_reset_at=moment))
        session.add(BudgetReservation(user_id="alice", expires_at=moment))
        session.add(
            ScopedBudget(
                scope_type="workspace", scope_id="w", budget_id="daily", period_start=moment, period_end=moment
            )
        )
        await session.commit()
        session.expunge_all()
        user = (await session.execute(select(User))).scalar_one()
        log = (await session.execute(select(BudgetResetLog))).scalar_one()
        reservation = (await session.execute(select(BudgetReservation))).scalar_one()
        ceiling = (await session.execute(select(ScopedBudget))).scalar_one()
        return [
            user.budget_started_at,
            user.next_budget_reset_at,
            log.reset_at,
            log.next_reset_at,
            reservation.expires_at,
            ceiling.period_start,
            ceiling.period_end,
        ]

    for value in _run(scenario):
        assert value is not None
        assert value.utcoffset() == timedelta(0), value
