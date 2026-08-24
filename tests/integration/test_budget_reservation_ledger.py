"""The reservation ledger: one row per in-flight hold, and what that row buys.

``users.reserved`` and ``scoped_budgets.reserved_spend`` were counters with no
identity behind them (mozilla-ai/otari#742), so two things the gateway is
supposed to guarantee were unreachable: a release could not tell whether it had
already run, and a leaked hold could be seen only in aggregate. These cover both,
plus the paths that deliberately write no row.

The counters themselves are covered by ``test_scoped_budgets.py`` and
``test_budget_race_condition.py``; nothing here re-asserts how a hold is taken.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.models.entities import Budget, BudgetReservation, BudgetReservationScope, ScopedBudget, User
from gateway.services import budget_reservation_ledger as ledger
from gateway.services.budget_service import (
    increase_reservation,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)

from .conftest import _to_async_url
from .test_scoped_budgets import Fixture, _build_tenancy, _counters, _scoped


@pytest_asyncio.fixture
async def tenancy(async_db: AsyncSession) -> Fixture:
    return await _build_tenancy(async_db, "ledger")


async def _rows(db: AsyncSession, user_id: str) -> list[BudgetReservation]:
    return list(
        (
            await db.execute(
                select(BudgetReservation)
                .where(BudgetReservation.user_id == user_id)
                .order_by(BudgetReservation.created_at)
            )
        )
        .scalars()
        .all()
    )


async def _lines(db: AsyncSession, reservation_id: str) -> list[BudgetReservationScope]:
    return list(
        (
            await db.execute(
                select(BudgetReservationScope).where(BudgetReservationScope.reservation_id == reservation_id)
            )
        )
        .scalars()
        .all()
    )


async def _status(db: AsyncSession, reservation_id: str) -> str:
    """Re-read a reservation's status from the database.

    ``try_terminate`` is a Core UPDATE with ``synchronize_session=False``, so a
    row this test already loaded still carries its pre-claim status in the
    identity map. Expire first, or the assertion tests the session rather than
    the database.
    """
    db.expire_all()
    return (await db.get_one(BudgetReservation, reservation_id)).status


async def _user(db: AsyncSession, user_id: str) -> User:
    await db.refresh(await db.get_one(User, user_id))
    return await db.get_one(User, user_id)


async def _with_budget(db: AsyncSession, tenancy: Fixture, *, max_budget: float | None = 20.0) -> None:
    """Give the billed user an enforced per-user budget."""
    budget_id = f"ledger-{uuid.uuid4()}"
    db.add(Budget(budget_id=budget_id, max_budget=max_budget))
    await db.flush()
    user = await db.get_one(User, tenancy.user_id)
    user.budget_id = budget_id
    await db.commit()


@pytest.mark.asyncio
async def test_a_hold_becomes_a_row_on_both_mechanisms(async_db: AsyncSession, tenancy: Fixture) -> None:
    """One reservation, one row, and one line per ceiling it holds against."""
    await _with_budget(async_db, tenancy)
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=10.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 2.0, scope=tenancy.scope())

    assert handle.reservation_id is not None
    rows = await _rows(async_db, tenancy.user_id)
    assert len(rows) == 1
    assert rows[0].id == handle.reservation_id
    assert rows[0].status == ledger.RESERVATION_ACTIVE
    assert rows[0].user_reserved is True
    assert rows[0].estimate == Decimal("2.000000")
    assert rows[0].expires_at > datetime.now(UTC)

    lines = await _lines(async_db, handle.reservation_id)
    assert [line.scoped_budget_id for line in lines] == [cap.id]
    assert lines[0].amount == Decimal("2.000000")


@pytest.mark.asyncio
async def test_reconcile_is_idempotent_by_reservation_identity(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A second reconcile records no second spend and returns no second hold.

    Without the row this passed silently in the worst possible direction: the
    release expression clamps at zero, so the double subtraction showed up as an
    under-count of live holds rather than as an error, weakening the very cap the
    reserve gate enforces.
    """
    await _with_budget(async_db, tenancy)
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=10.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 4.0, scope=tenancy.scope())
    await reconcile_reservation(async_db, handle, 1.0)
    await reconcile_reservation(async_db, handle, 1.0)

    user = await _user(async_db, tenancy.user_id)
    assert user.spend == Decimal("1.000000")
    assert user.reserved == Decimal("0.000000")
    current, reserved = await _counters(async_db, cap.id)
    assert current == pytest.approx(1.0)
    assert reserved == pytest.approx(0.0)

    rows = await _rows(async_db, tenancy.user_id)
    assert rows[0].status == ledger.RESERVATION_SETTLED


@pytest.mark.asyncio
async def test_refund_is_idempotent_and_records_no_spend(async_db: AsyncSession, tenancy: Fixture) -> None:
    """Two refunds for one request return the hold once and never write spend.

    ``release_reservation`` in ``_pipeline`` is reachable from roughly seven
    sites; only a ``raise`` after each has kept two of them from firing.
    """
    await _with_budget(async_db, tenancy)
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=10.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 3.0, scope=tenancy.scope())
    await refund_reservation(async_db, handle)
    await refund_reservation(async_db, handle)

    user = await _user(async_db, tenancy.user_id)
    assert user.spend == Decimal("0.000000")
    assert user.reserved == Decimal("0.000000")
    current, reserved = await _counters(async_db, cap.id)
    assert current == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)
    rows = await _rows(async_db, tenancy.user_id)
    assert rows[0].status == ledger.RESERVATION_RELEASED


@pytest.mark.asyncio
async def test_reconcile_after_a_refund_writes_nothing(async_db: AsyncSession, tenancy: Fixture) -> None:
    """The two settlement verbs race each other too, not just themselves."""
    await _with_budget(async_db, tenancy)
    handle = await reserve_budget(async_db, tenancy.user_id, 5.0)
    await refund_reservation(async_db, handle)
    await reconcile_reservation(async_db, handle, 5.0)

    user = await _user(async_db, tenancy.user_id)
    assert user.spend == Decimal("0.000000")
    assert user.reserved == Decimal("0.000000")


@pytest.mark.asyncio
async def test_a_second_hold_survives_the_first_being_reclaimed(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A leaked hold is reclaimed on its own, and a live one beside it is untouched.

    This is the guarantee a counter could not give: before the ledger the only
    visible fact was a total, so the leaked amount could be cleared only by the
    budget's next reset, which for a budget with no reset period never came.
    """
    await _with_budget(async_db, tenancy)
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=10.0)
    async_db.add(cap)
    await async_db.commit()

    leaked = await reserve_budget(async_db, tenancy.user_id, 3.0, scope=tenancy.scope())
    live = await reserve_budget(async_db, tenancy.user_id, 2.0, scope=tenancy.scope())

    # Age only the first one past its TTL.
    assert leaked.reservation_id is not None
    row = await async_db.get_one(BudgetReservation, leaked.reservation_id)
    row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await async_db.commit()

    reclaimed = await ledger.reclaim_expired_for_user(async_db, tenancy.user_id)
    assert reclaimed == 1

    user = await _user(async_db, tenancy.user_id)
    assert user.reserved == Decimal("2.000000")  # only the live hold remains
    _, reserved = await _counters(async_db, cap.id)
    assert reserved == pytest.approx(2.0)

    assert await _status(async_db, leaked.reservation_id) == ledger.RESERVATION_EXPIRED
    assert live.reservation_id is not None
    assert await _status(async_db, live.reservation_id) == ledger.RESERVATION_ACTIVE


@pytest.mark.asyncio
async def test_settling_a_reclaimed_hold_does_not_release_it_twice(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """The sweep and the request's own settlement contend; only one releases.

    The request that leaked may still be alive and finish after the sweep has
    given its hold back. Its reconcile must then record the spend it owes but not
    subtract the hold a second time.
    """
    await _with_budget(async_db, tenancy)
    handle = await reserve_budget(async_db, tenancy.user_id, 4.0)
    assert handle.reservation_id is not None
    row = await async_db.get_one(BudgetReservation, handle.reservation_id)
    row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await async_db.commit()

    assert await ledger.sweep_expired(async_db, batch_size=10) == 1
    user = await _user(async_db, tenancy.user_id)
    assert user.reserved == Decimal("0.000000")

    # The late reconcile records the spend it owes (users.spend is the sum of that
    # user's rows, and the request really did cost this) but releases nothing: the
    # sweep already gave the hold back.
    await reconcile_reservation(async_db, handle, 1.0)
    user = await _user(async_db, tenancy.user_id)
    assert user.spend == Decimal("1.000000")
    assert user.reserved == Decimal("0.000000")
    assert await _status(async_db, handle.reservation_id) == ledger.RESERVATION_SETTLED

    # And a second late settlement is still a no-op.
    await reconcile_reservation(async_db, handle, 1.0)
    user = await _user(async_db, tenancy.user_id)
    assert user.spend == Decimal("1.000000")
    assert user.reserved == Decimal("0.000000")


@pytest.mark.asyncio
async def test_two_concurrent_settlements_release_the_hold_once(
    async_db: AsyncSession, postgres_url: str, tenancy: Fixture
) -> None:
    """Six simultaneous reconciles of one reservation: exactly one lands.

    The sequential cases above are the shape this actually takes in production (a
    streaming callback and an error path both firing), but the claim has to hold
    under real contention too, which is the DoD line this issue was filed on.
    """
    await _with_budget(async_db, tenancy)
    handle = await reserve_budget(async_db, tenancy.user_id, 6.0)

    engine = create_async_engine(_to_async_url(postgres_url), pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def settle() -> None:
        async with factory() as session:
            await reconcile_reservation(session, handle, 1.5)

    try:
        await asyncio.gather(*(settle() for _ in range(6)))
    finally:
        await engine.dispose()

    user = await _user(async_db, tenancy.user_id)
    assert user.spend == Decimal("1.500000")
    assert user.reserved == Decimal("0.000000")


@pytest.mark.asyncio
async def test_a_top_up_grows_the_row_it_already_has(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A grown reservation stays one row.

    Two rows for one request would each carry their own TTL and could be
    reclaimed independently, handing back part of a hold that is still live.
    """
    await _with_budget(async_db, tenancy)
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=10.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 1.0, scope=tenancy.scope())
    await increase_reservation(async_db, handle, 2.0)

    rows = await _rows(async_db, tenancy.user_id)
    assert len(rows) == 1
    assert rows[0].estimate == Decimal("3.000000")
    assert handle.reservation_id is not None
    lines = await _lines(async_db, handle.reservation_id)
    assert lines[0].amount == Decimal("3.000000")

    # And the grown hold is released in full, not just the original estimate.
    await refund_reservation(async_db, handle)
    user = await _user(async_db, tenancy.user_id)
    assert user.reserved == Decimal("0.000000")
    _, reserved = await _counters(async_db, cap.id)
    assert reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_request_that_holds_nothing_writes_no_row(async_db: AsyncSession, tenancy: Fixture) -> None:
    """No budget and no ceiling means nothing to reclaim, so no row.

    The ledger is on the hot path of every billable request; a row for a hold
    that does not exist would be pure write amplification.
    """
    handle = await reserve_budget(async_db, tenancy.user_id, 1.0)

    assert handle.reservation_id is None
    assert await _rows(async_db, tenancy.user_id) == []


@pytest.mark.asyncio
async def test_a_budget_exempt_request_writes_no_row(async_db: AsyncSession, tenancy: Fixture) -> None:
    """An exempt key reserves nothing on either mechanism, so it ledgers nothing."""
    await _with_budget(async_db, tenancy)
    handle = await reserve_budget(async_db, tenancy.user_id, 1.0, counts_toward_budget=False)

    assert handle.reservation_id is None
    assert await _rows(async_db, tenancy.user_id) == []


@pytest.mark.asyncio
async def test_the_sweep_reclaims_across_users(async_db: AsyncSession) -> None:
    """The global sweep is what covers a user who never reserves again.

    ``reclaim_expired_for_user`` only fires when that user's next request
    reserves, so a user whose single request leaked would hold against their
    budget indefinitely.
    """
    first = await _build_tenancy(async_db, "sweep-one")
    second = await _build_tenancy(async_db, "sweep-two")
    for fixture in (first, second):
        await _with_budget(async_db, fixture)
        handle = await reserve_budget(async_db, fixture.user_id, 2.0)
        assert handle.reservation_id is not None
        row = await async_db.get_one(BudgetReservation, handle.reservation_id)
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await async_db.commit()

    assert await ledger.sweep_expired(async_db, batch_size=10) == 2
    for fixture in (first, second):
        assert (await _user(async_db, fixture.user_id)).reserved == Decimal("0.000000")


@pytest.mark.asyncio
async def test_an_orphan_scope_line_does_not_block_the_release(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A ceiling deleted mid-flight leaves a line the release skips.

    ``scoped_budget_id`` is deliberately FK-less for this: a delete must not have
    to cascade into live holds.
    """
    await _with_budget(async_db, tenancy)
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=10.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 2.0, scope=tenancy.scope())
    await async_db.delete(await async_db.get_one(ScopedBudget, cap.id))
    await async_db.commit()

    await refund_reservation(async_db, handle)
    assert (await _user(async_db, tenancy.user_id)).reserved == Decimal("0.000000")
