"""The reservation ledger: one row per in-flight budget hold.

``users.reserved`` and ``scoped_budgets.reserved_spend`` remain the counters the
budget gate reads, and this module does not change how a hold is taken. What it
adds is the identity behind a hold (mozilla-ai/otari#742), which is what the two
outstanding guarantees need:

* **Release is idempotent.** ``reserve_budget`` has roughly seven release sites
  and nothing but control flow (each one is followed by ``raise``) keeps two of
  them from firing for one request. A second release subtracts the hold twice,
  and because the release expression clamps at zero that surfaces not as an
  error but as an under-count of live holds, weakening the overspend guarantee
  the gate exists to provide. Here the ACTIVE -> terminal claim decides, so only
  the first release does the work.
* **A leaked hold is reclaimable individually.** #724 fixed a path where a
  release never ran; the residue of that class of bug used to be an amount that
  could be seen only in aggregate and cleared only by the budget's next reset
  (or never, for a budget with no reset period). A row gives it an owner, an age
  and a TTL.

**No row locks**, matching :mod:`gateway.services.scoped_budget_service`: the
sweep reads candidate rows without ``FOR UPDATE`` and lets the conditional
UPDATE in :func:`try_terminate` arbitrate. Two sweepers may read the same row,
but only one wins the transition and releases. That also keeps the chain
dialect-neutral, since SQLite has no ``SKIP LOCKED``.

The row is written **after** the holds it records. Of the two windows in which
the counter and the ledger can disagree, only that one is safe: a hold with no
row is the pre-existing leak this sweep bounds, while a row with no hold would
have the sweep hand back an amount nobody is holding and under-count the live
ones.

**The claim commits before the release**, which is what makes a race between two
finalizers resolve correctly, and it leaves one window this module does not
close: a process that dies between the two leaves a terminal row whose hold was
never returned, and no later sweep will look at it again. That is the same
per-step-commit trade-off the rest of this path already takes (see
``scoped_budget_service``'s "the price is that a partial reservation is
possible"), and it leaves such a hold exactly where the pre-ledger code left
every leaked hold. Closing it needs the claim and both releases in one
transaction, which ``scoped_budget_service.release`` committing for itself
currently prevents.

Standalone mode only. Hybrid mode reserves nothing locally, because the platform
holds against its own ledger.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import case, select, update

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import BudgetReservation, BudgetReservationScope, User
from gateway.services.scoped_budget_service import release as release_scoped

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from gateway.services.scoped_budget_service import ApplicableBudget

ZERO = Decimal(0)

# The lifecycle, as stored. Plain strings rather than a database enum so a new
# state needs no enum migration (the same reasoning as ``scoped_budgets.scope_type``).
RESERVATION_ACTIVE = "active"
RESERVATION_SETTLED = "settled"  # Actual recorded, hold released
RESERVATION_RELEASED = "released"  # Hold returned with no spend recorded
RESERVATION_EXPIRED = "expired"  # Reclaimed by the TTL sweep after leaking


def release_reserved_expression(estimate: Decimal) -> object:
    """Column expression that subtracts ``estimate`` from ``users.reserved``, clamped at 0.

    Uses CASE rather than GREATEST for SQLite compatibility. Both arms are
    ``Decimal`` so the CASE resolves as ``numeric``: a bare ``0.0`` in the clamp
    arm would make PostgreSQL type the whole expression ``double precision`` and
    round-trip the untouched amount through a binary float.

    Lives here rather than in :mod:`gateway.services.budget_service` because the
    reclaim path needs the same expression, and two copies of that reasoning
    would drift.
    """
    return case(
        (User.reserved - estimate < ZERO, ZERO),
        else_=User.reserved - estimate,
    )


async def record(
    db: AsyncSession,
    *,
    user_id: str,
    estimate: Decimal,
    user_reserved: bool,
    counts_toward_budget: bool,
    scoped_budgets: Sequence[ApplicableBudget],
    scoped_estimate: Decimal,
    ttl_seconds: int,
) -> str | None:
    """Write the row for a hold that has already been taken, returning its id.

    Returns ``None`` when the request holds nothing on either mechanism, which
    is the common case for a free model, a budget-exempt key or a user with no
    budget row: there is no hold to make reclaimable, so a row would be pure
    write amplification on the hot path.
    """
    if not user_reserved and not scoped_budgets:
        return None

    reservation_id = str(uuid.uuid4())
    db.add(
        BudgetReservation(
            id=reservation_id,
            user_id=user_id,
            estimate=estimate,
            user_reserved=user_reserved,
            counts_toward_budget=counts_toward_budget,
            status=RESERVATION_ACTIVE,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
        )
    )
    for applicable in scoped_budgets:
        db.add(
            BudgetReservationScope(
                reservation_id=reservation_id,
                scoped_budget_id=applicable.budget_id,
                amount=scoped_estimate,
            )
        )
    await db.commit()
    return reservation_id


async def grow(
    db: AsyncSession,
    reservation_id: str | None,
    *,
    user_delta: Decimal,
    scoped_delta: Decimal,
) -> None:
    """Fold a top-up into the row that already records this request's hold.

    ``increase_reservation`` grows the counters in place, so the ledger grows the
    same row rather than opening a second one: two rows for one request would
    each carry their own TTL and could be reclaimed independently, releasing part
    of a live hold.

    Guarded on ``status`` so a top-up cannot resurrect a row a concurrent sweep
    has already reclaimed; if it has, the delta the caller just took is left for
    that row's own reclaim to find, which is the same outcome as a hold taken
    without a ledger row at all.
    """
    if reservation_id is None:
        return
    if user_delta > ZERO:
        await db.execute(
            update(BudgetReservation)
            .where(
                BudgetReservation.id == reservation_id,
                BudgetReservation.status == RESERVATION_ACTIVE,
            )
            .values(estimate=BudgetReservation.estimate + user_delta)
            .execution_options(synchronize_session=False)
        )
    if scoped_delta > ZERO:
        await db.execute(
            update(BudgetReservationScope)
            .where(BudgetReservationScope.reservation_id == reservation_id)
            .values(amount=BudgetReservationScope.amount + scoped_delta)
            .execution_options(synchronize_session=False)
        )
    await db.commit()


async def try_terminate(db: AsyncSession, reservation_id: str | None, status: str) -> bool:
    """Claim the ACTIVE -> terminal transition, reporting whether this caller won.

    This is the whole point of the ledger. The ``WHERE status = ACTIVE`` guard is
    what makes reconcile, refund and the TTL sweep safe against each other: only
    the transaction whose UPDATE matches an active row goes on to touch the
    counters, so a hold is never given back twice.

    A ``None`` id means the request holds nothing worth ledgering (see
    :func:`record`), and the caller keeps its pre-ledger behavior.
    """
    if reservation_id is None:
        return True
    result = await db.execute(
        update(BudgetReservation)
        .where(
            BudgetReservation.id == reservation_id,
            BudgetReservation.status == RESERVATION_ACTIVE,
        )
        .values(status=status)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return bool(getattr(result, "rowcount", 0))


async def try_settle_reclaimed(db: AsyncSession, reservation_id: str | None) -> bool:
    """Claim EXPIRED -> SETTLED for a request that outlived its own hold.

    The sweep reclaims a hold on the assumption that its request is gone, and it
    is sometimes wrong: a request slower than the TTL finishes afterwards and
    still owes what it spent. Its hold is already back, so there is nothing to
    release, but ``users.spend`` is the sum of that user's rows and dropping the
    cost would leave the counter a 403 is decided against permanently short.

    Same CAS as :func:`try_terminate`, one state further along, so only the first
    late settlement records the spend and a second is still a no-op.
    """
    if reservation_id is None:
        return False
    result = await db.execute(
        update(BudgetReservation)
        .where(
            BudgetReservation.id == reservation_id,
            BudgetReservation.status == RESERVATION_EXPIRED,
        )
        .values(status=RESERVATION_SETTLED)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return bool(getattr(result, "rowcount", 0))


async def _release_holds(db: AsyncSession, reservation: BudgetReservation) -> None:
    """Return every hold one reclaimed row placed, on both mechanisms.

    Only ever called by the winner of :func:`try_terminate`, so it can subtract
    unconditionally.
    """
    lines = (
        (
            await db.execute(
                select(BudgetReservationScope.scoped_budget_id, BudgetReservationScope.amount).where(
                    BudgetReservationScope.reservation_id == reservation.id
                )
            )
        )
        .tuples()
        .all()
    )
    # Grouped by amount because ``scoped_budget_service.release`` takes one figure
    # for a set of ceilings. Today every line of a reservation carries the same
    # amount, so this is one call; it stays correct if that ever stops being true.
    by_amount: dict[Decimal, list[str]] = {}
    for scoped_budget_id, amount in lines:
        if amount > ZERO:
            by_amount.setdefault(amount, []).append(scoped_budget_id)
    for amount, budget_ids in by_amount.items():
        await release_scoped(db, budget_ids, amount)

    if reservation.user_reserved and reservation.estimate > ZERO:
        await db.execute(
            update(User)
            .where(User.user_id == reservation.user_id, User.deleted_at.is_(None))
            .values(reserved=release_reserved_expression(reservation.estimate))
            .execution_options(synchronize_session=False)
        )
        await db.commit()


async def _reclaim(db: AsyncSession, expired: Sequence[BudgetReservation]) -> int:
    """Expire and release each leaked hold, returning how many this call claimed."""
    reclaimed = 0
    for reservation in expired:
        if not await try_terminate(db, reservation.id, RESERVATION_EXPIRED):
            continue
        await _release_holds(db, reservation)
        reclaimed += 1
    return reclaimed


async def reclaim_expired_for_user(db: AsyncSession, user_id: str, *, limit: int = 20) -> int:
    """Reclaim this user's leaked holds, opportunistically, on the reserve path.

    The same idiom as the budget period roll: work the hot path can do cheaply
    for itself, so a user whose earlier request leaked is not refused by its
    residue. Bounded by ``limit`` so a pathological backlog cannot turn one
    request into an unbounded scan; the scheduled sweep drains the rest.
    """
    expired = (
        (
            await db.execute(
                select(BudgetReservation)
                .where(
                    BudgetReservation.user_id == user_id,
                    BudgetReservation.status == RESERVATION_ACTIVE,
                    BudgetReservation.expires_at < datetime.now(UTC),
                )
                .order_by(BudgetReservation.expires_at)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    if not expired:
        return 0
    reclaimed = await _reclaim(db, expired)
    if reclaimed:
        logger.info("Reclaimed %d leaked budget reservation(s) for user %s", reclaimed, user_id)
    return reclaimed


async def sweep_expired(db: AsyncSession, *, batch_size: int) -> int:
    """Reclaim leaked holds across every user, for the scheduled sweep.

    :func:`reclaim_expired_for_user` only runs when *that* user next reserves, so
    a user whose lone request leaked would hold against their budget until its
    next reset. This drains those. Returns how many it reclaimed; a caller
    working through a backlog loops while the count equals ``batch_size``.
    """
    expired = (
        (
            await db.execute(
                select(BudgetReservation)
                .where(
                    BudgetReservation.status == RESERVATION_ACTIVE,
                    BudgetReservation.expires_at < datetime.now(UTC),
                )
                .order_by(BudgetReservation.expires_at)
                .limit(batch_size)
            )
        )
        .scalars()
        .all()
    )
    if not expired:
        return 0
    reclaimed = await _reclaim(db, expired)
    if reclaimed:
        logger.info("Reclaimed %d leaked budget reservation(s) via the global sweep", reclaimed)
    return reclaimed


# How many batches one tick will drain before yielding. A backlog larger than
# this is left for the next tick rather than letting one pass run unbounded and
# hold a session open behind it.
_MAX_SWEEP_PASSES = 10


async def run_reservation_sweeper(interval: float, *, batch_size: int) -> None:
    """Reclaim leaked holds on a timer, forever. Cancelled at shutdown.

    :func:`reclaim_expired_for_user` only fires when *that* user next reserves,
    which is enough for a busy user and useless for the case that matters most: a
    user whose one request leaked and who then makes no more. That hold would sit
    against their budget until its next reset, or forever without one.

    Every error is swallowed and retried on the next tick, matching the other
    lifespan refreshers. A database blip must not kill the sweeper, because
    nothing would restart it and the worker would stop reclaiming for as long as
    it stayed up.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                for _ in range(_MAX_SWEEP_PASSES):
                    if await sweep_expired(db, batch_size=batch_size) < batch_size:
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Budget reservation sweep failed; retrying in %ss",
                interval,
                exc_info=True,
            )


__all__ = [
    "RESERVATION_ACTIVE",
    "RESERVATION_EXPIRED",
    "RESERVATION_RELEASED",
    "RESERVATION_SETTLED",
    "grow",
    "reclaim_expired_for_user",
    "record",
    "run_reservation_sweeper",
    "release_reserved_expression",
    "sweep_expired",
    "try_settle_reclaimed",
    "try_terminate",
]
