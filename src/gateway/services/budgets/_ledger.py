"""The reservation ledger holds one row per in-flight budget hold.

A row gives a hold an identity, an age and a time to live (TTL).
Release is idempotent, because only the first claim of an active row does the work.
The sweep reclaims a leaked hold on its own.
The sweep takes no row lock, and the conditional UPDATE in :func:`try_terminate` decides between two sweepers.
A row is written after the holds it records, so the sweep never releases an amount that nobody holds.
A claim and the release it authorizes commit in one transaction, so a terminal row never has an outstanding hold.
This module runs in standalone mode only, because in hybrid mode the platform holds against its own ledger.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import case, delete, select, update
from sqlalchemy.orm import Mapped

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.budgets import (
    RESERVATION_ACTIVE,
    RESERVATION_EXPIRED,
    RESERVATION_RELEASED,
    RESERVATION_SETTLED,
    BudgetReservation,
    BudgetReservationScope,
    ReservationStatus,
)
from gateway.models.users import User
from gateway.services.budgets._scoped_enforcement import release as release_scoped

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from gateway.services.budgets._scoped_enforcement import ApplicableBudget

ZERO = Decimal(0)

# The three a row can rest in. Written out as a set the retention query can ask
# for by equality: ``status != ACTIVE`` reads the same but is an inequality on
# the leading column of ``ix_budget_reservations_status_expires_at``, which the
# planner cannot use, so it degenerated into a full scan of a table that gains a
# row per billable request.
TERMINAL_STATUSES = (RESERVATION_SETTLED, RESERVATION_RELEASED, RESERVATION_EXPIRED)


def release_reserved_expression(estimate: Decimal) -> object:
    """Return the column expression that subtracts ``estimate`` from ``users.reserved``, clamped at 0.

    The expression uses CASE rather than GREATEST for SQLite compatibility.
    Both arms are ``Decimal``, because a float arm makes PostgreSQL type the expression ``double precision``.
    """
    return case(
        (User.reserved - estimate < ZERO, ZERO),
        else_=User.reserved - estimate,
    )


def release_reserved_count_expression(column: Mapped[int], amount: int) -> object:
    """The same clamp for one of the token or request holds, whose arms are integers."""
    return case((column - amount < 0, 0), else_=column - amount)


async def record(
    db: AsyncSession,
    *,
    user_id: str,
    estimate: Decimal,
    user_reserved: bool,
    scoped_budgets: Sequence[ApplicableBudget],
    scoped_estimate: Decimal,
    token_estimate: int = 0,
    scoped_token_estimate: int = 0,
    request_estimate: int = 0,
    ttl_seconds: int,
) -> str | None:
    """Write the row for a hold that has already been taken, returning its id.

    Returns ``None`` when the request holds nothing on either mechanism, which
    is the common case for a free model, a budget-exempt key or a user with no
    budget row: there is no hold to make reclaimable, so a row would be pure
    write amplification on the hot path.

    Every amount is per leg, as the caller knows them: ``token_estimate`` is what
    the per-user leg holds and ``scoped_token_estimate`` what each ceiling does,
    and the two diverge whenever a top-up grows one and not the other. The request
    count takes one figure because it never grows, so both legs hold what they
    held at admission; the row records it only when that leg holds at all.
    """
    if not user_reserved and not scoped_budgets:
        return None

    reservation_id = str(uuid.uuid4())
    db.add(
        BudgetReservation(
            id=reservation_id,
            user_id=user_id,
            estimate=estimate,
            token_estimate=token_estimate,
            request_estimate=request_estimate if user_reserved else 0,
            user_reserved=user_reserved,
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
                token_amount=scoped_token_estimate,
                request_amount=request_estimate,
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
    token_delta: int = 0,
    scoped_token_delta: int = 0,
) -> bool:
    """Fold a top-up into the row that already records this request's hold.

    A top-up grows the dollar and token holds; it never grows the request one,
    because the top-up belongs to a request the ceiling has already counted. Each
    token delta is the one its own leg took: the ceilings grow whenever the
    request has any, and the per-user leg only when it has a budget to grow.

    ``increase_reservation`` grows the counters in place, so the ledger grows the
    same row rather than opening a second one: two rows for one request would
    each carry their own TTL and could be reclaimed independently, releasing part
    of a live hold.

    Reports whether the row was still active, and **both** legs are guarded on
    that, not just the per-user one. A row the sweep already reclaimed will never
    be looked at again, so a delta folded into it is held on the counters by
    something nothing will ever release; the caller has to give that delta back
    rather than merely decline to record it. True when there is no row at all,
    which is the unledgered case and keeps the pre-ledger behavior.
    """
    if reservation_id is None:
        return True
    result = await db.execute(
        update(BudgetReservation)
        .where(
            BudgetReservation.id == reservation_id,
            BudgetReservation.status == RESERVATION_ACTIVE,
        )
        .values(
            estimate=BudgetReservation.estimate + user_delta,
            # Gated on the row's own record of whether the per-user leg holds, not
            # on the dollar delta: a top-up can grow tokens alone (a model priced
            # at zero, an expanded prompt), and reading ``user_delta > 0`` as "the
            # user leg grew" would drop exactly that hold.
            token_estimate=BudgetReservation.token_estimate
            + case((BudgetReservation.user_reserved, token_delta), else_=0),
        )
        .execution_options(synchronize_session=False)
    )
    if not getattr(result, "rowcount", 0):
        # A rollback here would expire every ORM instance in the session and force sync IO on the caller's next read.
        return False
    if scoped_delta > ZERO or scoped_token_delta > 0:
        # In the same transaction as the guarded UPDATE above, so the lines cannot
        # grow against a row that turned terminal between the two statements. Each
        # axis grows only by what it was actually given, so a token-only top-up
        # leaves the dollar amount alone and a dollar-only one leaves the tokens.
        await db.execute(
            update(BudgetReservationScope)
            .where(BudgetReservationScope.reservation_id == reservation_id)
            .values(
                amount=BudgetReservationScope.amount + scoped_delta,
                token_amount=BudgetReservationScope.token_amount + scoped_token_delta,
            )
            .execution_options(synchronize_session=False)
        )
    await db.commit()
    return True


async def try_terminate(db: AsyncSession, reservation_id: str | None, status: ReservationStatus) -> bool:
    """Claim the ACTIVE -> terminal transition, reporting whether this caller won.

    This is the whole point of the ledger. The ``WHERE status = ACTIVE`` guard is
    what makes reconcile, refund and the TTL sweep safe against each other: only
    the transaction whose UPDATE matches an active row goes on to touch the
    counters, so a hold is never given back twice.

    A ``None`` id means the request holds nothing worth ledgering (see
    :func:`record`), and the caller keeps its pre-ledger behavior.

    **Never commits.** The claim stays in the caller's transaction so the status
    and the release it authorizes land together; committing here first would leave
    a terminal row with a live hold whenever the release then failed, and no later
    sweep revisits a terminal row. There is deliberately no ``commit`` switch,
    because a call site taking a permissive default is exactly how that window
    would come back. A concurrent finalizer blocks on this row's lock until the
    caller's transaction ends, then re-reads the now-terminal status and gets
    ``rowcount`` 0, which is the answer it would have got anyway, one wait later.
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
    return bool(getattr(result, "rowcount", 0))


async def try_settle_reclaimed(db: AsyncSession, reservation_id: str | None) -> bool:
    """Claim EXPIRED -> SETTLED for a request that outlived its own hold.

    The sweep reclaims a hold on the assumption that its request is gone, and it
    is sometimes wrong: a request slower than the TTL finishes afterwards and
    still owes what it spent. Its hold is already back, so there is nothing to
    release, but ``users.spend`` is the sum of that user's rows and dropping the
    cost would leave the counter a 403 is decided against permanently short.

    Same CAS as :func:`try_terminate`, one state further along, so only the first
    late settlement records the spend and a second is still a no-op. Never commits,
    for the reason given there.
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
    return bool(getattr(result, "rowcount", 0))


async def _release_holds(db: AsyncSession, reservation: BudgetReservation) -> None:
    """Return every hold one reclaimed row placed, on both mechanisms.

    Only ever called by the winner of :func:`try_terminate`, so it can subtract
    unconditionally. Does not commit: the caller commits the claim and these
    releases together, so a row can never end up terminal with its hold still
    outstanding.
    """
    lines = (
        (
            await db.execute(
                select(
                    BudgetReservationScope.scoped_budget_id,
                    BudgetReservationScope.amount,
                    BudgetReservationScope.token_amount,
                    BudgetReservationScope.request_amount,
                ).where(BudgetReservationScope.reservation_id == reservation.id)
            )
        )
        .tuples()
        .all()
    )
    # ``release_scoped`` takes one figure per axis for a set of ceilings, so the lines group by their amounts.
    by_amounts: dict[tuple[Decimal, int, int], list[str]] = {}
    for scoped_budget_id, amount, token_amount, request_amount in lines:
        if amount > ZERO or token_amount > 0 or request_amount > 0:
            by_amounts.setdefault((amount, token_amount, request_amount), []).append(scoped_budget_id)
    for (amount, token_amount, request_amount), budget_ids in by_amounts.items():
        await release_scoped(
            db,
            budget_ids,
            amount,
            tokens=token_amount,
            requests=request_amount,
            commit=False,
        )

    if reservation.user_reserved:
        values: dict[str, object] = {}
        if reservation.estimate > ZERO:
            values["reserved"] = release_reserved_expression(reservation.estimate)
        if reservation.token_estimate > 0:
            values["reserved_tokens"] = release_reserved_count_expression(
                User.reserved_tokens, reservation.token_estimate
            )
        if reservation.request_estimate > 0:
            values["reserved_requests"] = release_reserved_count_expression(
                User.reserved_requests, reservation.request_estimate
            )
        if values:
            await db.execute(
                update(User)
                .where(User.user_id == reservation.user_id, User.deleted_at.is_(None))
                .values(**values)
                .execution_options(synchronize_session=False)
            )


async def _reclaim(db: AsyncSession, expired: Sequence[BudgetReservation]) -> int:
    """Expire and release each leaked hold, returning how many this call claimed."""
    reclaimed = 0
    for reservation in expired:
        # Claim and release in one transaction. Committing the claim first would
        # mean a failure in the release left a row terminal with its hold still
        # held, and no later sweep would look at that row again.
        if not await try_terminate(db, reservation.id, RESERVATION_EXPIRED):
            # This commits the empty transaction, because a rollback would expire the rows this loop iterates over.
            await db.commit()
            continue
        await _release_holds(db, reservation)
        await db.commit()
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
    a user whose lone request leaked would hold against their budget with nothing
    ever releasing it. This drains those. Returns how many it reclaimed; a caller
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


async def prune_terminal(db: AsyncSession, *, older_than_sec: int, batch_size: int) -> int:
    """Delete settled, released and expired rows past the retention window.

    The table gains a row per billable request and nothing else ever removes one,
    so without this it grows without bound. A terminal row's job is finished:
    what a request actually cost lives in ``usage_logs``, which is the durable
    accounting record, and this row only ever existed to make the hold
    reclaimable while it was in flight.

    Deletes by id rather than with one predicate DELETE so the statement stays
    bounded, which is what keeps a first run against a long-lived deployment from
    taking a lock over most of the table. The lines cascade.

    Ages rows by ``expires_at`` rather than by ``updated_at``, and asks for the
    terminal statuses by equality, so the whole predicate rides
    ``ix_budget_reservations_status_expires_at``. ``updated_at`` is the more
    literal answer to "when did this go terminal" and would need a third index to
    ask for: ``.limit()`` bounds the result and not the scan, so without one the
    steady state is the worst case, a full scan every tick to return nothing.
    ``expires_at`` is the row's creation plus the TTL, so over a retention window
    measured in days the two differ by minutes.
    """
    if older_than_sec <= 0:
        return 0
    cutoff = datetime.now(UTC) - timedelta(seconds=older_than_sec)
    ids = (
        (
            await db.execute(
                select(BudgetReservation.id)
                .where(
                    BudgetReservation.status.in_(TERMINAL_STATUSES),
                    BudgetReservation.expires_at < cutoff,
                )
                .limit(batch_size)
            )
        )
        .scalars()
        .all()
    )
    if not ids:
        return 0
    await db.execute(
        delete(BudgetReservation)
        .where(BudgetReservation.id.in_(list(ids)))
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return len(ids)


# How many batches one tick will drain before yielding. A backlog larger than
# this is left for the next tick rather than letting one pass run unbounded and
# hold a session open behind it.
_MAX_SWEEP_PASSES = 10


async def run_reservation_sweeper(interval: float, *, batch_size: int, retention_sec: int) -> None:
    """Reclaim leaked holds on a timer, forever. Cancelled at shutdown.

    :func:`reclaim_expired_for_user` only fires when *that* user next reserves,
    which is enough for a busy user and useless for the case that matters most: a
    user whose one request leaked and who then makes no more. That hold would sit
    against their budget with nothing ever releasing it.

    Also prunes terminal rows past ``retention_sec``, since nothing else deletes
    one and the table gains a row per billable request.

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
                # Retention runs on the same tick rather than a timer of its own:
                # it is the same bounded, best-effort maintenance against the same
                # table, and a second lifespan task would double the machinery for
                # work that is never urgent.
                for _ in range(_MAX_SWEEP_PASSES):
                    if await prune_terminal(db, older_than_sec=retention_sec, batch_size=batch_size) < batch_size:
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
    "grow",
    "reclaim_expired_for_user",
    "prune_terminal",
    "record",
    "run_reservation_sweeper",
    "release_reserved_expression",
    "TERMINAL_STATUSES",
    "sweep_expired",
    "try_settle_reclaimed",
    "try_terminate",
]
