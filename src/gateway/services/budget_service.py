from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.metered_pricing import estimate_metered_cost
from gateway.log_config import logger
from gateway.metrics import record_budget_exceeded
from gateway.models.entities import Budget, BudgetResetLog, ModelPricing, User
from gateway.models.money import to_usd
from gateway.repositories.users_repository import get_active_user
from gateway.services import budget_reservation_ledger as ledger
from gateway.services.budget_periods import budget_window
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import provider_key
from gateway.services.scoped_budget_service import (
    ApplicableBudget,
    BudgetScopeRequest,
    applicable_budgets,
)
from gateway.services.scoped_budget_service import release as release_scoped
from gateway.services.scoped_budget_service import reserve as reserve_scoped
from gateway.services.scoped_budget_service import settle as settle_scoped
from gateway.types.budget_state import BudgetState

# Every counter in this module is a ``NUMERIC(18, 6)`` column, so the constants
# the SQL is built from are ``Decimal`` too. A bare ``0.0`` in a CASE arm or a
# ``.values()`` would make PostgreSQL resolve the whole expression as double
# precision and hand a binary-rounded amount back to an exact column.
ZERO = Decimal(0)

# Fallback when a caller does not thread the deployment's
# ``budget_reservation_ttl_sec`` through. Only the reserve site in ``_pipeline``
# has the config object to hand; the batch, search and pass-through sites take
# this. Fifteen minutes is well past the slowest request any of them serves, and
# reclaiming a live hold is the one failure the TTL must not have.
DEFAULT_RESERVATION_TTL_SEC = 900


async def _cas_reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> User:
    # Both cadences, through the one derivation. Reading only
    # ``budget_duration_sec`` here left a calendar-aligned budget with a null
    # next reset, and a null next reset never fires, so the row never refilled.
    window = budget_window(now, budget)
    started_at, next_reset_at = window if window is not None else (now, None)

    result = await db.execute(
        update(User)
        .where(
            User.user_id == user.user_id,
            User.deleted_at.is_(None),
            User.next_budget_reset_at.is_not(None),
            User.next_budget_reset_at <= now,
        )
        .values(
            spend=ZERO,
            # The window's start, not ``now``: an aligned budget rolled late
            # belongs to the period it is in, not to the moment it was noticed.
            budget_started_at=started_at,
            next_budget_reset_at=next_reset_at,
        )
        .execution_options(synchronize_session=False)
    )

    rowcount = getattr(result, "rowcount", 0)
    if rowcount and rowcount > 0:
        # Captured before commit: rollback() expires ORM instances, so reading
        # user.user_id in the error path would attempt sync IO in the async
        # session (MissingGreenlet), masking the original commit error.
        user_id_str = user.user_id
        reset_log = BudgetResetLog(
            user_id=user_id_str,
            budget_id=budget.budget_id,
            previous_spend=user.spend,
            reset_at=now,
            next_reset_at=next_reset_at,
        )
        db.add(reset_log)
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for user '%s': %s", user_id_str, e)
            raise
        refreshed = await get_active_user(db, user_id_str)
        return refreshed or user

    await db.rollback()
    return user


async def _get_budget(db: AsyncSession, budget_id: str) -> Budget | None:
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    return result.scalar_one_or_none()


async def get_budget_state(db: AsyncSession, user_id: str) -> BudgetState:
    """Read what a routing policy's budget conditions need, in one round trip.

    ``used_pct`` and ``remaining_usd`` are computed from ``spend + reserved``
    against ``max_budget``, which is the same committed total the budget gate
    enforces (:func:`reserve_budget`). Using bare ``spend`` instead would let a
    tier-down rule read a smaller number than the gate does and fire late.

    Both fields are ``None`` when the percentage is undefined: no user row, no
    budget attached, or an unlimited budget (``max_budget is None``). An undefined
    value never matches a condition, so the policy falls through to its default.
    """
    row = (
        await db.execute(
            select(User.spend, User.reserved, Budget.max_budget)
            .outerjoin(Budget, User.budget_id == Budget.budget_id)
            .where(User.user_id == user_id, User.deleted_at.is_(None))
        )
    ).one_or_none()
    if row is None:
        return BudgetState()
    spend, reserved, max_budget = row
    if max_budget is None or max_budget <= 0:
        return BudgetState()
    # Narrowed deliberately: ``BudgetState`` feeds a routing policy's percentage
    # and headroom conditions, which are thresholds rather than accounting, and
    # the compiler that reads them is float throughout.
    committed = float(spend or 0) + float(reserved or 0)
    return BudgetState(
        used_pct=committed / float(max_budget) * 100.0,
        remaining_usd=max(0.0, float(max_budget) - committed),
    )


async def _is_model_free(
    db: AsyncSession,
    model: str,
    *,
    pricing_provider: str | None = None,
    organization_id: uuid.UUID | None = None,
) -> bool:
    """Check if a model is free (both input and output prices are 0).

    Args:
        db: Database session
        model: Model identifier (e.g., "provider/model" or "model")
        pricing_provider: Resolved provider instance, when ``model`` is already
            the bare model name.
        organization_id: Whose rates decide it. A model the deployment prices at
            zero is not free to an organization that overrode it, and one the
            deployment charges for is free to an organization that overrode it to
            zero, so this has to be the same resolution the request is billed by.

    Returns:
        True if the model is free, False otherwise or if pricing not found

    """
    try:
        if pricing_provider is None:
            provider, model_name = AnyLLM.split_model_provider(model)
            pricing_provider = provider_key(provider) or None
        else:
            model_name = model
        pricing = await find_model_pricing(db, pricing_provider, model_name, organization_id=organization_id)
        if pricing:
            return pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0
    except (AnyLLMError, ValueError, SQLAlchemyError) as e:
        logger.warning("Failed to determine provider pricing: %s", e)

    return False


def _normalize_strategy(strategy: str | None) -> str:
    normalized = (strategy or "for_update").strip().lower()
    if normalized not in {"for_update", "cas", "disabled"}:
        return "for_update"
    return normalized


@dataclass
class ReservationHandle:
    """Tracks a budget reservation so it can be reconciled or released.

    ``estimate`` is the amount added to ``users.reserved`` at reservation time;
    ``reserved`` records whether that write actually happened (it is skipped for
    the ``disabled`` strategy, users without a budget, and free models). The
    handle is passed to :func:`reconcile_reservation` on success or
    :func:`refund_reservation` on failure.

    The scoped-budget fields track the second, independent mechanism (see
    :mod:`gateway.services.scoped_budget_service`). They are separate from
    ``estimate`` / ``reserved`` because the two can diverge: a user with no
    budget row still holds against every scoped ceiling that applies, and a
    reservation that grows may grow on one mechanism and not the other.
    """

    user_id: str
    estimate: Decimal
    reserved: bool
    strategy: str
    # When false, reconciliation records the usage row's cost but does NOT write it
    # to ``users.spend`` and never gates enforcement. Set for requests on keys flagged
    # ``exclude_from_budget`` (and reused for imported usage). Defaults to true so every
    # existing construction site keeps the normal enforced behavior.
    counts_toward_budget: bool = True
    # The scoped ceilings this reservation is holding against, in the order the
    # holds were taken, and the amount held on each. Empty for a caller that
    # passes no scope, which is what keeps every pre-existing construction site
    # (an empty handle for external spend, the vision side-call) inert here.
    scoped_budgets: tuple[ApplicableBudget, ...] = ()
    scoped_estimate: Decimal = ZERO
    # The ledger row recording this hold, or None when the request holds nothing
    # worth ledgering (a free model, a budget-exempt key, a user with no budget
    # and no scoped ceiling). It is what makes reconcile/refund idempotent and a
    # leaked hold reclaimable by identity; see
    # :mod:`gateway.services.budget_reservation_ledger`.
    reservation_id: str | None = None

    @property
    def scoped_budget_ids(self) -> tuple[str, ...]:
        """The ids of the scoped ceilings this reservation holds against."""
        return tuple(applicable.budget_id for applicable in self.scoped_budgets)


def estimate_cost(
    pricing: ModelPricing | None,
    *,
    prompt_chars: int,
    max_output_tokens: int | None,
    default_output_tokens: int,
    cache_write_ttl: Literal["5m", "1h"] | None = None,
) -> Decimal:
    """Estimate request cost up front for budget pre-debit.

    There is no tokenizer in the gateway, so prompt tokens are approximated as
    ``chars / 4`` (a common rough heuristic), rounded up to a whole token.
    Output tokens default to the
    request's declared max, falling back to ``default_output_tokens`` when the
    caller leaves the output unbounded. When Anthropic cache creation is
    requested, the input is conservatively reserved at the cache-write rate,
    since any prompt token could become a cache write. The estimate is
    reconciled to actual usage on completion.
    """
    if pricing is None:
        return ZERO
    # Whole tokens, rounded up: a fraction of a token is not billable, and the
    # estimate is an upper bound, so the fraction rounds towards the gateway.
    prompt_tokens = (max(prompt_chars, 0) + 3) // 4
    # `is None` rather than falsy: max_output_tokens == 0 is an explicit "no
    # output" bound and must not fall through to the default cap. Clamp negatives
    # so a hostile max_output_tokens can't produce a negative estimate.
    output_tokens = max_output_tokens if max_output_tokens is not None else default_output_tokens
    output_tokens = max(output_tokens, 0)
    # Stays ``Decimal``: the estimate is an upper bound reconciled against an
    # exact settlement and so *could* be narrowed here, but on the
    # stream-without-usage path it is what ``log_usage`` settles the row at
    # (``cost_override=reservation.estimate``), which makes it accounting after
    # all. Keeping it exact also means the amount released is the amount held
    # (mozilla-ai/otari#691).
    return estimate_metered_cost(
        pricing,
        estimated_input_tokens=prompt_tokens,
        estimated_output_tokens=output_tokens,
        cache_write_ttl=cache_write_ttl,
    )


async def _held_handle(
    db: AsyncSession,
    *,
    user_id: str,
    estimate: Decimal,
    user_reserved: bool,
    strategy: str,
    counts_toward_budget: bool,
    scoped: tuple[ApplicableBudget, ...],
    scoped_estimate: Decimal,
    ttl_seconds: int,
    record_reservation: bool,
) -> ReservationHandle:
    """Build the handle for a hold that has been taken, and ledger it.

    The row is written here, after every conditional UPDATE has landed, so the
    ledger never claims a hold the counters do not have. The reverse window (a
    hold with no row) is the leak the sweep already bounds; this one would have
    the sweep hand back an amount nobody holds.

    A ledger write that fails does not fail the request. The hold is already live
    at this point, so raising would leave the caller with no handle to reconcile
    or refund and guarantee the very leak the ledger exists to prevent. Degrading
    to an unledgered hold keeps the caller's normal settlement path working,
    which is exactly the behavior every reservation had before this table existed.
    """
    reservation_id: str | None = None
    if record_reservation:
        try:
            reservation_id = await ledger.record(
                db,
                user_id=user_id,
                estimate=estimate,
                user_reserved=user_reserved,
                scoped_budgets=scoped,
                scoped_estimate=scoped_estimate,
                ttl_seconds=ttl_seconds,
            )
        except SQLAlchemyError:
            # Leave the session usable for the caller's own settlement writes; a
            # failed commit otherwise poisons it for the rest of the request.
            with contextlib.suppress(SQLAlchemyError):
                await db.rollback()
            logger.warning(
                "Could not write the reservation ledger row for user %s; the hold is live but "
                "unledgered, so it settles through the handle and is not reclaimable by the sweep.",
                user_id,
                exc_info=True,
            )
    return ReservationHandle(
        user_id=user_id,
        estimate=estimate,
        reserved=user_reserved,
        strategy=strategy,
        counts_toward_budget=counts_toward_budget,
        scoped_budgets=scoped,
        scoped_estimate=scoped_estimate,
        reservation_id=reservation_id,
    )


async def reserve_budget(
    db: AsyncSession,
    user_id: str,
    estimate: Decimal | float,
    *,
    model: str | None = None,
    pricing_provider: str | None = None,
    strategy: str = "for_update",
    counts_toward_budget: bool = True,
    scope: BudgetScopeRequest | None = None,
    organization_id: uuid.UUID | None = None,
    reservation_ttl_sec: int = DEFAULT_RESERVATION_TTL_SEC,
    record_reservation: bool = True,
) -> ReservationHandle:
    """Atomically pre-debit an estimated cost against every budget that applies.

    This replaces the old check-then-call pattern (validate, release the lock,
    call the provider, write spend in a *later* transaction) that allowed
    concurrent requests to all pass a stale budget check and collectively
    overspend. Here the estimate is committed to ``users.reserved`` via a single
    conditional UPDATE: if it would push ``spend + reserved`` past ``max_budget``
    the row count is zero and we reject with 403. No row lock is held across the
    provider network call.

    ``organization_id`` is whose rate overrides decide whether ``model`` is free,
    and it is passed in rather than derived here: the caller has already resolved
    it for its own pricing lookup, and re-deriving it would repeat the two
    un-memoized reads a master-key request pays for ``default_workspace_id``.
    Omitted, the free-model check reads the deployment price list, which is what
    it did before overrides existed.

    ``scope`` opts the request into the second mechanism, the tenancy-scoped
    ceilings in ``scoped_budgets``. Those are resolved from the workspace the
    request bills to and the identity behind the key, and every one of them must
    also admit the estimate. They are held first, and released again if the
    per-user gate then refuses, so a rejected request leaves no counter behind.

    The returned handle must be passed to :func:`reconcile_reservation` (success)
    or :func:`refund_reservation` (failure) so the reservation does not leak.

    Every hold taken here is also written to the reservation ledger, which is what
    gives it an identity: reconcile and refund claim that row before touching a
    counter, so a second one is a no-op rather than a second refund, and a hold
    the request never gets back to is reclaimable on its own rather than only in
    aggregate. ``record_reservation=False`` is for :func:`increase_reservation`,
    which grows the row this request already has instead of opening a second one.
    """
    # Widened once, here, so every expression below is exact whatever the caller
    # handed in: a route that estimates a flat dollar amount still passes a float,
    # and in PostgreSQL a float added to a NUMERIC column resolves the whole
    # expression as double precision.
    #
    # Defense-in-depth: estimates derive from client-controlled fields (max
    # tokens, image count). A negative estimate would *reduce* users.reserved and
    # weaken the budget gate, so never let one reach the DB.
    held = max(to_usd(estimate), ZERO)
    normalized = _normalize_strategy(strategy)
    user = await get_active_user(db, user_id, for_update=False)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found",
        )
    if user.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' is blocked",
        )

    # Budget-exempt request (e.g. a key flagged exclude_from_budget): the user is
    # still validated and a block still rejects, but no estimate is reserved and
    # reconciliation will not write spend. The handle carries the decision so every
    # downstream reconcile/refund site inherits it.
    if not counts_toward_budget:
        return ReservationHandle(
            user_id=user_id,
            estimate=ZERO,
            reserved=False,
            strategy=normalized,
            counts_toward_budget=False,
        )

    no_reservation = ReservationHandle(user_id=user_id, estimate=ZERO, reserved=False, strategy=normalized)

    if normalized == "disabled":
        return no_reservation

    budget = await _get_budget(db, user.budget_id) if user.budget_id else None

    if budget is not None:
        now = datetime.now(UTC)
        if user.next_budget_reset_at and now >= user.next_budget_reset_at:
            # Always reset via the atomic CAS path: reserve_budget never holds a
            # row lock (see for_update=False above), so a non-atomic
            # read-modify-write reset would let concurrent requests at the reset
            # boundary double-reset (duplicate reset logs, clobbered spend).
            user = await _cas_reset_user_budget(db, user, budget, now)

    # Resolved after the per-user lookup so a caller that passes no scope keeps
    # the original fast path exactly, and skipped entirely when there is nothing
    # to hold against on either mechanism.
    scoped = await applicable_budgets(db, user_id=user_id, scope=scope) if scope is not None else ()
    if budget is None and not scoped:
        return no_reservation

    # Free models do not consume budget; nothing to reserve on either mechanism.
    # Reconciliation will add their (zero) cost to spend. Priced at the caller's
    # organization's rate, so "free" means free at what this request will settle
    # at rather than at the deployment's list price.
    if model and await _is_model_free(db, model, pricing_provider=pricing_provider, organization_id=organization_id):
        return no_reservation

    # Reclaim this user's leaked holds, the same idiom as the period roll above: a
    # hold an earlier request left behind keeps shrinking this user's headroom
    # with nothing ever releasing it: the reset zeroes spend and leaves the hold
    # where it is.
    #
    # Deliberately here rather than earlier, so it costs a query only on a request
    # that is actually about to take a hold. Everything above this line returns a
    # handle holding nothing (no budget and no ceiling, a free model, a disabled
    # strategy, an exempt key), and none of those can be refused by a leak, so
    # sweeping for one would be a read per request bought for nothing. Skipped on
    # a top-up too, which runs inside a request whose own hold is live.
    if record_reservation:
        await ledger.reclaim_expired_for_user(db, user_id)

    if scoped:
        refused = await reserve_scoped(db, scoped, held)
        if refused is not None:
            record_budget_exceeded()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{refused.subject} has exceeded budget limit",
            )

    if budget is None:
        # Reachable only with scoped ceilings held (the no-budget, no-scope case
        # returned above), so the user leg of the handle is deliberately empty.
        return await _held_handle(
            db,
            user_id=user_id,
            estimate=ZERO,
            user_reserved=False,
            strategy=normalized,
            counts_toward_budget=counts_toward_budget,
            scoped=scoped,
            scoped_estimate=held,
            ttl_seconds=reservation_ttl_sec,
            record_reservation=record_reservation,
        )

    if budget.max_budget is None:
        # No cap to enforce, but still reserve so reconciliation math is uniform
        # and concurrent spend is reflected immediately.
        await db.execute(
            update(User)
            .where(User.user_id == user_id, User.deleted_at.is_(None))
            .values(reserved=User.reserved + held)
            .execution_options(synchronize_session=False)
        )
        await db.commit()
        return await _held_handle(
            db,
            user_id=user_id,
            estimate=held,
            user_reserved=True,
            strategy=normalized,
            counts_toward_budget=counts_toward_budget,
            scoped=scoped,
            scoped_estimate=held if scoped else ZERO,
            ttl_seconds=reservation_ttl_sec,
            record_reservation=record_reservation,
        )

    result = await db.execute(
        update(User)
        .where(
            User.user_id == user_id,
            User.deleted_at.is_(None),
            # Already at/over the cap → reject (matches the pre-reservation
            # `spend >= max_budget` semantics, and also catches zero-estimate
            # requests like audio for a maxed-out user).
            User.spend + User.reserved < budget.max_budget,
            # ...and this request must not push committed spend past the cap.
            User.spend + User.reserved + held <= budget.max_budget,
        )
        .values(reserved=User.reserved + held)
        .execution_options(synchronize_session=False)
    )
    await db.commit()

    if not getattr(result, "rowcount", 0):
        record_budget_exceeded()
        # The scoped ceilings admitted this request and are already holding the
        # estimate, so give it back before rejecting. Without this the holds would
        # leak on every per-user refusal and permanently shrink each ceiling.
        await release_scoped(db, [item.budget_id for item in scoped], held)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )

    return await _held_handle(
        db,
        user_id=user_id,
        estimate=held,
        user_reserved=True,
        strategy=normalized,
        counts_toward_budget=counts_toward_budget,
        scoped=scoped,
        scoped_estimate=held if scoped else ZERO,
        ttl_seconds=reservation_ttl_sec,
        record_reservation=record_reservation,
    )


def _release_reserved(estimate: Decimal) -> object:
    """Column expression that subtracts ``estimate`` from reserved, clamped at 0.

    Defined in :mod:`gateway.services.budget_reservation_ledger`, which needs the
    same expression for the reclaim path; re-exported here so the call sites in
    this module read as they always have.
    """
    return ledger.release_reserved_expression(estimate)


async def reconcile_reservation(db: AsyncSession, handle: ReservationHandle, actual_cost: Decimal | float) -> None:
    """Settle a reservation: record actual spend and release the held estimate.

    Note: if this UPDATE/commit fails (e.g. a transient DB error after the
    provider call succeeded), the held estimate is not released and stays in
    ``users.reserved``. The claim and the writes below are one transaction, so a
    failure rolls the row back to active rather than leaving it terminal with its
    hold still held, and the TTL sweep reclaims it. Before the ledger nothing did:
    the budget reset zeroes ``spend`` and leaves ``reserved`` untouched, so such a
    hold was never given back.

    Idempotent by reservation identity: the first caller to claim the ledger row
    does the work and any later one is a no-op, so two settlement sites firing for
    one request cannot release the hold twice. A request that outlived its own TTL
    is the one exception: the sweep has already returned its hold, so this records
    the spend it still owes and releases nothing.

    This is the single authority for writing ``users.spend`` on the billable
    path: the usage-log writer no longer touches spend, so reconciliation must
    run for every served request (even when ``actual_cost`` is 0, to release the
    reservation). Runs inline in the request, not in the (possibly batched) log
    writer, so the next request's reservation sees fresh totals.
    """
    # The settled cost reaches ``users.spend`` unchanged: both are exact to the
    # micro-dollar, so the sum of a user's rows is the counter a 403 is decided
    # against. A caller still holding a float (an imported amount, a platform
    # report) is widened rather than the counter narrowed. Never let a negative
    # cost reduce recorded spend.
    # Claim the terminal transition first: whoever wins it is the only caller that
    # releases the hold and records the spend. Without this the second reconcile
    # for one request would subtract the hold again, and because the release
    # expression clamps at zero that would pass silently as an under-count of
    # live holds rather than fail.
    reclaimed_early = False
    if not await ledger.try_terminate(db, handle.reservation_id, ledger.RESERVATION_SETTLED):
        # Losing that claim has two causes and they settle differently. Another
        # settlement site for this request already ran, and there is nothing left
        # to do; or the TTL sweep reclaimed the hold while the request was still
        # alive, in which case the hold is gone but the spend it went on to incur
        # is still owed. Dropping it would leave ``users.spend`` permanently short
        # of the sum of that user's rows, which is the counter a 403 is decided
        # against.
        if not await ledger.try_settle_reclaimed(db, handle.reservation_id):
            # Commit rather than roll back the empty transaction: the guarded UPDATE
            # matched nothing, so there is nothing to undo, and ``rollback()`` expires
            # every ORM instance in the session regardless of ``expire_on_commit``,
            # turning the caller's next attribute read into sync IO on an async session.
            await db.commit()
            return
        reclaimed_early = True
        logger.warning(
            "Reservation %s was reclaimed as leaked before its request settled; recording the spend "
            "without releasing a hold. Raise budget_reservation_ttl_sec above the slowest request served.",
            handle.reservation_id,
        )

    spent = max(to_usd(actual_cost), ZERO)
    values: dict[str, object] = {}
    # Budget-exempt rows are recorded (their cost still lands on the usage row) but
    # never fold into users.spend, so they cannot gate a later request. Gating the
    # spend write here (not merely skipping the reserve) is what makes an empty
    # handle safe at every reconcile site.
    if spent and handle.counts_toward_budget:
        values["spend"] = User.spend + spent
    # Nothing to give back when the reclaim already did it: subtracting a second
    # time is the double release the ledger exists to prevent.
    if handle.reserved and not reclaimed_early:
        values["reserved"] = _release_reserved(handle.estimate)
    # Every scoped ceiling the reservation held against has to be unwound too, or
    # the hold outlives the request and permanently shrinks that ceiling.
    await settle_scoped(
        db,
        handle.scoped_budget_ids,
        actual_cost=spent,
        held=ZERO if reclaimed_early else handle.scoped_estimate,
        counts_toward_budget=handle.counts_toward_budget,
        commit=False,
    )
    if values:
        await db.execute(
            update(User)
            .where(User.user_id == handle.user_id, User.deleted_at.is_(None))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
    # One commit for the claim, the ceilings and the user row. Committing the
    # claim on its own would mean a failure below left the row terminal with its
    # hold still held and its spend unrecorded, which no sweep would ever revisit.
    await db.commit()


async def record_external_spend(db: AsyncSession, user_id: str, cost: Decimal | float) -> None:
    """Fold already-incurred cost into ``users.spend`` outside the reservation flow.

    Used by asynchronous billable paths (batch results) where the create-time
    reservation has already been reconciled and no live hold exists at the point
    the cost becomes known. This deliberately does not enforce the budget: the
    spend has already happened upstream at the provider, so it is recorded, not
    gated. Writing goes through :func:`reconcile_reservation` (with an empty
    handle) so ``users.spend`` still has a single writer.

    Scoped ceilings are deliberately not touched here. The handle that held them
    was reconciled when the batch was created, and this path has no request scope
    to resolve a workspace or a provider from, so folding the cost in would have
    to guess which ceilings it belonged to.
    """
    handle = ReservationHandle(user_id=user_id, estimate=ZERO, reserved=False, strategy="disabled")
    await reconcile_reservation(db, handle, cost)


async def refund_reservation(db: AsyncSession, handle: ReservationHandle) -> None:
    """Release a reservation without recording spend (e.g. provider failure).

    Idempotent by reservation identity, for the same reason as
    :func:`reconcile_reservation`: ``release_reservation`` in ``_pipeline`` is
    reachable from roughly seven sites, and only control flow (a ``raise`` after
    each) has kept two of them from firing for one request.
    """
    if not await ledger.try_terminate(db, handle.reservation_id, ledger.RESERVATION_RELEASED):
        # Commit rather than roll back the empty transaction: the guarded UPDATE
        # matched nothing, so there is nothing to undo, and ``rollback()`` expires
        # every ORM instance in the session regardless of ``expire_on_commit``,
        # turning the caller's next attribute read into sync IO on an async session.
        await db.commit()
        return
    await release_scoped(db, handle.scoped_budget_ids, handle.scoped_estimate, commit=False)
    if handle.reserved:
        await db.execute(
            update(User)
            .where(User.user_id == handle.user_id, User.deleted_at.is_(None))
            .values(reserved=_release_reserved(handle.estimate))
            .execution_options(synchronize_session=False)
        )
    # One commit, for the reason given in :func:`reconcile_reservation`.
    await db.commit()


async def increase_reservation(
    db: AsyncSession,
    handle: ReservationHandle,
    additional_estimate: Decimal | float,
    *,
    model: str | None = None,
    strategy: str = "for_update",
) -> None:
    """Grow an existing reservation atomically when the request size increases.

    Used when the billable size grows after the initial reservation — e.g. the
    content normalizer expands an attachment into extracted prompt text. The
    delta is reserved with the same atomic conditional UPDATE as
    :func:`reserve_budget` (so the budget gate stays effective on the true
    size), then folded into ``handle`` so the existing reconcile/refund path
    releases the full held amount.

    Like :func:`reserve_budget`, this raises on budget rejection and does *not*
    clean up the prior hold — the caller owns refunding ``handle`` on failure
    (the request routes wrap the whole post-reservation setup in a
    refund-on-error block). No-op when ``additional_estimate`` is not positive.

    The scoped ceilings are grown on exactly the rows the original reservation
    took, not re-resolved: the request scope has not changed, and re-resolving
    would risk holding twice against a ceiling that appeared in between.
    """
    additional = to_usd(additional_estimate)
    if additional <= ZERO:
        return
    # A budget-exempt request never grows a reservation: there is nothing to hold and
    # nothing to gate. Without this, the top-up path would silently re-enter the
    # enforced flow and reserve against the user's budget.
    if not handle.counts_toward_budget:
        return
    if handle.scoped_budgets:
        refused = await reserve_scoped(db, handle.scoped_budgets, additional)
        if refused is not None:
            record_budget_exceeded()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{refused.subject} has exceeded budget limit",
            )
        handle.scoped_estimate += additional
    # No scope is passed through: the scoped ceilings were just grown above, and
    # letting the inner call resolve them again would hold the delta twice.
    # ``record_reservation=False`` for the same reason in the ledger: this request
    # already has a row, and a second one would carry its own TTL and could be
    # reclaimed on its own, handing back part of a live hold.
    delta = await reserve_budget(
        db,
        handle.user_id,
        additional,
        model=model,
        strategy=strategy,
        record_reservation=False,
    )
    if delta.reserved:
        handle.estimate += delta.estimate
        handle.reserved = True

    # Fold both deltas into the row under one guard, after both holds have landed,
    # for the same reason the original reserve writes its row last: the ledger must
    # never claim more than the counters hold.
    #
    # Losing the guard means the sweep reclaimed this hold while the request was
    # still running. The row is terminal and nothing will look at it again, so the
    # deltas just taken would be held by something with no owner. Give them back,
    # and unwind the handle, so the request carries on against the amount it
    # actually holds: nothing. It will still settle, and the late-settlement path
    # in :func:`reconcile_reservation` records what it spent.
    grown = await ledger.grow(
        db,
        handle.reservation_id,
        user_delta=delta.estimate if delta.reserved else ZERO,
        scoped_delta=additional if handle.scoped_budgets else ZERO,
    )
    if not grown:
        logger.warning(
            "Reservation %s was reclaimed as leaked before its top-up; returning the delta. "
            "Raise budget_reservation_ttl_sec above the slowest request served.",
            handle.reservation_id,
        )
        if handle.scoped_budgets:
            await release_scoped(db, handle.scoped_budget_ids, additional)
            handle.scoped_estimate -= additional
        if delta.reserved:
            await db.execute(
                update(User)
                .where(User.user_id == handle.user_id, User.deleted_at.is_(None))
                .values(reserved=_release_reserved(delta.estimate))
                .execution_options(synchronize_session=False)
            )
            await db.commit()
            handle.estimate -= delta.estimate
