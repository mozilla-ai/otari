from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from fastapi import HTTPException, status
from sqlalchemy import case, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.metrics import record_budget_exceeded
from gateway.models.entities import Budget, BudgetResetLog, ModelPricing, User
from gateway.repositories.users_repository import get_active_user
from gateway.services.metered_pricing import estimate_metered_cost
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


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate next budget reset datetime.

    Args:
        start: Starting datetime for the budget period
        duration_sec: Duration in seconds

    Returns:
        datetime when the budget should next reset

    """
    return start + timedelta(seconds=duration_sec)


async def _cas_reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> User:
    next_reset_at = calculate_next_reset(now, budget.budget_duration_sec) if budget.budget_duration_sec else None

    result = await db.execute(
        update(User)
        .where(
            User.user_id == user.user_id,
            User.deleted_at.is_(None),
            User.next_budget_reset_at.is_not(None),
            User.next_budget_reset_at <= now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
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
            previous_spend=float(user.spend),
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
    committed = float(spend or 0.0) + float(reserved or 0.0)
    return BudgetState(
        used_pct=committed / float(max_budget) * 100.0,
        remaining_usd=max(0.0, float(max_budget) - committed),
    )


async def _is_model_free(
    db: AsyncSession,
    model: str,
    *,
    pricing_provider: str | None = None,
) -> bool:
    """Check if a model is free (both input and output prices are 0).

    Args:
        db: Database session
        model: Model identifier (e.g., "provider/model" or "model")
        pricing_provider: Resolved provider instance, when ``model`` is already
            the bare model name.

    Returns:
        True if the model is free, False otherwise or if pricing not found

    """
    try:
        if pricing_provider is None:
            provider, model_name = AnyLLM.split_model_provider(model)
            pricing_provider = provider_key(provider) or None
        else:
            model_name = model
        pricing = await find_model_pricing(db, pricing_provider, model_name)
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
    estimate: float
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
    scoped_estimate: float = 0.0

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
) -> float:
    """Estimate request cost up front for budget pre-debit.

    There is no tokenizer in the gateway, so prompt tokens are approximated as
    ``chars / 4`` (a common rough heuristic). Output tokens default to the
    request's declared max, falling back to ``default_output_tokens`` when the
    caller leaves the output unbounded. When Anthropic cache creation is
    requested, the input is conservatively reserved at the cache-write rate,
    since any prompt token could become a cache write. The estimate is
    reconciled to actual usage on completion.
    """
    if pricing is None:
        return 0.0
    prompt_tokens = max(prompt_chars, 0) / 4
    # `is None` rather than falsy: max_output_tokens == 0 is an explicit "no
    # output" bound and must not fall through to the default cap. Clamp negatives
    # so a hostile max_output_tokens can't produce a negative estimate.
    output_tokens = max_output_tokens if max_output_tokens is not None else default_output_tokens
    output_tokens = max(output_tokens, 0)
    return estimate_metered_cost(
        pricing,
        estimated_input_tokens=prompt_tokens,
        estimated_output_tokens=output_tokens,
        cache_write_ttl=cache_write_ttl,
    )


async def reserve_budget(
    db: AsyncSession,
    user_id: str,
    estimate: float,
    *,
    model: str | None = None,
    pricing_provider: str | None = None,
    strategy: str = "for_update",
    counts_toward_budget: bool = True,
    scope: BudgetScopeRequest | None = None,
) -> ReservationHandle:
    """Atomically pre-debit an estimated cost against every budget that applies.

    This replaces the old check-then-call pattern (validate, release the lock,
    call the provider, write spend in a *later* transaction) that allowed
    concurrent requests to all pass a stale budget check and collectively
    overspend. Here the estimate is committed to ``users.reserved`` via a single
    conditional UPDATE: if it would push ``spend + reserved`` past ``max_budget``
    the row count is zero and we reject with 403. No row lock is held across the
    provider network call.

    ``scope`` opts the request into the second mechanism, the tenancy-scoped
    ceilings in ``scoped_budgets``. Those are resolved from the workspace the
    request bills to and the identity behind the key, and every one of them must
    also admit the estimate. They are held first, and released again if the
    per-user gate then refuses, so a rejected request leaves no counter behind.

    The returned handle must be passed to :func:`reconcile_reservation` (success)
    or :func:`refund_reservation` (failure) so the reservation does not leak.
    """
    # Defense-in-depth: estimates derive from client-controlled fields (max
    # tokens, image count). A negative estimate would *reduce* users.reserved and
    # weaken the budget gate, so never let one reach the DB.
    estimate = max(estimate, 0.0)
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
            estimate=0.0,
            reserved=False,
            strategy=normalized,
            counts_toward_budget=False,
        )

    no_reservation = ReservationHandle(user_id=user_id, estimate=0.0, reserved=False, strategy=normalized)

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
    # Reconciliation will add their (zero) cost to spend.
    if model and await _is_model_free(db, model, pricing_provider=pricing_provider):
        return no_reservation

    if scoped:
        refused = await reserve_scoped(db, scoped, estimate)
        if refused is not None:
            record_budget_exceeded()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{refused.subject} has exceeded budget limit",
            )

    if budget is None:
        # Reachable only with scoped ceilings held (the no-budget, no-scope case
        # returned above), so the user leg of the handle is deliberately empty.
        return ReservationHandle(
            user_id=user_id,
            estimate=0.0,
            reserved=False,
            strategy=normalized,
            scoped_budgets=scoped,
            scoped_estimate=estimate,
        )

    if budget.max_budget is None:
        # No cap to enforce, but still reserve so reconciliation math is uniform
        # and concurrent spend is reflected immediately.
        await db.execute(
            update(User)
            .where(User.user_id == user_id, User.deleted_at.is_(None))
            .values(reserved=User.reserved + estimate)
            .execution_options(synchronize_session=False)
        )
        await db.commit()
        return ReservationHandle(
            user_id=user_id,
            estimate=estimate,
            reserved=True,
            strategy=normalized,
            scoped_budgets=scoped,
            scoped_estimate=estimate if scoped else 0.0,
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
            User.spend + User.reserved + estimate <= budget.max_budget,
        )
        .values(reserved=User.reserved + estimate)
        .execution_options(synchronize_session=False)
    )
    await db.commit()

    if not getattr(result, "rowcount", 0):
        record_budget_exceeded()
        # The scoped ceilings admitted this request and are already holding the
        # estimate, so give it back before rejecting. Without this the holds would
        # leak on every per-user refusal and permanently shrink each ceiling.
        await release_scoped(db, [item.budget_id for item in scoped], estimate)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )

    return ReservationHandle(
        user_id=user_id,
        estimate=estimate,
        reserved=True,
        strategy=normalized,
        scoped_budgets=scoped,
        scoped_estimate=estimate if scoped else 0.0,
    )


def _release_reserved(estimate: float) -> object:
    """Column expression that subtracts ``estimate`` from reserved, clamped at 0.

    Uses CASE rather than GREATEST for SQLite compatibility.
    """
    return case(
        (User.reserved - estimate < 0, 0.0),
        else_=User.reserved - estimate,
    )


async def reconcile_reservation(db: AsyncSession, handle: ReservationHandle, actual_cost: float) -> None:
    """Settle a reservation: record actual spend and release the held estimate.

    Note: if this UPDATE/commit fails (e.g. a transient DB error after the
    provider call succeeded), the held estimate is not released and stays in
    ``users.reserved``. That shrinks the user's effective budget until the next
    budget reset zeroes it; a future enhancement could add a stale-reservation
    sweep. This is the cost of fail-closed pre-debit and is rare in practice.

    This is the single authority for writing ``users.spend`` on the billable
    path — the usage-log writer no longer touches spend, so reconciliation must
    run for every served request (even when ``actual_cost`` is 0, to release the
    reservation). Runs inline in the request, not in the (possibly batched) log
    writer, so the next request's reservation sees fresh totals.
    """
    # Never let a negative cost reduce recorded spend.
    actual_cost = max(actual_cost, 0.0)
    values: dict[str, object] = {}
    # Budget-exempt rows are recorded (their cost still lands on the usage row) but
    # never fold into users.spend, so they cannot gate a later request. Gating the
    # spend write here (not merely skipping the reserve) is what makes an empty
    # handle safe at every reconcile site.
    if actual_cost and handle.counts_toward_budget:
        values["spend"] = User.spend + actual_cost
    if handle.reserved:
        values["reserved"] = _release_reserved(handle.estimate)
    # Every scoped ceiling the reservation held against has to be unwound too, or
    # the hold outlives the request and permanently shrinks that ceiling.
    await settle_scoped(
        db,
        handle.scoped_budget_ids,
        actual_cost=actual_cost,
        held=handle.scoped_estimate,
        counts_toward_budget=handle.counts_toward_budget,
    )
    if not values:
        return
    await db.execute(
        update(User)
        .where(User.user_id == handle.user_id, User.deleted_at.is_(None))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def record_external_spend(db: AsyncSession, user_id: str, cost: float) -> None:
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
    handle = ReservationHandle(user_id=user_id, estimate=0.0, reserved=False, strategy="disabled")
    await reconcile_reservation(db, handle, cost)


async def refund_reservation(db: AsyncSession, handle: ReservationHandle) -> None:
    """Release a reservation without recording spend (e.g. provider failure)."""
    await release_scoped(db, handle.scoped_budget_ids, handle.scoped_estimate)
    if not handle.reserved:
        return
    await db.execute(
        update(User)
        .where(User.user_id == handle.user_id, User.deleted_at.is_(None))
        .values(reserved=_release_reserved(handle.estimate))
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def increase_reservation(
    db: AsyncSession,
    handle: ReservationHandle,
    additional_estimate: float,
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
    if additional_estimate <= 0:
        return
    # A budget-exempt request never grows a reservation: there is nothing to hold and
    # nothing to gate. Without this, the top-up path would silently re-enter the
    # enforced flow and reserve against the user's budget.
    if not handle.counts_toward_budget:
        return
    if handle.scoped_budgets:
        refused = await reserve_scoped(db, handle.scoped_budgets, additional_estimate)
        if refused is not None:
            record_budget_exceeded()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{refused.subject} has exceeded budget limit",
            )
        handle.scoped_estimate += additional_estimate
    # No scope is passed through: the scoped ceilings were just grown above, and
    # letting the inner call resolve them again would hold the delta twice.
    delta = await reserve_budget(db, handle.user_id, additional_estimate, model=model, strategy=strategy)
    if delta.reserved:
        handle.estimate += delta.estimate
        handle.reserved = True
