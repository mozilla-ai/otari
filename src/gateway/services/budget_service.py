from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
from gateway.services.pricing_service import find_model_pricing


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate next budget reset datetime.

    Args:
        start: Starting datetime for the budget period
        duration_sec: Duration in seconds

    Returns:
        datetime when the budget should next reset

    """
    return start + timedelta(seconds=duration_sec)


async def reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> None:
    """Reset user's budget spend and schedule next reset."""

    previous_spend = float(user.spend)
    user_id_str = user.user_id

    user.spend = 0.0
    user.budget_started_at = now

    if budget.budget_duration_sec:
        user.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
    else:
        user.next_budget_reset_at = None

    reset_log = BudgetResetLog(
        user_id=user.user_id,
        budget_id=budget.budget_id,
        previous_spend=previous_spend,
        reset_at=now,
        next_reset_at=user.next_budget_reset_at,
    )
    db.add(reset_log)

    try:
        await db.commit()
        await db.refresh(user)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for user '%s': %s", user_id_str, e)
        raise


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
        reset_log = BudgetResetLog(
            user_id=user.user_id,
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
            logger.error("Failed to commit CAS budget reset for user '%s': %s", user.user_id, e)
            raise
        refreshed = await get_active_user(db, user.user_id)
        return refreshed or user

    await db.rollback()
    return user


async def _get_budget(db: AsyncSession, budget_id: str) -> Budget | None:
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    return result.scalar_one_or_none()


async def validate_user_budget(
    db: AsyncSession,
    user_id: str,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> User:
    """Validate user exists, is not blocked, and has available budget.

    Args:
        db: Database session
        user_id: User identifier
        model: Optional model identifier (e.g., "provider/model") to check if it's a free model

    Returns:
        User object if validation passes

    Raises:
        HTTPException: If user is blocked, doesn't exist, or exceeded budget

    """
    normalized_strategy = strategy or "for_update"
    normalized_strategy = normalized_strategy.strip().lower()
    if normalized_strategy not in {"for_update", "cas", "disabled"}:
        normalized_strategy = "for_update"

    lock_for_update = normalized_strategy == "for_update"
    user = await get_active_user(db, user_id, for_update=lock_for_update)

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

    if normalized_strategy == "disabled" or not user.budget_id:
        return user

    budget = await _get_budget(db, user.budget_id)
    if not budget:
        return user

    now = datetime.now(UTC)
    if user.next_budget_reset_at and now >= user.next_budget_reset_at:
        if normalized_strategy == "cas":
            user = await _cas_reset_user_budget(db, user, budget, now)
        else:
            await reset_user_budget(db, user, budget, now)

    if budget.max_budget is not None and user.spend >= budget.max_budget:
        if model and await _is_model_free(db, model):
            return user
        record_budget_exceeded()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )

    return user


async def _is_model_free(db: AsyncSession, model: str) -> bool:
    """Check if a model is free (both input and output prices are 0).

    Args:
        db: Database session
        model: Model identifier (e.g., "provider/model" or "model")

    Returns:
        True if the model is free, False otherwise or if pricing not found

    """
    try:
        provider, model_name = AnyLLM.split_model_provider(model)
        provider_str = provider.value if provider else None
        pricing = await find_model_pricing(db, provider_str, model_name)
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
    """

    user_id: str
    estimate: float
    reserved: bool
    strategy: str


def estimate_cost(
    pricing: ModelPricing | None,
    *,
    prompt_chars: int,
    max_output_tokens: int | None,
    default_output_tokens: int,
) -> float:
    """Estimate request cost up front for budget pre-debit.

    There is no tokenizer in the gateway, so prompt tokens are approximated as
    ``chars / 4`` (a common rough heuristic). Output tokens default to the
    request's declared max, falling back to ``default_output_tokens`` when the
    caller leaves the output unbounded. The estimate is intentionally an
    upper-ish bound; it is reconciled to actual usage on completion.
    """
    if pricing is None:
        return 0.0
    prompt_tokens = max(prompt_chars, 0) / 4
    # `is None` rather than falsy: max_output_tokens == 0 is an explicit "no
    # output" bound and must not fall through to the default cap. Clamp negatives
    # so a hostile max_output_tokens can't produce a negative estimate.
    output_tokens = max_output_tokens if max_output_tokens is not None else default_output_tokens
    output_tokens = max(output_tokens, 0)
    return (prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
        output_tokens / 1_000_000
    ) * pricing.output_price_per_million


async def reserve_budget(
    db: AsyncSession,
    user_id: str,
    estimate: float,
    *,
    model: str | None = None,
    strategy: str = "for_update",
) -> ReservationHandle:
    """Atomically pre-debit an estimated cost against the user's budget.

    This replaces the old check-then-call pattern (validate, release the lock,
    call the provider, write spend in a *later* transaction) that allowed
    concurrent requests to all pass a stale budget check and collectively
    overspend. Here the estimate is committed to ``users.reserved`` via a single
    conditional UPDATE: if it would push ``spend + reserved`` past ``max_budget``
    the row count is zero and we reject with 403. No row lock is held across the
    provider network call.

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

    no_reservation = ReservationHandle(user_id=user_id, estimate=0.0, reserved=False, strategy=normalized)

    if normalized == "disabled" or not user.budget_id:
        return no_reservation

    budget = await _get_budget(db, user.budget_id)
    if not budget:
        return no_reservation

    now = datetime.now(UTC)
    if user.next_budget_reset_at and now >= user.next_budget_reset_at:
        # Always reset via the atomic CAS path: reserve_budget never holds a
        # row lock (see for_update=False above), so the read-modify-write
        # reset_user_budget would let concurrent requests at the reset boundary
        # double-reset (duplicate reset logs, clobbered spend).
        user = await _cas_reset_user_budget(db, user, budget, now)

    # Free models do not consume budget; nothing to reserve. Reconciliation will
    # add their (zero) cost to spend.
    if model and await _is_model_free(db, model):
        return no_reservation

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
        return ReservationHandle(user_id=user_id, estimate=estimate, reserved=True, strategy=normalized)

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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )

    return ReservationHandle(user_id=user_id, estimate=estimate, reserved=True, strategy=normalized)


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
    if actual_cost:
        values["spend"] = User.spend + actual_cost
    if handle.reserved:
        values["reserved"] = _release_reserved(handle.estimate)
    if not values:
        return
    await db.execute(
        update(User)
        .where(User.user_id == handle.user_id, User.deleted_at.is_(None))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def refund_reservation(db: AsyncSession, handle: ReservationHandle) -> None:
    """Release a reservation without recording spend (e.g. provider failure)."""
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
    """
    if additional_estimate <= 0:
        return
    delta = await reserve_budget(db, handle.user_id, additional_estimate, model=model, strategy=strategy)
    if delta.reserved:
        handle.estimate += delta.estimate
        handle.reserved = True
