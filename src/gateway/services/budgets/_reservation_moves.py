from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from gateway.exceptions.budget_exceptions import ProviderSpendCeilingExceededError, SpendCeilingExceededError
from gateway.models.budgets import MAX_COUNT_LIMIT
from gateway.repositories.budgets import HeldAmounts
from gateway.services.budgets._refusals import refusal_detail
from gateway.services.budgets._reservations import (
    BUDGET_EXCEEDED,
    ZERO,
    EstimateInputs,
    ReservationHandle,
    estimate_cost,
    estimate_tokens,
)
from gateway.services.budgets._scoped_enforcement import (
    ApplicableBudget,
    get_provider_ceilings,
    refused_axis,
    roll_ended_windows,
    to_applicable_budget,
)
from gateway.services.pricing_service import is_free_pricing

if TYPE_CHECKING:
    from gateway.models.pricing import ModelPricing
    from gateway.repositories.budgets import ApplicableCeiling, BudgetRepositories
    from gateway.types.attempt import Attempt

type PricingLookup = Callable[[str, str], Awaitable[ModelPricing | None]]


@dataclass(frozen=True)
class ReservationMove:
    """What moving one reservation onto the ceilings of a candidate's provider changes."""

    target: tuple[ApplicableCeiling, ...]
    added: tuple[ApplicableCeiling, ...]
    shared: tuple[ApplicableBudget, ...]
    dropped: tuple[ApplicableBudget, ...]
    held: HeldAmounts
    # ``None`` for a free candidate, whose dollars are neither held nor gated.
    gated_amount: Decimal | None
    reserved_output_tokens: int | None
    reservation_id: str
    writes_row: bool


@dataclass(frozen=True)
class MoveOutcome:
    """What applying a move did, which is nothing when the reservation was reclaimed or a ceiling refused."""

    reclaimed: bool = False
    refusal: SpendCeilingExceededError | None = None


@dataclass(frozen=True)
class _GatedHold:
    """A hold on one ceiling that the ceiling can refuse."""

    budget: ApplicableBudget
    amount: Decimal | None
    tokens: int
    requests: int
    new_request: bool


class _ReservationMoves:
    """Move a reservation onto the ceilings that bind the provider a candidate calls."""

    def __init__(self, repositories: BudgetRepositories) -> None:
        self._ceilings = repositories.ceilings
        self._reservations = repositories.reservations

    async def plan(
        self,
        handle: ReservationHandle,
        attempt: Attempt,
        *,
        estimate_inputs: EstimateInputs | None,
        price_of: PricingLookup,
    ) -> ReservationMove | None:
        """Return what moving ``handle`` onto ``attempt``'s ceilings changes, or None when it changes nothing.

        A fallback chain's holds only grow, so a ceiling holds the larger of its hold and ``attempt``'s estimate.
        """
        if not handle.counts_toward_budget or handle.strategy == "disabled":
            return None
        target = get_provider_ceilings(handle.scope_ceilings, attempt.instance)
        if not target and not handle.scoped_budgets:
            return None
        pricing: ModelPricing | None = None
        if estimate_inputs is not None:
            pricing = await price_of(attempt.instance, attempt.model)
            amount = estimate_cost(
                pricing,
                prompt_chars=estimate_inputs.prompt_chars,
                max_output_tokens=estimate_inputs.max_output_tokens,
                default_output_tokens=estimate_inputs.default_output_tokens,
                cache_write_ttl=estimate_inputs.cache_write_ttl,
            )
        else:
            amount = handle.scoped_estimate if handle.scoped_budgets else handle.estimate
        if handle.scoped_budgets:
            held = HeldAmounts(
                amount=max(amount, handle.scoped_estimate),
                tokens=handle.scoped_token_estimate,
                requests=handle.scoped_request_estimate,
            )
        else:
            tokens = (
                estimate_tokens(
                    prompt_chars=estimate_inputs.prompt_chars,
                    max_output_tokens=estimate_inputs.max_output_tokens,
                    default_output_tokens=estimate_inputs.default_output_tokens,
                )
                if estimate_inputs is not None
                else handle.token_estimate
            )
            held = HeldAmounts(amount=amount, tokens=min(max(tokens, 0), MAX_COUNT_LIMIT), requests=1)
        unchanged = tuple(ceiling.scoped_budget_id for ceiling in target) == handle.scoped_budget_ids
        if unchanged and held.amount <= handle.scoped_estimate:
            return None
        # A free request that holds no dollars is not gated on them, and on ceilings that cap no count holds nothing.
        free = is_free_pricing(pricing) and held.amount == ZERO
        if free and not any(to_applicable_budget(ceiling).caps_counts for ceiling in target):
            return None
        target_ids = {ceiling.scoped_budget_id for ceiling in target}
        held_ids = set(handle.scoped_budget_ids)
        return ReservationMove(
            target=target,
            added=tuple(ceiling for ceiling in target if ceiling.scoped_budget_id not in held_ids),
            shared=tuple(budget for budget in handle.scoped_budgets if budget.budget_id in target_ids),
            dropped=tuple(budget for budget in handle.scoped_budgets if budget.budget_id not in target_ids),
            held=held,
            gated_amount=None if free else held.amount,
            reserved_output_tokens=estimate_inputs.get_reserved_output_tokens(pricing)
            if estimate_inputs is not None
            else None,
            reservation_id=handle.reservation_id or str(uuid.uuid4()),
            writes_row=handle.reservation_id is None and bool(target),
        )

    async def apply(self, handle: ReservationHandle, move: ReservationMove) -> MoveOutcome:
        """Take and record the holds of ``move``, or take none when a ceiling refuses one.

        The ceilings are locked in the order of their IDs before any hold, so two concurrent moves cannot deadlock.
        The holds a ceiling can refuse are taken first, and given back when one is refused.

        NOTE: The caller must run this in one block of the Unit of Work.
        """
        if handle.reservation_id is not None and not await self._reservations.lock_active(handle.reservation_id):
            return MoveOutcome(reclaimed=True)
        await self._ceilings.lock(
            sorted(
                [ceiling.scoped_budget_id for ceiling in move.added]
                + [budget.budget_id for budget in (*move.shared, *move.dropped)]
            )
        )
        now = datetime.now(UTC)
        await roll_ended_windows(self._ceilings, move.added, now)
        growth = move.held.amount - handle.scoped_estimate
        gated = [
            _GatedHold(to_applicable_budget(ceiling), move.gated_amount, move.held.tokens, move.held.requests, True)
            for ceiling in move.added
        ]
        if growth > ZERO:
            gated += [_GatedHold(budget, growth, 0, 0, False) for budget in move.shared]
        taken: list[_GatedHold] = []
        for hold in sorted(gated, key=lambda item: item.budget.budget_id):
            if not await self._ceilings.hold(
                hold.budget.budget_id,
                hold.amount,
                tokens=hold.tokens,
                requests=hold.requests,
                new_request=hold.new_request,
            ):
                for held in taken:
                    await self._ceilings.release_holds(
                        [held.budget.budget_id], held.amount or ZERO, tokens=held.tokens, requests=held.requests
                    )
                return MoveOutcome(refusal=await self._refusal(hold, move.reserved_output_tokens))
            taken.append(hold)
        if move.dropped:
            await self._ceilings.release_holds(
                [budget.budget_id for budget in move.dropped],
                handle.scoped_estimate,
                tokens=handle.scoped_token_estimate,
                requests=handle.scoped_request_estimate,
            )
        if handle.reservation_id is not None:
            await self._reservations.move_lines(
                handle.reservation_id,
                dropped=[budget.budget_id for budget in move.dropped],
                added=[ceiling.scoped_budget_id for ceiling in move.added],
                held=move.held,
            )
        elif move.writes_row:
            await self._reservations.add(
                move.reservation_id,
                user_id=handle.user_id,
                user_reserved=handle.reserved,
                user_held=HeldAmounts(
                    amount=handle.estimate,
                    tokens=handle.token_estimate,
                    requests=handle.request_estimate if handle.reserved else 0,
                ),
                ceiling_held=move.held,
                scoped_budget_ids=[ceiling.scoped_budget_id for ceiling in move.target],
                expires_at=now + timedelta(seconds=handle.reservation_ttl_sec),
            )
        return MoveOutcome()

    async def landed(self, handle: ReservationHandle, move: ReservationMove) -> bool:
        """Report whether the ledger records the holds of ``move``, waiting for a commit of it that is still resolving.

        NOTE: The caller must run this in a block of the Unit of Work.
        """
        if move.writes_row:
            # A released row under the move's ID is the fence: it is a duplicate exactly when the move's row landed.
            return not await self._reservations.add_released_unless_exists(move.reservation_id, user_id=handle.user_id)
        if not await self._reservations.lock_active(move.reservation_id):
            return False
        lines = await self._reservations.list_lines(move.reservation_id)
        # NOTE: A move that only grows its holds keeps its ceilings, so only the amounts show that it landed.
        target_ids = {ceiling.scoped_budget_id for ceiling in move.target}
        return {line.scoped_budget_id for line in lines} == target_ids and all(line.held == move.held for line in lines)

    @staticmethod
    def retarget(handle: ReservationHandle, move: ReservationMove) -> None:
        """Make ``handle`` name the ceilings and holds of ``move``."""
        handle.scoped_budgets = tuple(to_applicable_budget(ceiling) for ceiling in move.target)
        handle.scoped_estimate = move.held.amount if move.target else ZERO
        handle.scoped_token_estimate = move.held.tokens if move.target else 0
        handle.scoped_request_estimate = move.held.requests if move.target else 0
        if move.writes_row:
            handle.reservation_id = move.reservation_id

    async def _refusal(self, hold: _GatedHold, reserved_output_tokens: int | None) -> SpendCeilingExceededError:
        """Return the error for a hold its ceiling has no room for, naming the axis it ran out of.

        A ceiling on one provider refuses that provider only, so its error says another provider may still fit.
        """
        usage = await self._ceilings.get_usage(hold.budget.budget_id)
        blocked = refused_axis(
            usage, amount=hold.amount, tokens=hold.tokens, requests=hold.requests, new_request=hold.new_request
        )
        detail = refusal_detail(
            hold.budget.subject, blocked, new_request=hold.new_request, reserved_output_tokens=reserved_output_tokens
        )
        retry_after = blocked.retry_after(datetime.now(UTC))
        if hold.budget.provider_key_id is not None:
            return ProviderSpendCeilingExceededError(detail, scope_type=hold.budget.scope_type, retry_after=retry_after)
        BUDGET_EXCEEDED.inc()
        return SpendCeilingExceededError(detail, scope_type=hold.budget.scope_type, retry_after=retry_after)
