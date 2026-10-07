"""Move a budget's ceilings and users onto the cadence it now carries.

A ceiling and a user each store their own window and read the cadence through their budget.
A change to the budget's period leaves the two in disagreement until they are retimed.
One left with no window never rolls, so its spend accumulates while the budget reports a cadence.
"""

from datetime import UTC, datetime

from sqlalchemy import case, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.budgets import ScopedBudget, ceiling_counters_rolled_if_ended
from gateway.models.users import User
from gateway.services.budgets._periods import budget_window

__all__ = ["cadence_of", "retime_for_budget"]


def cadence_of(budget: object) -> tuple[object, ...]:
    """Everything that decides a window, as one comparable value.

    Exists so both callers compare the same thing, read before and after the
    mutation. Keyed on the cadence rather than on "an update happened", because
    retiming on every write would restart a period for a rename or a limit
    change, throwing away the part of it a ceiling had already spent.

    The whole settings tuple rather than the cycle alone: moving a monthly budget
    from the 1st to the 15th leaves ``reset_cycle`` where it was and is still a
    different window, which a cycle-only comparison would miss.
    """
    return (
        getattr(budget, "reset_cycle", None),
        getattr(budget, "reset_every_n", None),
        getattr(budget, "reset_anchor_at", None),
        getattr(budget, "reset_weekdays", None),
        getattr(budget, "reset_month_day", None),
        getattr(budget, "reset_month", None),
    )


async def retime_for_budget(db: AsyncSession, budget: object, *, budget_id: str) -> None:
    """Rewrite the window on every ceiling naming this budget, and on every user holding it.

    One statement per table rather than a row each: the window is derived from the
    budget and from now, not from anything an individual row holds, so it is the
    same for all of them. Users matter as much as ceilings: a user's reset fires
    only once `next_budget_reset_at` has passed, so a stale date keeps them on the
    old cadence until then, and a budget moved from "no reset" to a cadence left
    them with no date at all, never resetting.

    Counters are deliberately untouched, on both, unless the period they count
    had already ended: then they are rolled as a request would have rolled them.
    Spend already recorded in a live period stays, matching
    what re-pointing a ceiling at a different budget does: the ceiling is the same
    allowance held to a different figure from here on, not a fresh one.
    ``reserved_spend`` is left alone too, so a hold taken before the change is
    still released against the counter it came from.

    Not committed here. The caller owns the transaction, so a write that is
    refused afterwards takes the retiming back with it.

    A cadence of neither kind clears the window rather than deriving one, which is
    what "no reset" means and what ``budget_window`` returns None for.
    """
    now = datetime.now(UTC)
    window = budget_window(now, budget)
    period_start, period_end = window if window is not None else (None, None)
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.budget_id == budget_id)
        .values(period_start=period_start, period_end=period_end, **ceiling_counters_rolled_if_ended(now))
        .execution_options(synchronize_session=False)
    )
    # The same fallback an assignment writes: a user with no cadence has started
    # now and resets never.
    await db.execute(
        update(User)
        .where(User.budget_id == budget_id, User.deleted_at.is_(None))
        .values(
            budget_started_at=period_start or now,
            next_budget_reset_at=period_end,
            # Rolled first if the period had already ended, for the reason
            # `ceiling_counters_rolled_if_ended` gives.
            spend=User.spend_this_period(),
            current_tokens=case((User.next_budget_reset_at <= now, 0), else_=User.current_tokens),
            current_requests=case((User.next_budget_reset_at <= now, 0), else_=User.current_requests),
        )
        .execution_options(synchronize_session=False)
    )
