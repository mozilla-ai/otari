"""Move a budget's ceilings and users onto the cadence it now carries.

A ceiling and a user each store their own window and read the cadence through their budget.
A change to the budget's period leaves the two in disagreement until they are retimed.
One left with no window never rolls, so its spend accumulates while the budget reports a cadence.
"""

from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.budgets import Budget, ScopedBudget, ceiling_counters_rolled_if_ended
from gateway.repositories.users_repository import retime_budget_holders
from gateway.services.budgets._periods import CycleSettings, budget_window, cycle_settings_of

__all__ = ["cadence_of", "retime_for_budget"]


def cadence_of(budget: Budget) -> CycleSettings:
    """Everything that decides a window, as one comparable value.

    Exists so both callers compare the same thing, read before and after the
    mutation. Keyed on the cadence rather than on "an update happened", because
    retiming on every write would restart a period for a rename or a limit
    change, throwing away the part of it a ceiling had already spent.

    The whole settings tuple rather than the cycle alone: moving a monthly budget
    from the 1st to the 15th leaves ``reset_cycle`` where it was and is still a
    different window, which a cycle-only comparison would miss.
    """
    return cycle_settings_of(budget)


async def retime_for_budget(db: AsyncSession, budget: Budget, *, budget_id: str) -> None:
    """Rewrite the window on every ceiling naming this budget, and on every user holding it.

    One statement per table rather than a row each: the window is derived from the
    budget and from now, not from anything an individual row holds, so it is the
    same for all of them. Users matter as much as ceilings: a user's reset fires
    only once `next_budget_reset_at` has passed, so a stale date keeps them on the
    old cadence until then, and a user with no date never resets.

    Counters are deliberately untouched, on both, unless the period they count
    had already ended: then they are rolled as a request would have rolled them,
    though without the reset log a request's roll writes.
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
    await retime_budget_holders(db, budget_id, period_start=period_start, period_end=period_end)
