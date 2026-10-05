"""Move a budget's ceilings onto the cadence it now carries.

A ceiling stores its own window and reads the cadence through its budget.
A change to the budget's period leaves the two in disagreement until the ceilings are retimed.
A ceiling left with no window never rolls, so its spend accumulates while the budget reports a cadence.
"""

from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.budgets import ScopedBudget
from gateway.services.budgets._periods import budget_window

__all__ = ["cadence_of", "retime_ceilings_for_budget"]


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


async def retime_ceilings_for_budget(db: AsyncSession, budget: object, *, budget_id: str) -> None:
    """Rewrite the window on every ceiling naming this budget.

    One statement rather than a row per ceiling: the window is derived from the
    budget and from now, not from anything an individual ceiling holds, so it is
    the same for all of them.

    Counters are deliberately untouched. Spend already recorded stays, matching
    what re-pointing a ceiling at a different budget does: the ceiling is the same
    allowance held to a different figure from here on, not a fresh one.
    ``reserved_spend`` is left alone too, so a hold taken before the change is
    still released against the counter it came from.

    Not committed here. The caller owns the transaction, so a write that is
    refused afterwards takes the retiming back with it.

    A cadence of neither kind clears the window rather than deriving one, which is
    what "no reset" means and what ``budget_window`` returns None for.
    """
    window = budget_window(datetime.now(UTC), budget)
    period_start, period_end = window if window is not None else (None, None)
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.budget_id == budget_id)
        .values(period_start=period_start, period_end=period_end)
        .execution_options(synchronize_session=False)
    )
