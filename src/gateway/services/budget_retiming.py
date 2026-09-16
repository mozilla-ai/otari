"""Move a budget's ceilings onto the cadence it now carries.

A leaf module for the same reason :mod:`gateway.services.budget_periods` is one,
and it sits directly on top of it. Two surfaces change a budget's period, and
neither can import the other's module:

- ``api/routes/budgets.py``, the deployment-wide surface.
- ``services/tenancy/organization_budget_service.py``, the tenant-scoped one,
  which cannot reach ``scoped_budget_service`` because that module imports
  ``workspace_scope`` and closes a cycle back through ``tenancy/__init__``.

Without a shared home the retiming would be written twice, and a rule with two
copies is a rule that will hold on one surface and not the other. This imports
the entity models and ``budget_periods`` only, so nothing that depends on it
gains a cycle, and ``tests/unit/test_service_module_imports.py`` pins that.

**Why retiming is necessary at all.** A ceiling holds its own
``period_start``/``period_end`` and reads the cadence *through* the budget it
names, so editing the budget's period leaves the two disagreeing. One direction
is an enforcement bug rather than a cosmetic one:
``scoped_budget_service._roll_expired_periods`` only ever updates a row whose
``period_end`` is not null (that guard is what makes the roll a lock-free
compare-and-swap at the boundary), so a budget moved from "no reset" to a
periodic cadence leaves its ceilings with NULL windows that never roll at all,
accumulating spend forever while the API reports the new cadence. The reverse
leaves a stale boundary that fires exactly once, at an arbitrary moment, zeroing
counters for no reason anyone can point at.
"""

from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.budgets import ScopedBudget
from gateway.services.budget_periods import period_window

__all__ = ["cadence_of", "retime_ceilings_for_budget"]


def cadence_of(duration: int | None, alignment: str | None) -> tuple[int | None, str | None]:
    """The pair that decides whether a retiming is needed, as one comparable value.

    Exists so both callers compare the same thing, read before and after the
    mutation. Keyed on the cadence rather than on "an update happened", because
    retiming on every write would restart a period for a rename or a limit
    change, throwing away the part of it a ceiling had already spent.
    """
    return (duration, alignment)


async def retime_ceilings_for_budget(
    db: AsyncSession,
    *,
    budget_id: str,
    duration: int | None,
    alignment: str | None,
) -> None:
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
    what "no reset" means and what ``period_window`` returns None for.
    """
    window = period_window(datetime.now(UTC), duration=duration, alignment=alignment)
    period_start, period_end = window if window is not None else (None, None)
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.budget_id == budget_id)
        .values(period_start=period_start, period_end=period_end)
        .execution_options(synchronize_session=False)
    )
