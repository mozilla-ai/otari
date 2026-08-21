"""How a budget's reset cadence becomes a window, and the vocabulary for saying it.

A leaf module on purpose. Both the code that enforces a scoped ceiling
(:mod:`gateway.services.scoped_budget_service`) and the code that materializes one
from a workspace default (:mod:`gateway.services.tenancy.workspace_budget_default_service`)
have to derive a window, and the second cannot import the first: that module
imports ``workspace_scope``, which reaches ``tenancy.provisioning_service``,
``tenancy/__init__`` and ``workspace_service``, which imports the default service
back. The cycle is real, and the previous answer to it was a second copy of the
derivation that only understood durations, so a calendar-aligned budget
materialized a ceiling with no window at all and never reset.

Nothing here imports from the gateway, so both sides can depend on it and
``tests/unit/test_service_module_imports.py`` keeps the graph acyclic.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal, get_args

ALIGN_DAY = "calendar_day"
ALIGN_WEEK = "calendar_week"
ALIGN_MONTH = "calendar_month"

# The other way a budget can carry a period, and the one place the wire
# vocabulary is written. A budget holds either a duration in seconds (a rolling
# window measured from the last reset) or one of these (a window snapped to a UTC
# calendar boundary), never both: a CHECK on ``budgets`` refuses the fourth
# state. This is not a period enum, it only says which boundary a reset snaps to,
# and a calendar month is the one no number of seconds can name.
ResetAlignment = Literal["calendar_day", "calendar_week", "calendar_month"]
RESET_ALIGNMENTS: tuple[ResetAlignment, ...] = get_args(ResetAlignment)

# An upper bound on a period length, in seconds (roughly ten years). Without one,
# ``now + timedelta(seconds=...)`` overflows on an arbitrarily large value:
# ``timedelta`` itself refuses more than ``timedelta.max``, so a request near that
# raises ``OverflowError`` (a 500) instead of the 422 an out-of-range period
# should be.
MAX_BUDGET_DURATION_SEC = 10 * 365 * 24 * 3600


def aligned_window(alignment: str, now: datetime) -> tuple[datetime, datetime]:
    """The UTC calendar window containing ``now``.

    Derived from the boundary rather than from ``now`` itself, which is the point:
    a budget rolled late still lands on the window it belongs to, so when anything
    looked at the row stops leaking into the period it gets.
    """
    day = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if alignment == ALIGN_DAY:
        return day, day + timedelta(days=1)
    if alignment == ALIGN_WEEK:
        # ISO weeks, so a week runs Monday 00:00 to the next Monday 00:00.
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=7)
    if alignment == ALIGN_MONTH:
        start = day.replace(day=1)
        end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
        return start, end
    raise ValueError(f"Unknown reset alignment: {alignment!r}")


def period_window(
    now: datetime,
    *,
    duration: int | None,
    alignment: str | None,
) -> tuple[datetime, datetime] | None:
    """The window a budget with this cadence occupies at ``now``, or None for one that never resets.

    The single place a period is derived, so a window written at creation, a
    window written when a ceiling is materialized, a window rewritten by a
    retiming, and a window rolled at the gate cannot drift apart. Raises
    ``ValueError`` for an alignment this codebase does not know.
    """
    if alignment is not None:
        return aligned_window(alignment, now)
    if duration:
        return now, now + timedelta(seconds=duration)
    return None


def budget_window(now: datetime, budget: object) -> tuple[datetime, datetime] | None:
    """The window a ``Budget`` row occupies at ``now``, or None if it never resets.

    A thin read of :func:`period_window` off the two columns, so a caller holding
    a budget cannot accidentally consult one of them and not the other. That is
    the bug this module exists to stop: reading only ``budget_duration_sec``
    silently ignores a calendar cadence, and the row then never resets at all.
    """
    return period_window(
        now,
        duration=getattr(budget, "budget_duration_sec", None),
        alignment=getattr(budget, "reset_alignment", None),
    )


__all__ = [
    "ALIGN_DAY",
    "ALIGN_MONTH",
    "ALIGN_WEEK",
    "MAX_BUDGET_DURATION_SEC",
    "RESET_ALIGNMENTS",
    "ResetAlignment",
    "aligned_window",
    "budget_window",
    "period_window",
]
