"""How a budget's reset cadence becomes a window.

Every surface that caps spend derives its window here, so a budget gets the same window on each.

Two families. A **calendar** cycle snaps to a UTC boundary: the window containing
``now`` is derived from the boundary rather than from ``now``, so a budget rolled
late still lands on the window it belongs to. An **interval** cycle counts a fixed
number of hours or days from an anchor, and the window is the one that interval
lands in, so it keeps its phase instead of walking forward on a quiet workspace.
"""

from __future__ import annotations

import calendar
from collections.abc import Set
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

from gateway.models.budgets import (
    CYCLE_DAILY,
    CYCLE_DAYS,
    CYCLE_HOURS,
    CYCLE_MONTHLY,
    CYCLE_WEEKLY,
    CYCLE_YEARLY,
    MAX_EVERY_N_DAYS,
    MAX_EVERY_N_HOURS,
    WEEKDAY_MASK_MAX,
)

if TYPE_CHECKING:
    from gateway.models.budgets import Budget

_MAX_ANCHOR_AHEAD = timedelta(days=MAX_EVERY_N_DAYS)
_MAX_EVERY_N: dict[str | None, int] = {CYCLE_HOURS: MAX_EVERY_N_HOURS, CYCLE_DAYS: MAX_EVERY_N_DAYS}

DAY = timedelta(days=1)


def weekdays_from_mask(mask: int) -> tuple[int, ...]:
    """The weekdays a mask names, in ``date.weekday()`` order (Monday 0)."""
    return tuple(day for day in range(7) if mask & (1 << day))


def _midnight(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _weekly_window(mask: int, now: datetime) -> tuple[datetime, datetime]:
    """The window between two of the selected weekdays.

    With several days selected the periods are deliberately uneven: Monday and
    Friday give a four-day period and a three-day one, because the limit applies
    to each period between the selected days rather than to a week divided up.
    """
    selected = weekdays_from_mask(mask & WEEKDAY_MASK_MAX)
    if not selected:
        raise ValueError("A weekly reset cycle names no weekday")
    day = _midnight(now)
    # Walk back to the most recent selected weekday, which is today when today is
    # one of them: a budget resets at 00:00 on its day, so from 00:00 onward that
    # day's window is the current one.
    start = day
    while start.weekday() not in selected:
        start -= DAY
    end = start + DAY
    while end.weekday() not in selected:
        end += DAY
    return start, end


def _add_months(moment: datetime, months: int) -> datetime:
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    # Clamped for safety rather than for correctness: `reset_month_day` is capped
    # at 28 by its CHECK, so every month has the day and this never truncates.
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def _monthly_window(month_day: int, now: datetime) -> tuple[datetime, datetime]:
    day = _midnight(now)
    start = day.replace(day=month_day)
    if day.day < month_day:
        start = _add_months(start, -1)
    return start, _add_months(start, 1)


def _yearly_window(month: int, month_day: int, now: datetime) -> tuple[datetime, datetime]:
    day = _midnight(now)
    start = day.replace(month=month, day=month_day)
    if (day.month, day.day) < (month, month_day):
        start = start.replace(year=start.year - 1)
    return start, start.replace(year=start.year + 1)


def _interval_window(step: timedelta, anchor: datetime, now: datetime) -> tuple[datetime, datetime]:
    """The window this interval lands in, counted from the anchor.

    Derived by counting whole steps from the anchor rather than by starting at
    ``now``, which is what keeps the phase: a budget anchored at 09:00 on a
    six-hour cycle rolls at 15:00 and 21:00 whether or not anything was spent in
    between.

    An anchor in the future is a budget whose first period has not opened yet, so
    the window is the first one.
    """
    anchor = anchor.astimezone(UTC)
    if now <= anchor:
        return anchor, anchor + step
    elapsed = now - anchor
    steps = elapsed // step
    start = anchor + steps * step
    return start, start + step


class CycleSettings(NamedTuple):
    """One budget's cadence, as the columns that carry it.

    A tuple rather than six parameters threaded through a caller, so a query that
    selects the cycle cannot select some of its settings and not the others: the
    bug this module exists to stop is a window derived from half a cadence.
    """

    cycle: str | None
    every_n: int | None
    anchor_at: datetime | None
    weekdays: int | None
    month_day: int | None
    month: int | None

    def window(self, now: datetime) -> tuple[datetime, datetime] | None:
        """The window these settings occupy at ``now``, or None for "never"."""
        if self.cycle is None:
            return None
        return cycle_window(
            now,
            cycle=self.cycle,
            every_n=self.every_n,
            anchor_at=self.anchor_at,
            weekdays=self.weekdays,
            month_day=self.month_day,
            month=self.month,
        )


# What each cycle carries, and therefore what it must not carry. The model's
# CHECKs enforce the same thing, and this is the half that answers a 4xx instead of
# letting a write reach them and come back as a 500.
CYCLE_SETTING_NAMES: dict[str | None, frozenset[str]] = {
    None: frozenset(),
    CYCLE_HOURS: frozenset({"reset_every_n", "reset_anchor_at"}),
    CYCLE_DAYS: frozenset({"reset_every_n", "reset_anchor_at"}),
    CYCLE_DAILY: frozenset(),
    CYCLE_WEEKLY: frozenset({"reset_weekdays"}),
    CYCLE_MONTHLY: frozenset({"reset_month_day"}),
    CYCLE_YEARLY: frozenset({"reset_month", "reset_month_day"}),
}

ALL_SETTING_NAMES: frozenset[str] = frozenset().union(*CYCLE_SETTING_NAMES.values())

# The cadence's wire fields, in `CycleSettings` order, so a caller can settle an
# update field by field without respelling the list and getting it out of step
# with the tuple it builds.
CYCLE_FIELD_ORDER: tuple[str, ...] = (
    "reset_cycle",
    "reset_every_n",
    "reset_anchor_at",
    "reset_weekdays",
    "reset_month_day",
    "reset_month",
)
CYCLE_FIELDS: frozenset[str] = frozenset(CYCLE_FIELD_ORDER)

# Wire name to the `CycleSettings` field holding it.
_FIELD_OF = {
    "reset_every_n": "every_n",
    "reset_anchor_at": "anchor_at",
    "reset_weekdays": "weekdays",
    "reset_month_day": "month_day",
    "reset_month": "month",
}


def settle_cycle(stored: CycleSettings, submitted: CycleSettings, submitted_names: Set[str]) -> CycleSettings:
    """The cadence an update leaves behind, given what it named and what is stored.

    An omitted field contributes what is stored, *unless the cycle itself
    changed*: then the settings of the cycle being left behind are cleared rather
    than carried onto a cycle that does not take them. Without that rule the
    obvious request is the one that fails, because a budget moved from monthly to
    daily keeps a day-of-month nobody asked to keep and the write is refused for
    a field the caller never mentioned.

    Clearing rather than ignoring, because a stray setting is not harmless: it is
    what the budget reverts to on the next switch back, silently.
    """
    changing = "reset_cycle" in submitted_names and submitted.cycle != stored.cycle
    base = CycleSettings(submitted.cycle, None, None, None, None, None) if changing else stored
    return CycleSettings(
        *(
            getattr(submitted, field) if name in submitted_names else getattr(base, field)
            for name, field in zip(CYCLE_FIELD_ORDER, CycleSettings._fields, strict=True)
        )
    )


def validate_cycle_settings(settings: CycleSettings) -> None:
    """Refuse a cadence carrying the wrong settings for its cycle.

    Raises ``ValueError`` naming what is missing or stray. Both directions
    matter, and the stray one is the less obvious: leaving a weekday mask on a
    budget switched to monthly is a setting nobody can see on the surface that
    stored it, and the row it makes is one the CHECKs refuse, so the write comes
    back a 500 rather than saying which field was wrong.
    """
    cycle = settings.cycle
    if cycle is not None and cycle not in CYCLE_SETTING_NAMES:
        raise ValueError(f"Unknown reset cycle: {cycle!r}")
    required = CYCLE_SETTING_NAMES[cycle]
    # `CycleSettings` spells its fields without the `reset_` the wire uses, so the
    # sets are compared on the wire names and read through the short ones.
    present = {name for name in ALL_SETTING_NAMES if getattr(settings, _FIELD_OF[name]) is not None}
    missing = sorted(required - present)
    stray = sorted(present - required)
    if missing:
        raise ValueError(f"A {cycle or 'never-resetting'} budget needs {', '.join(missing)}")
    if stray:
        raise ValueError(f"A {cycle or 'never-resetting'} budget does not take {', '.join(stray)}")
    ceiling = _MAX_EVERY_N.get(cycle)
    if ceiling is not None and settings.every_n is not None and settings.every_n > ceiling:
        raise ValueError(f"reset_every_n is at most {ceiling} for {cycle}")
    # Far enough out to name any real start date, and short of the `timedelta`
    # overflow an anchor near year 9999 would raise on every request it gates.
    if settings.anchor_at is not None and settings.anchor_at > datetime.now(UTC) + _MAX_ANCHOR_AHEAD:
        raise ValueError("reset_anchor_at is at most ten years from now")


def cycle_window(
    now: datetime,
    *,
    cycle: str,
    every_n: int | None,
    anchor_at: datetime | None,
    weekdays: int | None,
    month_day: int | None,
    month: int | None,
) -> tuple[datetime, datetime]:
    """The window a budget on this cycle occupies at ``now``.

    Raises ``ValueError`` for a cycle this codebase does not know, and for one
    whose settings are absent. Both are states the table's CHECKs refuse, so
    reaching either means a row was written around them.
    """
    if cycle == CYCLE_DAILY:
        day = _midnight(now)
        return day, day + DAY
    if cycle == CYCLE_WEEKLY:
        if weekdays is None:
            raise ValueError("A weekly reset cycle carries no weekdays")
        return _weekly_window(weekdays, now)
    if cycle == CYCLE_MONTHLY:
        if month_day is None:
            raise ValueError("A monthly reset cycle carries no day")
        return _monthly_window(month_day, now)
    if cycle == CYCLE_YEARLY:
        if month is None or month_day is None:
            raise ValueError("A yearly reset cycle carries no date")
        return _yearly_window(month, month_day, now)
    if cycle in (CYCLE_HOURS, CYCLE_DAYS):
        if every_n is None or anchor_at is None:
            raise ValueError(f"An interval reset cycle ({cycle}) carries no interval or anchor")
        step = timedelta(hours=every_n) if cycle == CYCLE_HOURS else timedelta(days=every_n)
        return _interval_window(step, anchor_at, now)
    raise ValueError(f"Unknown reset cycle: {cycle!r}")


def budget_window(now: datetime, budget: Budget) -> tuple[datetime, datetime] | None:
    """The window a ``Budget`` row occupies at ``now``, or None if it never resets.

    A thin read of :func:`cycle_window` off the cycle's columns, so a caller
    holding a budget cannot consult some of them and not the others. That is the
    bug this module exists to stop: reading one column and ignoring the rest
    silently gives a budget a cadence nobody set.
    """
    return cycle_settings_of(budget).window(now)


def cycle_settings_of(budget: Budget) -> CycleSettings:
    """The cadence a ``Budget`` row carries, read off every one of its columns."""
    return CycleSettings(*(getattr(budget, name) for name in CYCLE_FIELD_ORDER))


__all__ = [
    "ALL_SETTING_NAMES",
    "CYCLE_FIELDS",
    "CYCLE_FIELD_ORDER",
    "CYCLE_SETTING_NAMES",
    "CycleSettings",
    "budget_window",
    "cycle_settings_of",
    "cycle_window",
    "settle_cycle",
    "validate_cycle_settings",
    "weekdays_from_mask",
]
