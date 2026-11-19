"""How a budget's reset cadence becomes a window.

Every surface that caps spend derives its window here, so a calendar-aligned budget gets the same window on each.
"""

import math
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gateway.models.budgets import ALIGN_DAY, ALIGN_MONTH, ALIGN_WEEK

# An upper bound on a period length, in seconds (roughly ten years). Without one,
# ``now + timedelta(seconds=...)`` overflows on an arbitrarily large value:
# ``timedelta`` itself refuses more than ``timedelta.max``, so a request near that
# raises ``OverflowError`` (a 500) instead of the 422 an out-of-range period
# should be.
MAX_BUDGET_DURATION_SEC = 10 * 365 * 24 * 3600


def aligned_window(alignment: str, now: datetime, timezone: str | None = None) -> tuple[datetime, datetime]:
    """The calendar window containing ``now``, in ``timezone`` (UTC when None), as UTC instants.

    Derived from the boundary rather than from ``now`` itself, which is the point:
    a budget rolled late still lands on the window it belongs to, so when anything
    looked at the row stops leaking into the period it gets.

    The boundaries are local midnights, so across a daylight saving change a day
    runs 23 or 25 hours. The calendar arithmetic is done on naive local dates and
    each boundary is converted once: adding a ``timedelta`` to an aware local time
    would keep its offset and land an hour off midnight.

    Raises ``ValueError`` for an alignment or a timezone this deployment does not
    know, which a write could only have stored around the API.
    """
    try:
        zone = ZoneInfo(timezone) if timezone else UTC
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown reset timezone: {timezone!r}") from None
    today = now.astimezone(zone).date()
    if alignment == ALIGN_DAY:
        start, end = today, today + timedelta(days=1)
    elif alignment == ALIGN_WEEK:
        # ISO weeks, so a week runs Monday 00:00 to the next Monday 00:00.
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=7)
    elif alignment == ALIGN_MONTH:
        start = today.replace(day=1)
        end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    else:
        raise ValueError(f"Unknown reset alignment: {alignment!r}")
    return _midnight(start, zone), _midnight(end, zone)


def _midnight(day: date, zone: tzinfo) -> datetime:
    """The UTC instant ``day`` starts in ``zone``.

    A zone that skips midnight itself (a few move their clocks at 00:00) has no
    such wall time, and ``zoneinfo`` reads the gap as the earlier offset, which
    lands on the first instant the day exists.
    """
    return datetime.combine(day, time(), zone).astimezone(UTC)


def period_window(
    now: datetime,
    *,
    duration: int | None,
    alignment: str | None,
    timezone: str | None = None,
) -> tuple[datetime, datetime] | None:
    """The window a budget with this cadence occupies at ``now``, or None for one that never resets.

    The single place a period is derived, so a window written at creation, a
    window written when a ceiling is materialized, a window rewritten by a
    retiming, and a window rolled at the gate cannot drift apart. Raises
    ``ValueError`` for an alignment or a timezone this deployment does not know.
    """
    if alignment is not None:
        return aligned_window(alignment, now, timezone)
    if duration:
        return now, now + timedelta(seconds=duration)
    return None


def budget_window(now: datetime, budget: object) -> tuple[datetime, datetime] | None:
    """The window a ``Budget`` row occupies at ``now``, or None if it never resets.

    A thin read of :func:`period_window` off the two columns, so a caller holding
    a budget cannot accidentally consult one of them and not the other. That is
    the bug this module exists to stop: reading only ``budget_duration_sec``
    silently ignores a calendar cadence, and the row then never resets at all.
    ``reset_timezone`` is read with them for the same reason.
    """
    return period_window(
        now,
        duration=getattr(budget, "budget_duration_sec", None),
        alignment=getattr(budget, "reset_alignment", None),
        timezone=getattr(budget, "reset_timezone", None),
    )


def seconds_until(end: datetime, now: datetime) -> int:
    """Whole seconds from ``now`` until ``end``, rounded up and never less than one.

    Rounded up so a client that waits this long lands in the next window rather
    than a fraction of a second before it.
    """
    return max(math.ceil((end - now).total_seconds()), 1)


__all__ = [
    "MAX_BUDGET_DURATION_SEC",
    "aligned_window",
    "budget_window",
    "period_window",
    "seconds_until",
]
