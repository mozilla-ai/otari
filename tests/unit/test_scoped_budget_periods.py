"""Where a scoped ceiling's period window starts and ends.

The enforcement path around this is in ``tests/integration/test_scoped_budgets.py``;
this covers the arithmetic itself, including the boundaries a calendar month makes
awkward (December, February, a leap year), which a rolling window in seconds
cannot express at all.
"""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.services.scoped_budget_service import period_window


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def test_no_cadence_has_no_window() -> None:
    """A ceiling with neither a duration nor an alignment never resets."""
    assert period_window(_at("2026-08-19T09:30:00"), duration=None, alignment=None) is None


def test_a_duration_still_measures_from_now() -> None:
    """The rolling window is unchanged: N seconds from the moment it is rolled."""
    now = _at("2026-08-19T09:30:00")
    assert period_window(now, duration=3600, alignment=None) == (now, now + timedelta(hours=1))


@pytest.mark.parametrize(
    ("alignment", "now", "expected_start", "expected_end"),
    [
        ("calendar_day", "2026-08-19T09:30:00", "2026-08-19T00:00:00", "2026-08-20T00:00:00"),
        # Midnight is the first instant of its own window, not the last of the
        # previous one, so a roll exactly on the boundary opens the new period.
        ("calendar_day", "2026-08-19T00:00:00", "2026-08-19T00:00:00", "2026-08-20T00:00:00"),
        # A Wednesday, so the week runs back to Monday.
        ("calendar_week", "2026-08-19T09:30:00", "2026-08-17T00:00:00", "2026-08-24T00:00:00"),
        # A Monday is the start of its own week, and a Sunday belongs to the week
        # that began six days earlier rather than the one starting tomorrow.
        ("calendar_week", "2026-08-17T00:00:01", "2026-08-17T00:00:00", "2026-08-24T00:00:00"),
        ("calendar_week", "2026-08-23T23:59:59", "2026-08-17T00:00:00", "2026-08-24T00:00:00"),
        ("calendar_month", "2026-08-19T09:30:00", "2026-08-01T00:00:00", "2026-09-01T00:00:00"),
        # A 31-day month, then a 30-day one, then December, whose next boundary is
        # in the following year.
        ("calendar_month", "2026-01-31T23:00:00", "2026-01-01T00:00:00", "2026-02-01T00:00:00"),
        ("calendar_month", "2026-04-30T12:00:00", "2026-04-01T00:00:00", "2026-05-01T00:00:00"),
        ("calendar_month", "2026-12-25T12:00:00", "2026-12-01T00:00:00", "2027-01-01T00:00:00"),
        # February, common and leap: the month's length is never assumed.
        ("calendar_month", "2026-02-28T12:00:00", "2026-02-01T00:00:00", "2026-03-01T00:00:00"),
        ("calendar_month", "2028-02-29T12:00:00", "2028-02-01T00:00:00", "2028-03-01T00:00:00"),
    ],
)
def test_an_alignment_snaps_to_its_utc_boundary(
    alignment: str,
    now: str,
    expected_start: str,
    expected_end: str,
) -> None:
    assert period_window(_at(now), duration=None, alignment=alignment) == (_at(expected_start), _at(expected_end))


def test_a_month_never_spans_a_fixed_number_of_seconds() -> None:
    """The reason the column exists: 2592000 seconds is a different product.

    Twelve calendar months are a year; twelve thirty-day windows are eleven days
    short of one, so every monthly cap expressed in seconds would be about 1.5
    percent more generous than it reads.
    """
    lengths = set()
    for month in range(1, 13):
        window = period_window(_at(f"2026-{month:02d}-15T00:00:00"), duration=None, alignment="calendar_month")
        assert window is not None
        lengths.add(window[1] - window[0])
    assert lengths == {timedelta(days=28), timedelta(days=30), timedelta(days=31)}


def test_an_alignment_is_read_in_utc_whatever_the_caller_holds() -> None:
    """A boundary is UTC, so an aware timestamp in another offset lands on the
    same window as the instant it names."""
    tokyo_morning = datetime.fromisoformat("2026-08-19T08:00:00+09:00")
    assert period_window(tokyo_morning, duration=None, alignment="calendar_day") == (
        _at("2026-08-18T00:00:00"),
        _at("2026-08-19T00:00:00"),
    )


def test_an_unknown_alignment_is_refused_rather_than_guessed() -> None:
    """Only reachable from a write that went around the API, and the caller (the
    roll at the gate) leaves the exhausted window in place instead of guessing."""
    with pytest.raises(ValueError, match="Unknown reset alignment"):
        period_window(_at("2026-08-19T09:30:00"), duration=None, alignment="calendar_quarter")
