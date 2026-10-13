"""The ``Retry-After`` a budget refusal carries: seconds until the refusing budget's window ends."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.api.routes._pipeline import spend_ceiling_refusal
from gateway.core.error_codes import BUDGET_EXCEEDED, error_headers
from gateway.exceptions.budget_exceptions import SpendCeilingExceededError
from gateway.services.budgets._periods import period_window, seconds_until
from gateway.services.budgets._refusals import BlockedAxis, blocked_on

NOW = datetime(2026, 10, 13, 15, 30, 0, 250_000, tzinfo=UTC)


def test_seconds_until_rounds_a_fraction_up() -> None:
    assert seconds_until(NOW + timedelta(seconds=10, milliseconds=1), NOW) == 11
    assert seconds_until(NOW + timedelta(seconds=10), NOW) == 10


def test_seconds_until_is_never_less_than_one() -> None:
    """A window that ended a moment ago and has not rolled yet still asks for a wait, not zero."""
    assert seconds_until(NOW, NOW) == 1
    assert seconds_until(NOW - timedelta(seconds=30), NOW) == 1


def test_seconds_until_reads_a_naive_end_as_utc() -> None:
    """SQLite hands a stored timestamp back naive."""
    assert seconds_until((NOW + timedelta(seconds=60)).replace(tzinfo=None), NOW) == 60


def test_calendar_windows_end_on_their_boundaries() -> None:
    day = period_window(NOW, duration=None, alignment="calendar_day")
    week = period_window(NOW, duration=None, alignment="calendar_week")
    month = period_window(NOW, duration=None, alignment="calendar_month")
    assert day is not None and week is not None and month is not None
    # 2026-10-13 is a Tuesday: the ISO week ends Monday 2026-10-19.
    assert seconds_until(day[1], NOW) == 8 * 3600 + 30 * 60
    assert seconds_until(week[1], NOW) == 5 * 86400 + 8 * 3600 + 30 * 60
    assert seconds_until(month[1], NOW) == 18 * 86400 + 8 * 3600 + 30 * 60


def test_a_refusal_waits_for_its_window_to_end() -> None:
    blocked = BlockedAxis("budget", resets_at=NOW + timedelta(minutes=30))
    assert blocked.retry_after(NOW) == 1800


def test_a_budget_that_never_resets_gives_no_wait() -> None:
    assert BlockedAxis("budget").retry_after(NOW) is None


def test_a_hold_larger_than_the_cap_gives_no_wait() -> None:
    """A fresh window refuses it too, so a wait would only send the client back to the same refusal."""
    blocked = blocked_on("budget", committed=Decimal(0), cap=Decimal(1), wanted=Decimal(2), new_request=True)
    assert blocked is not None
    assert blocked.exceeds_cap
    assert blocked.remaining is not None
    assert BlockedAxis("budget", exceeds_cap=True, resets_at=NOW + timedelta(hours=1)).retry_after(NOW) is None


def test_a_spent_axis_does_not_exceed_its_cap() -> None:
    blocked = blocked_on("token", committed=Decimal(100), cap=Decimal(100), wanted=Decimal(10), new_request=True)
    assert blocked is not None
    assert not blocked.exceeds_cap


def test_error_headers_carry_retry_after_only_when_given() -> None:
    assert "Retry-After" not in error_headers(BUDGET_EXCEEDED, budget_scope="user")
    assert error_headers(BUDGET_EXCEEDED, budget_scope="user", retry_after=42)["Retry-After"] == "42"


def test_a_reservation_move_refusal_carries_its_wait() -> None:
    """The ceiling that refuses a fallback candidate's move answers with the same header as one refusing admission."""
    refusal = spend_ceiling_refusal(SpendCeilingExceededError("no room", scope_type="workspace", retry_after=42))
    assert refusal.headers is not None
    assert refusal.headers["Retry-After"] == "42"
    assert refusal.headers["Otari-Budget-Scope"] == "workspace"

    without = spend_ceiling_refusal(SpendCeilingExceededError("no room", scope_type="workspace"))
    assert without.headers is not None
    assert "Retry-After" not in without.headers
