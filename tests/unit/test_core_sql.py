"""Unit tests for the cross-layer SQL helpers in `gateway.core.sql`."""

from datetime import UTC, datetime, timedelta, timezone

from gateway.core.sql import utc_bound


def test_naive_bound_is_pinned_to_utc() -> None:
    """An offset-less ISO bound would otherwise resolve against the process's local
    timezone, so the same query would select different rows per deployment."""
    assert utc_bound(datetime(2026, 8, 12, 8, 0)) == datetime(2026, 8, 12, 8, 0, tzinfo=UTC)


def test_aware_bound_is_left_alone() -> None:
    offset = timezone(timedelta(hours=-4))
    aware = datetime(2026, 8, 12, 8, 0, tzinfo=offset)
    assert utc_bound(aware) is aware


def test_missing_bound_stays_missing() -> None:
    assert utc_bound(None) is None
