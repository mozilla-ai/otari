"""Read-time increment math for stored metric points (research.md R4).

Points are persisted exactly as OTLP reported them, so turning a series into "how
much happened inside this window" is done here, at read time, not at ingest.
"""

from datetime import UTC, datetime, timedelta

from gateway.services.agent_telemetry_service import compute_series_increment

_T0 = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)


def _at(minutes: int) -> datetime:
    return _T0 + timedelta(minutes=minutes)


def test_delta_series_points_are_summed() -> None:
    points = [(_at(0), 3.0), (_at(1), 2.0), (_at(2), 5.0)]
    assert compute_series_increment(points, "delta") == 10.0


def test_cumulative_series_is_diffed_consecutively() -> None:
    """A running total counts only its growth inside the window, never its level."""
    points = [(_at(0), 10.0), (_at(1), 14.0), (_at(2), 21.0)]
    assert compute_series_increment(points, "cumulative") == 11.0


def test_cumulative_series_ignores_point_order_of_arrival() -> None:
    """Rows come back in whatever order the query yields; the math orders by time."""
    ordered = [(_at(0), 10.0), (_at(1), 14.0), (_at(2), 21.0)]
    shuffled = [ordered[2], ordered[0], ordered[1]]
    assert compute_series_increment(shuffled, "cumulative") == compute_series_increment(ordered, "cumulative")


def test_cumulative_series_tolerates_replayed_points() -> None:
    """A re-exported point is the same instant twice, so it adds no increment."""
    ordered = [(_at(0), 10.0), (_at(1), 14.0), (_at(2), 21.0)]
    replayed = [*ordered, (_at(1), 14.0), (_at(0), 10.0)]
    assert compute_series_increment(replayed, "cumulative") == compute_series_increment(ordered, "cumulative")


def test_cumulative_generations_are_summed_without_a_negative_increment() -> None:
    """A counter reset is a new ``series_start``, so its generation restarts at 0.

    The caller splits by generation before calling this; diffing across the reset
    as one series would subtract the pre-reset total and report a negative.
    """
    before_reset = [(_at(0), 40.0), (_at(1), 46.0)]
    after_reset = [(_at(2), 0.0), (_at(3), 3.0)]
    increments = [
        compute_series_increment(before_reset, "cumulative"),
        compute_series_increment(after_reset, "cumulative"),
    ]
    assert increments == [6.0, 3.0]
    assert sum(increments) == 9.0
    assert all(increment >= 0 for increment in increments)


def test_single_point_and_empty_generations() -> None:
    """One cumulative point shows no growth yet; one delta point is its own increment."""
    assert compute_series_increment([(_at(0), 12.0)], "cumulative") == 0.0
    assert compute_series_increment([(_at(0), 12.0)], "delta") == 12.0
    assert compute_series_increment([], "cumulative") == 0.0
    assert compute_series_increment([], "delta") == 0.0
