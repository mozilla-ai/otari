"""The window each reset cycle occupies, which is the one derivation every capped surface shares."""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.services.budgets._periods import (
    cycle_window,
    mask_from_weekdays,
    weekdays_from_mask,
)

MONDAY = 0
FRIDAY = 4
SUNDAY = 6

# A Wednesday, mid-afternoon, so no case accidentally sits on a boundary.
WEDNESDAY = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)


def window(now: datetime, cycle: str, **settings: object) -> tuple[datetime, datetime]:
    full: dict[str, object] = {
        "every_n": None,
        "anchor_at": None,
        "weekdays": None,
        "month_day": None,
        "month": None,
    }
    full.update(settings)
    return cycle_window(now, cycle=cycle, **full)  # type: ignore[arg-type]


class TestWeekdayMask:
    def test_round_trips_a_set_of_weekdays(self) -> None:
        assert weekdays_from_mask(mask_from_weekdays([MONDAY, FRIDAY])) == (MONDAY, FRIDAY)

    def test_ignores_the_order_it_is_given(self) -> None:
        assert mask_from_weekdays([FRIDAY, MONDAY]) == mask_from_weekdays([MONDAY, FRIDAY])

    def test_holds_every_day_of_the_week(self) -> None:
        assert weekdays_from_mask(mask_from_weekdays(range(7))) == (0, 1, 2, 3, 4, 5, 6)


class TestDaily:
    def test_opens_at_utc_midnight_and_runs_a_day(self) -> None:
        start, end = window(WEDNESDAY, "daily")
        assert start == datetime(2026, 10, 7, tzinfo=UTC)
        assert end == datetime(2026, 10, 8, tzinfo=UTC)

    def test_derives_the_window_from_the_boundary_not_from_now(self) -> None:
        # A budget rolled late still lands on the window it belongs to, so two
        # moments in one day get one window rather than two offset ones.
        early = window(WEDNESDAY.replace(hour=0, minute=1), "daily")
        late = window(WEDNESDAY.replace(hour=23, minute=59), "daily")
        assert early == late


class TestWeekly:
    def test_runs_between_two_selected_days_unevenly(self) -> None:
        # The spec's worked example: Monday and Friday give a four-day period and
        # a three-day one, because the limit applies to each period between the
        # selected days rather than to a week divided up.
        monday_leg = window(WEDNESDAY, "weekly", weekdays=mask_from_weekdays([MONDAY, FRIDAY]))
        assert monday_leg == (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 9, tzinfo=UTC))
        assert monday_leg[1] - monday_leg[0] == timedelta(days=4)

        saturday = datetime(2026, 10, 10, 9, 0, tzinfo=UTC)
        friday_leg = window(saturday, "weekly", weekdays=mask_from_weekdays([MONDAY, FRIDAY]))
        assert friday_leg == (datetime(2026, 10, 9, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC))
        assert friday_leg[1] - friday_leg[0] == timedelta(days=3)

    def test_a_single_day_runs_the_whole_week(self) -> None:
        start, end = window(WEDNESDAY, "weekly", weekdays=mask_from_weekdays([MONDAY]))
        assert end - start == timedelta(days=7)

    def test_opens_on_the_selected_day_itself(self) -> None:
        # From 00:00 on its own day, that day's window is the current one rather
        # than the one that ended at midnight.
        monday = datetime(2026, 10, 5, 0, 0, tzinfo=UTC)
        start, _ = window(monday, "weekly", weekdays=mask_from_weekdays([MONDAY]))
        assert start == monday

    def test_crosses_the_year_on_a_sunday_cycle(self) -> None:
        new_year = datetime(2027, 1, 1, 12, 0, tzinfo=UTC)  # a Friday
        start, end = window(new_year, "weekly", weekdays=mask_from_weekdays([SUNDAY]))
        assert start == datetime(2026, 12, 27, tzinfo=UTC)
        assert end == datetime(2027, 1, 3, tzinfo=UTC)

    def test_refuses_a_mask_naming_no_day(self) -> None:
        with pytest.raises(ValueError, match="names no weekday"):
            window(WEDNESDAY, "weekly", weekdays=0)


class TestMonthly:
    def test_runs_from_the_selected_day_to_the_next(self) -> None:
        start, end = window(WEDNESDAY, "monthly", month_day=15)
        assert start == datetime(2026, 9, 15, tzinfo=UTC)
        assert end == datetime(2026, 10, 15, tzinfo=UTC)

    def test_opens_on_the_day_itself(self) -> None:
        on_the_day = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
        start, _ = window(on_the_day, "monthly", month_day=15)
        assert start == on_the_day

    def test_day_28_survives_february(self) -> None:
        # 28 is the cap for exactly this reason: every month has the day, so a
        # monthly budget rolls twelve times a year rather than seven.
        start, end = window(datetime(2026, 2, 28, 6, 0, tzinfo=UTC), "monthly", month_day=28)
        assert start == datetime(2026, 2, 28, tzinfo=UTC)
        assert end == datetime(2026, 3, 28, tzinfo=UTC)

    def test_rolls_december_into_january(self) -> None:
        start, end = window(datetime(2026, 12, 20, tzinfo=UTC), "monthly", month_day=1)
        assert start == datetime(2026, 12, 1, tzinfo=UTC)
        assert end == datetime(2027, 1, 1, tzinfo=UTC)


class TestYearly:
    def test_runs_from_the_date_to_the_same_date_next_year(self) -> None:
        start, end = window(WEDNESDAY, "yearly", month=3, month_day=1)
        assert start == datetime(2026, 3, 1, tzinfo=UTC)
        assert end == datetime(2027, 3, 1, tzinfo=UTC)

    def test_before_the_date_sits_in_last_year_s_window(self) -> None:
        start, end = window(datetime(2026, 1, 5, tzinfo=UTC), "yearly", month=3, month_day=1)
        assert start == datetime(2025, 3, 1, tzinfo=UTC)
        assert end == datetime(2026, 3, 1, tzinfo=UTC)

    def test_spans_a_leap_day_without_moving(self) -> None:
        start, end = window(datetime(2028, 6, 1, tzinfo=UTC), "yearly", month=1, month_day=10)
        assert start == datetime(2028, 1, 10, tzinfo=UTC)
        assert end == datetime(2029, 1, 10, tzinfo=UTC)


class TestInterval:
    def test_hours_keep_their_phase_from_the_anchor(self) -> None:
        # The whole point of the anchor: a six-hour cycle anchored at 09:00 rolls
        # at 15:00 and 21:00 whether or not anything was spent in between, where
        # the old duration-based period restarted at the next request and walked.
        anchor = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
        start, end = window(WEDNESDAY, "every_n_hours", every_n=6, anchor_at=anchor)
        assert start == datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
        assert end == datetime(2026, 10, 7, 15, 0, tzinfo=UTC)

        later = window(datetime(2026, 10, 7, 20, 0, tzinfo=UTC), "every_n_hours", every_n=6, anchor_at=anchor)
        assert later == (
            datetime(2026, 10, 7, 15, 0, tzinfo=UTC),
            datetime(2026, 10, 7, 21, 0, tzinfo=UTC),
        )

    def test_an_hourly_cycle_does_not_drift_across_many_periods(self) -> None:
        anchor = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        start, end = window(datetime(2026, 10, 7, 7, 30, tzinfo=UTC), "every_n_hours", every_n=1, anchor_at=anchor)
        assert start == datetime(2026, 10, 7, 7, 0, tzinfo=UTC)
        assert end == datetime(2026, 10, 7, 8, 0, tzinfo=UTC)

    def test_days_count_whole_steps_from_the_anchor(self) -> None:
        anchor = datetime(2026, 1, 1, tzinfo=UTC)
        start, end = window(WEDNESDAY, "every_n_days", every_n=14, anchor_at=anchor)
        assert start == datetime(2026, 9, 24, tzinfo=UTC)
        assert end == datetime(2026, 10, 8, tzinfo=UTC)
        assert (start - anchor).days % 14 == 0

    def test_an_anchor_in_the_future_gives_the_first_period(self) -> None:
        anchor = datetime(2027, 1, 1, tzinfo=UTC)
        assert window(WEDNESDAY, "every_n_days", every_n=7, anchor_at=anchor) == (
            anchor,
            datetime(2027, 1, 8, tzinfo=UTC),
        )

    def test_an_anchor_in_another_zone_is_read_as_the_instant_it_names(self) -> None:
        from datetime import timezone

        plus_two = timezone(timedelta(hours=2))
        anchor = datetime(2026, 10, 7, 11, 0, tzinfo=plus_two)  # 09:00 UTC
        start, _ = window(WEDNESDAY, "every_n_hours", every_n=6, anchor_at=anchor)
        assert start == datetime(2026, 10, 7, 9, 0, tzinfo=UTC)

    @pytest.mark.parametrize("cycle", ["every_n_hours", "every_n_days"])
    def test_refuses_an_interval_with_no_anchor(self, cycle: str) -> None:
        with pytest.raises(ValueError, match="no interval or anchor"):
            window(WEDNESDAY, cycle, every_n=6)


class TestVocabulary:
    def test_refuses_a_cycle_this_codebase_does_not_know(self) -> None:
        # Including the vocabulary this replaced, so a row written by an older
        # process is a refusal rather than a silently wrong window.
        with pytest.raises(ValueError, match="Unknown reset cycle"):
            window(WEDNESDAY, "calendar_month")

    def test_every_window_contains_the_moment_it_was_asked_about(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = [
            ("daily", {}),
            ("weekly", {"weekdays": mask_from_weekdays([MONDAY, FRIDAY])}),
            ("monthly", {"month_day": 15}),
            ("yearly", {"month": 3, "month_day": 1}),
            ("every_n_hours", {"every_n": 6, "anchor_at": datetime(2026, 1, 1, tzinfo=UTC)}),
            ("every_n_days", {"every_n": 14, "anchor_at": datetime(2026, 1, 1, tzinfo=UTC)}),
        ]
        for cycle, settings in cases:
            start, end = window(WEDNESDAY, cycle, **settings)
            assert start <= WEDNESDAY < end, cycle
            assert start < end, cycle
