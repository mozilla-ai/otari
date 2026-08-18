"""Unit tests for the budget migration preflight's pure half.

Duration rounding and the shape of the report are decided without a database, so
they are tested without one. `tests/integration/test_budget_migration_report.py`
covers the query and the CLI command against a real schema.
"""

import json

import pytest

from gateway.services.budget_migration_report import (
    PERIOD_SECONDS,
    AttachedUser,
    BudgetMigrationReport,
    BudgetPlan,
    map_duration,
    render_text,
)

_DAY = 86_400


def _user(user_id: str, *, spend: float = 0.0, reserved: float = 0.0, deleted: bool = False) -> AttachedUser:
    return AttachedUser(user_id=user_id, spend=spend, reserved=reserved, is_deleted=deleted)


@pytest.mark.parametrize(
    ("duration_sec", "period"),
    [(_DAY, "daily"), (7 * _DAY, "weekly"), (30 * _DAY, "monthly")],
)
def test_exact_durations_map_without_drift(duration_sec: int, period: str) -> None:
    mapping = map_duration(duration_sec)

    assert mapping.period == period
    assert mapping.is_exact
    assert mapping.drift_sec == 0
    assert mapping.rate_factor == 1.0


@pytest.mark.parametrize(
    ("duration_sec", "period"),
    [
        (3_600, "daily"),
        (3 * _DAY, "daily"),
        (5 * _DAY, "weekly"),
        (14 * _DAY, "weekly"),
        (60 * _DAY, "monthly"),
    ],
)
def test_inexact_durations_round_to_the_nearest_period(duration_sec: int, period: str) -> None:
    mapping = map_duration(duration_sec)

    assert mapping.period == period
    assert not mapping.is_exact
    assert mapping.drift_sec == PERIOD_SECONDS[period] - duration_sec


def test_an_exact_tie_rounds_to_the_longer_period() -> None:
    # Four days sits exactly between daily and weekly. The longer period is the
    # one that cannot let a user outspend what the gateway allowed them.
    mapping = map_duration(4 * _DAY)

    assert mapping.period == "weekly"
    assert mapping.rate_factor < 1.0


def test_rate_factor_reports_the_direction_of_the_change() -> None:
    # A three-day cap becoming daily lets the same money be spent three times as fast.
    assert map_duration(3 * _DAY).rate_factor == pytest.approx(3.0)
    # A five-day cap becoming weekly stretches the same money over longer.
    assert map_duration(5 * _DAY).rate_factor == pytest.approx(5 / 7)


def test_shared_budget_counts_only_live_users() -> None:
    plan = BudgetPlan(
        budget_id="b-shared",
        name="Team",
        max_budget=100.0,
        duration_sec=_DAY,
        attached=[_user("alice"), _user("bob", deleted=True)],
    )

    # A soft deleted user migrates as a deactivated identity with no live cap, so
    # one live user plus one deleted user is not a shared budget.
    assert not plan.is_shared_pool
    assert [user.user_id for user in plan.live_attached] == ["alice"]


def test_a_budget_without_a_duration_has_no_mapping() -> None:
    plan = BudgetPlan(budget_id="b", name=None, max_budget=10.0, duration_sec=None, attached=[_user("alice")])

    assert plan.mapping is None


def _report() -> BudgetMigrationReport:
    return BudgetMigrationReport(
        budgets=[
            BudgetPlan("b-daily", "Daily", 100.0, _DAY, [_user("alice", spend=1.0)]),
            BudgetPlan("b-sprint", "Sprint", 250.0, 3 * _DAY, [_user("bob"), _user("carol", reserved=2.0)]),
            BudgetPlan("b-forever", None, 50.0, None, [_user("dave", deleted=True)]),
            BudgetPlan("b-orphan", "Orphan", 10.0, 7 * _DAY, []),
        ],
        reset_log_count=12,
    )


def test_report_partitions_every_budget() -> None:
    report = _report()

    assert [plan.budget_id for plan in report.rounded] == ["b-sprint"]
    assert [plan.budget_id for plan in report.exact] == ["b-daily", "b-orphan"]
    assert [plan.budget_id for plan in report.periodless] == ["b-forever"]
    assert [plan.budget_id for plan in report.shared_pools] == ["b-sprint"]
    # b-forever's only user is soft deleted, so nothing live points at it either.
    assert [plan.budget_id for plan in report.unattached] == ["b-forever", "b-orphan"]
    assert report.member_budgets_to_create == 3
    assert report.deleted_attachments == 1


def test_report_json_is_serializable_and_carries_the_decisions() -> None:
    payload = json.loads(json.dumps(_report().to_dict()))

    assert payload["summary"] == {
        "budgets": 4,
        "unattached_budgets": 2,
        "member_budgets_to_create": 3,
        "deleted_attachments": 1,
        "rounded_durations": 1,
        "exact_durations": 2,
        "budgets_without_a_duration": 1,
        "shared_pools": 1,
        "reset_logs_archived": 12,
    }
    (rounded,) = payload["rounded_durations"]
    assert rounded["period"] == "daily"
    assert rounded["rate_factor"] == 3.0
    (shared,) = payload["shared_pools"]
    assert [user["user_id"] for user in shared["attached_users"]] == ["bob", "carol"]


def test_render_text_names_every_budget_needing_a_decision() -> None:
    text = render_text(_report())

    assert "Rounded durations (1)" in text
    assert "3d -> daily" in text
    assert "rate 3.00x LOOSER" in text
    assert "Budgets with no duration (1)" in text
    assert "Shared budgets (1)" in text
    assert "bob: spend 0.0" in text
    assert "12 row(s) in budget_reset_logs are archived" in text
    # Calendar re-anchoring hits every budget, not only the rounded ones, so it is
    # stated even when nothing rounds.
    assert "re-anchored" in text


def test_render_text_says_so_when_there_is_nothing_to_decide() -> None:
    text = render_text(BudgetMigrationReport(budgets=[], reset_log_count=0))

    assert "Every duration maps onto a period exactly." in text
    assert "Every attached budget has exactly one user." in text
    assert "Unattached budgets" not in text


def test_render_text_holds_back_the_tail_of_a_very_large_pool() -> None:
    report = BudgetMigrationReport(
        budgets=[BudgetPlan("b-big", "Everyone", 5.0, _DAY, [_user(f"u{index}") for index in range(25)])],
        reset_log_count=0,
    )

    text = render_text(report)

    assert "u0: spend 0.0" in text
    assert "u9: spend 0.0" in text
    assert "u10: spend 0.0" not in text
    # The count is stated rather than the listing silently ending.
    assert "... and 15 more, listed in full by --json" in text
