"""How long each model has been failing, read from the runs that are kept."""

import uuid
from datetime import UTC, datetime, timedelta

from gateway.models.health_checks import HealthCheckResult, HealthCheckRun
from gateway.services.health_checks._summary import FailureStreak, failure_streaks

_START = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)


def _runs(count: int) -> list[HealthCheckRun]:
    return [HealthCheckRun(id=uuid.uuid4(), started_at=_START + timedelta(hours=i)) for i in range(count)]


def _result(run: HealthCheckRun, selector: str, status: str) -> HealthCheckResult:
    return HealthCheckResult(run_id=run.id, selector=selector, provider=selector.partition(":")[0], status=status)


def test_a_streak_starts_after_the_last_answer() -> None:
    runs = _runs(4)
    results = [
        _result(runs[0], "acme:a", "fail"),
        _result(runs[1], "acme:a", "pass"),
        _result(runs[2], "acme:a", "fail"),
        _result(runs[3], "acme:a", "fail"),
    ]

    assert failure_streaks(runs, results) == {"acme:a": FailureStreak(runs[2].started_at, 2)}


def test_a_run_that_left_a_model_out_does_not_break_its_streak() -> None:
    runs = _runs(3)
    results = [
        _result(runs[0], "mistral:r", "fail"),
        _result(runs[1], "mistral:r", "skip"),
        _result(runs[2], "mistral:r", "fail"),
    ]

    assert failure_streaks(runs, results) == {"mistral:r": FailureStreak(runs[0].started_at, 2)}


def test_a_model_whose_latest_check_answered_has_no_streak() -> None:
    runs = _runs(2)

    assert failure_streaks(runs, [_result(runs[0], "acme:a", "fail"), _result(runs[1], "acme:a", "pass")]) == {}
