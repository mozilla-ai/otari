"""When an install sends its update check: its own time of day, and a jittered backoff after a failure."""

import random
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from gateway.core.settings.update_check import UpdateCheckSettings
from gateway.services.update_check._schedule import (
    LONGEST_RETRY,
    daily_slot,
    next_daily_check,
    next_retry,
    release_version,
)
from gateway.services.update_check._service import preset_answer

_NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("v0.30.1", (0, 30, 1)), ("0.30.1", (0, 30, 1)), (" v1.2.10 ", (1, 2, 10))],
)
def test_a_release_tag_reads_as_its_version(raw: str, expected: tuple[int, int, int]) -> None:
    assert release_version(raw) == expected


@pytest.mark.parametrize("raw", ["0.0.0-dev", "5ce3fde79a1b2c3d", "", "v1.2", "1.2.3rc1"])
def test_a_development_build_has_no_release_version(raw: str) -> None:
    assert release_version(raw) is None


def test_an_install_keeps_its_time_of_day() -> None:
    install = uuid.uuid4()
    first = next_daily_check(install, _NOW)
    assert first > _NOW
    assert first - _NOW <= timedelta(days=1)
    assert next_daily_check(install, first) == first + timedelta(days=1)
    assert next_daily_check(install, _NOW + timedelta(hours=3)) in {first, first + timedelta(days=1)}


def test_installs_spread_over_the_whole_day() -> None:
    hours = {int(daily_slot(uuid.uuid4()).total_seconds() // 3600) for _ in range(2000)}
    assert hours == set(range(24))


def test_a_retry_waits_longer_after_each_failure_up_to_six_hours() -> None:
    install = uuid.UUID(int=(12 * 3600 - 1))  # checks at 11:59:59, so the daily cap is a day away
    rng = random.Random(1)
    waits = [next_retry(install, _NOW, failures, rng=rng) - _NOW for failures in (1, 2, 3, 1000)]
    assert timedelta(minutes=2, seconds=30) <= waits[0] <= timedelta(minutes=7, seconds=30)
    assert waits[0] < waits[2]
    assert waits[3] <= LONGEST_RETRY * 1.5


def test_retries_after_a_shared_outage_do_not_land_together() -> None:
    install = uuid.UUID(int=(12 * 3600 - 1))
    rng = random.Random(7)
    retries = {next_retry(install, _NOW, 3, rng=rng) for _ in range(50)}
    assert len(retries) == 50


def test_a_servers_retry_after_is_a_floor() -> None:
    install = uuid.UUID(int=(12 * 3600 - 1))
    retry = next_retry(install, _NOW, 1, retry_after=timedelta(hours=2), rng=random.Random(1))
    assert retry - _NOW >= timedelta(hours=2)


def test_a_retry_is_never_later_than_the_next_daily_check() -> None:
    install = uuid.UUID(int=(12 * 3600 + 60))  # checks at 12:01, a minute from now
    assert next_retry(install, _NOW, 10, rng=random.Random(1)) == next_daily_check(install, _NOW)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    return monkeypatch


def test_with_nothing_configured_an_operator_is_asked(clean_env: pytest.MonkeyPatch) -> None:
    assert preset_answer(UpdateCheckSettings()) is None


@pytest.mark.parametrize(("name", "value"), [("DO_NOT_TRACK", "1"), ("DO_NOT_TRACK", "true"), ("CI", "true")])
def test_do_not_track_and_ci_answer_no(clean_env: pytest.MonkeyPatch, name: str, value: str) -> None:
    clean_env.setenv(name, value)
    assert preset_answer(UpdateCheckSettings()) is False


def test_update_check_in_config_wins_over_do_not_track(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DO_NOT_TRACK", "1")
    assert preset_answer(UpdateCheckSettings(update_check=True)) is True
