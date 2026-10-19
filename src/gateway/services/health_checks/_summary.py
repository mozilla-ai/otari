"""Results folded into one line per provider. Pure, so it is tested without a database."""

import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from gateway.models.health_checks import HealthCheckResult, HealthCheckRun
from gateway.schemas.health_checks import ProviderHealthSummary, ProviderHistoryPoint


def without_model(result: HealthCheckResult) -> str:
    """The error with the model's id taken out, so one outage reads as one error across models."""
    detail = result.error or ""
    _instance, _, model_id = result.selector.partition(":")
    for spelling in {result.selector, model_id} - {""}:
        detail = detail.replace(spelling, "<model>")
    return detail


def _status(answered: int, total: int) -> Literal["ok", "degraded", "down"]:
    if answered == total:
        return "ok"
    return "down" if answered == 0 else "degraded"


def summarize(
    latest: Sequence[HealthCheckResult],
    history_runs: Sequence[HealthCheckRun],
    tallies: Mapping[tuple[str, uuid.UUID], tuple[int, int]],
) -> list[ProviderHealthSummary]:
    """One summary per provider in the latest results, worst first, each with its history.

    ``tallies`` is ``(answered, total)`` per provider and run. A skipped
    offering was not asked, so it counts toward neither side.
    """
    by_provider: dict[str, list[HealthCheckResult]] = defaultdict(list)
    for result in latest:
        if result.status != "skip":
            by_provider[result.provider].append(result)

    summaries: list[ProviderHealthSummary] = []
    for provider, results in by_provider.items():
        failed = [result for result in results if result.status == "fail"]
        answered = len(results) - len(failed)
        common = Counter(without_model(r) for r in failed).most_common(1)
        top_error, top_count = common[0] if common else (None, 0)
        history = [
            ProviderHistoryPoint(
                started_at=run.started_at,
                answered=tallies[(provider, run.id)][0],
                total=tallies[(provider, run.id)][1],
            )
            for run in history_runs
            if (provider, run.id) in tallies
        ]
        summaries.append(
            ProviderHealthSummary(
                provider=provider,
                status=_status(answered, len(results)),
                answered=answered,
                total=len(results),
                top_error=top_error,
                top_error_count=top_count,
                history=history,
            )
        )
    return sorted(summaries, key=lambda s: (s.answered / s.total if s.total else 1.0, s.provider))


@dataclass(frozen=True)
class FailureStreak:
    """How long a model has failed with no answer in between."""

    since: datetime
    runs: int
    # The streak's newest failure, which says what the model last answered.
    last: HealthCheckResult | None = field(default=None, compare=False)


def failure_streaks(runs: Sequence[HealthCheckRun], results: Sequence[HealthCheckResult]) -> dict[str, FailureStreak]:
    """For each model whose latest check failed, the streak of failures that check ends.

    ``runs`` is oldest first. A run that did not check a model, or skipped it,
    neither ends its streak nor adds to it. A streak can reach no further back
    than the oldest run still kept.
    """
    by_run: dict[uuid.UUID, list[HealthCheckResult]] = defaultdict(list)
    for result in results:
        by_run[result.run_id].append(result)
    streaks: dict[str, FailureStreak] = {}
    ended: set[str] = set()
    for run in reversed(runs):
        for result in by_run.get(run.id, []):
            if result.selector in ended or result.status == "skip":
                continue
            if result.status == "pass":
                ended.add(result.selector)
                continue
            previous = streaks.get(result.selector)
            streaks[result.selector] = FailureStreak(
                run.started_at, (previous.runs if previous else 0) + 1, previous.last if previous else result
            )
    return streaks
