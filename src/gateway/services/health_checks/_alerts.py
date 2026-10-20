"""Which changes a run emails to the alert list, and the email that says so.

A run emails only what differs from the last email: a model that stopped
answering, or one that answers again. A provider that stays down through a
night of hourly runs sends one email when it goes down and one when it is back,
not one per run.
"""

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlsplit

from gateway.services.health_checks._probe import ProbeOutcome
from gateway.services.health_checks._summary import FailureStreak
from gateway.services.mail import MailMessage, render_email

# Past this many models, a provider's section names the rest by count.
MODELS_LISTED_PER_PROVIDER = 10
PROVIDERS_NAMED = 3


@dataclass(frozen=True)
class AlertChange:
    """What a run found that the alert list has not been told."""

    failing: tuple[ProbeOutcome, ...]
    recovered: tuple[ProbeOutcome, ...]
    # The selectors the list will have been told are failing, once this change is emailed.
    alerted: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.failing or self.recovered)


def diff_alerts(alerted: Sequence[str], outcomes: Sequence[ProbeOutcome], offered: Collection[str]) -> AlertChange:
    """Compare one run with what the alert list was last told.

    A model this run did not probe, such as a reasoning model on a run that
    leaves them out, keeps the state it had. One the deployment no longer
    offers is dropped without an email, since there is nothing left to fix.
    """
    told = [selector for selector in alerted if selector in offered]
    known = set(told)
    failing = tuple(o for o in outcomes if o.status == "fail" and o.target.selector not in known)
    recovered = tuple(o for o in outcomes if o.status == "pass" and o.target.selector in known)
    answering = {o.target.selector for o in recovered}
    now = {selector for selector in told if selector not in answering} | {o.target.selector for o in failing}
    return AlertChange(failing, recovered, tuple(sorted(now)))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _names(providers: Sequence[str]) -> str:
    shown = list(providers[:PROVIDERS_NAMED])
    if len(providers) > PROVIDERS_NAMED:
        shown.append(_plural(len(providers) - PROVIDERS_NAMED, "more provider"))
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


@dataclass(frozen=True)
class _ProviderRun:
    answered: int
    total: int


def _by_provider(outcomes: Sequence[ProbeOutcome]) -> dict[str, _ProviderRun]:
    answered: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for outcome in outcomes:
        if outcome.status == "skip":
            continue
        total[outcome.target.provider] += 1
        answered[outcome.target.provider] += outcome.status == "pass"
    return {provider: _ProviderRun(answered[provider], total[provider]) for provider in total}


def _headline(change: AlertChange, runs: dict[str, _ProviderRun]) -> str:
    """Down providers first, since they are the worst news, then partial failures, then recoveries."""
    failing = sorted({o.target.provider for o in change.failing})
    down = [provider for provider in failing if runs[provider].answered == 0]
    partial = [provider for provider in failing if provider not in down]
    recovered = sorted({o.target.provider for o in change.recovered})
    parts: list[str] = []
    if down:
        parts.append(f"{_names(down)} {'is' if len(down) == 1 else 'are'} down")
    if len(partial) == 1:
        count = sum(o.target.provider == partial[0] for o in change.failing)
        parts.append(f"{_plural(count, 'model')} on {partial[0]} stopped answering")
    elif partial:
        parts.append(f"models stopped answering on {_names(partial)}")
    if len(recovered) == 1 and runs[recovered[0]].answered == runs[recovered[0]].total:
        parts.append(f"{recovered[0]} is answering again")
    elif recovered:
        parts.append(f"{_plural(len(change.recovered), 'model')} answering again on {_names(recovered)}")
    # Not capitalized: an instance name is an identifier, spelled as configured.
    return "; ".join(parts)


def _duration(delta: timedelta) -> str:
    minutes = max(1, round(delta.total_seconds() / 60))
    if minutes < 60:
        return _plural(minutes, "minute")
    hours = round(minutes / 60)
    return _plural(hours, "hour") if hours < 48 else _plural(round(hours / 24), "day")


def _model_line(outcome: ProbeOutcome, streak: FailureStreak | None, checked_at: datetime) -> str:
    if outcome.status == "pass":
        reason = (
            f"failed for {_duration(checked_at - streak.since)}, {_plural(streak.runs, 'run')} in a row"
            if streak
            else ""
        )
    else:
        reason = " · ".join(
            part
            for part in (f"HTTP {outcome.status_code}" if outcome.status_code else None, outcome.error)
            if part is not None
        )
    return f"  {outcome.target.selector}" + (f"  {reason}" if reason else "")


def _section(
    title: str,
    outcomes: Sequence[ProbeOutcome],
    runs: dict[str, _ProviderRun],
    down_for: Mapping[str, FailureStreak],
    checked_at: datetime,
) -> list[str]:
    lines = [title]
    grouped: dict[str, list[ProbeOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.target.provider].append(outcome)
    for provider in sorted(grouped):
        run = runs[provider]
        lines.append(f"{provider}: {run.answered} of {_plural(run.total, 'model')} answer")
        models = sorted(grouped[provider], key=lambda o: o.target.selector)
        lines.extend(
            _model_line(o, down_for.get(o.target.selector), checked_at) for o in models[:MODELS_LISTED_PER_PROVIDER]
        )
        if len(models) > MODELS_LISTED_PER_PROVIDER:
            lines.append(f"  and {len(models) - MODELS_LISTED_PER_PROVIDER} more")
    return lines


def deployment_name(ui_base_url: str) -> str:
    """The host an alert names, so mail from two regions says which one it is about."""
    return urlsplit(ui_base_url).hostname or "this deployment"


def render_health_alert(
    change: AlertChange,
    outcomes: Sequence[ProbeOutcome],
    *,
    down_for: Mapping[str, FailureStreak],
    checked_at: datetime,
    deployment: str,
    health_link: str,
    asset_root: str | None,
) -> MailMessage:
    """The email for one run's change, listing each affected model under its provider.

    ``down_for`` holds the streak each recovered model's answer ended, so the email can say how long it was out.
    """
    runs = _by_provider(outcomes)
    headline = _headline(change, runs)
    sections: list[str] = []
    if change.failing:
        sections.extend(_section("Stopped answering", change.failing, runs, down_for, checked_at))
    if change.recovered:
        if sections:
            sections.append("")
        sections.extend(_section("Answering again", change.recovered, runs, down_for, checked_at))
    found: list[str] = []
    if change.failing:
        found.append(f"{_plural(len(change.failing), 'model')} that stopped answering")
    if change.recovered:
        verb = "answers" if len(change.recovered) == 1 else "answer"
        found.append(f"{_plural(len(change.recovered), 'model')} that {verb} again")
    return render_email(
        "health_alert",
        subject="Otari health on {{DEPLOYMENT}}: {{HEADLINE}}",
        preheader="{{INTRO}}",
        asset_root=asset_root,
        values={
            "DEPLOYMENT": deployment,
            "HEADLINE": headline,
            "INTRO": f"The health check at {checked_at:%H:%M} UTC on {deployment} found {' and '.join(found)}.",
            "CHANGES": "\n".join(sections),
            "HEALTH_LINK": health_link,
        },
    )
