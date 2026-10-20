"""Which changes a run emails to the alert list, and what the email says."""

from datetime import UTC, datetime, timedelta
from typing import Literal

from gateway.services.health_checks._alerts import (
    deployment_name,
    diff_alerts,
    render_health_alert,
)
from gateway.services.health_checks._probe import HealthCheckOffering, ProbeOutcome
from gateway.services.health_checks._summary import FailureStreak

_AT = datetime(2026, 10, 16, 9, 1, tzinfo=UTC)


def _outcome(
    selector: str, status: Literal["pass", "fail", "skip"], status_code: int | None = None, error: str | None = None
) -> ProbeOutcome:
    provider = selector.partition(":")[0]
    return ProbeOutcome(
        HealthCheckOffering(selector, provider, False, True, provider, selector.partition(":")[2]),
        status,
        status_code,
        error,
        10,
    )


def _render(
    alerted: list[str], outcomes: list[ProbeOutcome], down_for: dict[str, FailureStreak] | None = None
) -> tuple[str, str]:
    change = diff_alerts(alerted, outcomes, {o.target.selector for o in outcomes})
    message = render_health_alert(
        change,
        outcomes,
        down_for=down_for or {},
        checked_at=_AT,
        deployment="us.otari.example",
        health_link="https://us.otari.example/#/health-checks",
        asset_root=None,
    )
    return message.subject, message.text


def test_a_model_that_stops_answering_is_emailed_once() -> None:
    outcomes = [_outcome("acme:fast", "pass"), _outcome("acme:retired", "fail", 404)]

    first = diff_alerts([], outcomes, {"acme:fast", "acme:retired"})
    second = diff_alerts(first.alerted, outcomes, {"acme:fast", "acme:retired"})

    assert [o.target.selector for o in first.failing] == ["acme:retired"]
    assert first.alerted == ("acme:retired",)
    assert not second


def test_a_model_that_answers_again_is_emailed_and_forgotten() -> None:
    change = diff_alerts(["acme:fast"], [_outcome("acme:fast", "pass")], {"acme:fast"})

    assert [o.target.selector for o in change.recovered] == ["acme:fast"]
    assert change.alerted == ()


def test_a_model_left_out_of_a_run_keeps_its_state() -> None:
    """A reasoning model on a run that leaves them out is neither failing nor recovered."""
    change = diff_alerts(["mistral:magistral"], [_outcome("acme:fast", "pass")], {"acme:fast", "mistral:magistral"})

    assert not change
    assert change.alerted == ("mistral:magistral",)


def test_a_model_no_longer_offered_is_dropped_without_an_email() -> None:
    change = diff_alerts(["acme:gone"], [_outcome("acme:fast", "pass")], {"acme:fast"})

    assert not change
    assert change.alerted == ()


def test_a_provider_with_nothing_answering_is_called_down() -> None:
    subject, text = _render(
        [], [_outcome("flakyco:a", "fail", 503, "upstream overloaded"), _outcome("flakyco:b", "fail", 503)]
    )

    assert subject == "Otari health on us.otari.example: flakyco is down"
    assert "flakyco: 0 of 2 models answer" in text
    assert "  flakyco:a  HTTP 503 · upstream overloaded" in text
    assert "https://us.otari.example/#/health-checks" in text


def test_a_provider_with_some_models_failing_names_how_many() -> None:
    subject, text = _render([], [_outcome("acme:fast", "pass"), _outcome("acme:retired", "fail", 404)])

    assert subject.endswith(": 1 model on acme stopped answering")
    assert "acme: 1 of 2 models answer" in text


def test_failures_and_recoveries_share_one_email() -> None:
    subject, text = _render(
        ["acme:fast"],
        [_outcome("acme:fast", "pass"), _outcome("bedrock:claude", "fail", 404), _outcome("bedrock:nova", "pass")],
    )

    assert subject.endswith(": 1 model on bedrock stopped answering; acme is answering again")
    assert text.index("Stopped answering") < text.index("Answering again")
    assert "found 1 model that stopped answering and 1 model that answers again." in text


def test_failures_on_several_providers_name_them() -> None:
    outcomes = [_outcome(f"{p}:m", "fail", 500) for p in ("a", "b", "c", "d", "e")]

    subject, _ = _render([], outcomes)

    assert subject.endswith(": a, b, c and 2 more providers are down")


def test_a_down_provider_leads_the_subject() -> None:
    subject, _ = _render(
        [],
        [
            _outcome("acme:fast", "pass"),
            _outcome("acme:retired", "fail", 404),
            _outcome("zeta:a", "fail", 503),
            _outcome("other:a", "pass"),
            _outcome("other:b", "fail", 500),
        ],
    )

    assert subject.endswith(": zeta is down; models stopped answering on acme and other")


def test_a_long_list_of_models_is_cut_short() -> None:
    outcomes = [_outcome(f"acme:m{i:02}", "fail", 404) for i in range(13)]

    _, text = _render([], outcomes)

    assert "  acme:m09" in text
    assert "acme:m10" not in text
    assert "  and 3 more" in text


def test_an_alert_names_the_deployment_it_is_about() -> None:
    assert deployment_name("https://eu.otari.example/ui/?edge=a") == "eu.otari.example"
    assert deployment_name("") == "this deployment"


def test_a_recovery_says_how_long_the_model_was_out() -> None:
    _, text = _render(
        ["flakyco:a"],
        [_outcome("flakyco:a", "pass")],
        {"flakyco:a": FailureStreak(_AT - timedelta(hours=3), 4)},
    )

    assert "  flakyco:a  failed for 3 hours, 4 runs in a row" in text
