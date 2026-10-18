"""``/api/v1/health-checks`` and the run the worker carries out: settings, the claim, and the results by provider."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.exceptions import ModelNotFoundError
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.health_checks import (
    HealthCheckResultRepository,
    HealthCheckRunRepository,
    HealthCheckSettingsRepository,
)
from gateway.schemas.health_checks import HealthCheckSettingsUpdate
from gateway.services.health_checks import HealthCheckOffering, HealthCheckService
from gateway.services.mail import MailDelivery, Mailer, MailMessage

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_CHECKS = f"{API_ROOT}/health-checks"
_PROBE = "gateway.services.health_checks._probe"

_OFFERINGS = [
    HealthCheckOffering("deepinfra:llama", "deepinfra", reasoning=False, chat=True),
    HealthCheckOffering("deepinfra:qwen", "deepinfra", reasoning=False, chat=True),
    HealthCheckOffering("bedrock:claude", "bedrock", reasoning=False, chat=True),
    HealthCheckOffering("bedrock:nova", "bedrock", reasoning=False, chat=True),
    HealthCheckOffering("bedrock:titan-embed", "bedrock", reasoning=False, chat=False),
    HealthCheckOffering("mistral:magistral", "mistral", reasoning=True, chat=True),
]


async def _offerings() -> Sequence[HealthCheckOffering]:
    return _OFFERINGS


def _resolve(_config: GatewayConfig, selector: str) -> SimpleNamespace:
    provider, _, model = selector.partition(":")
    return SimpleNamespace(provider=provider, model=model, kwargs={})


async def _completion(*, model: str, provider: str, max_tokens: int, **_kwargs: Any) -> object:
    if provider == "bedrock":
        msg = f"model {model} is not available in this account"
        raise ModelNotFoundError(msg, provider_name="bedrock")
    return object()


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _mail_on(config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "mail_transport", "console")
    monkeypatch.setattr(config, "public_base_url", "https://otari.example.com")


class _Outbox:
    """Stands in for ``Mailer.send``, keeping what was sent and answering ``delivered``."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, MailMessage]] = []
        self.delivered = True

    async def __call__(self, *, to: str, message: MailMessage) -> MailDelivery:
        self.sent.append((to, message))
        return MailDelivery(delivered=self.delivered, transport="console", reason=None if self.delivered else "down")

    @property
    def subjects(self) -> list[str]:
        return [message.subject for _to, message in self.sent]


def _service(db: AsyncSession, config: GatewayConfig, clock: _Clock) -> HealthCheckService:
    uow = UnitOfWork(db)
    return HealthCheckService(
        uow,
        HealthCheckSettingsRepository(uow),
        HealthCheckRunRepository(uow),
        HealthCheckResultRepository(uow),
        config,
        clock=clock,
    )


def test_the_schedule_is_off_until_an_operator_turns_it_on(client: TestClient) -> None:
    response = client.get(f"{_CHECKS}/settings", headers=_MASTER)

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["next_run_at"] is None


def test_turning_the_schedule_on_makes_a_run_due_now(client: TestClient) -> None:
    before = datetime.now(UTC)

    response = client.patch(f"{_CHECKS}/settings", json={"enabled": True, "interval_minutes": 30}, headers=_MASTER)

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["interval_minutes"] == 30
    assert datetime.fromisoformat(body["next_run_at"]) >= before - timedelta(seconds=1)


@pytest.mark.parametrize(
    "body",
    [{"interval_minutes": 5}, {"prompt": ""}, {"max_tokens": 0}, {"reasoning_every_runs": -1}, {"unknown": 1}],
)
def test_a_setting_out_of_range_is_refused(client: TestClient, body: dict[str, Any]) -> None:
    response = client.patch(f"{_CHECKS}/settings", json=body, headers=_MASTER)

    assert response.status_code == 422


def test_run_now_is_requested_even_with_the_schedule_off(client: TestClient) -> None:
    response = client.post(f"{_CHECKS}/run", headers=_MASTER)

    assert response.status_code == 202
    assert response.json()["run_requested"] is True
    assert response.json()["enabled"] is False


def test_the_status_is_empty_before_any_run(client: TestClient) -> None:
    response = client.get(_CHECKS, headers=_MASTER)

    assert response.status_code == 200
    body = response.json()
    assert body["last_run"] is None
    assert body["running"] is None
    assert body["providers"] == []
    assert body["results"] == []


def test_a_caller_without_a_key_is_refused(client: TestClient) -> None:
    assert client.get(_CHECKS).status_code == 401


def test_the_alert_list_is_stored_trimmed_lowercased_and_once(client: TestClient) -> None:
    response = client.patch(
        f"{_CHECKS}/settings",
        json={"alert_emails": [" Ops@Example.com", "ops@example.com", "oncall@example.com"]},
        headers=_MASTER,
    )

    assert response.status_code == 200
    assert response.json()["alert_emails"] == ["ops@example.com", "oncall@example.com"]
    assert client.get(f"{_CHECKS}/settings", headers=_MASTER).json()["alert_emails"] == [
        "ops@example.com",
        "oncall@example.com",
    ]


@pytest.mark.parametrize("emails", [["not-an-address"], [f"ops{i}@example.com" for i in range(21)]])
def test_an_alert_list_that_cannot_be_mailed_is_refused(client: TestClient, emails: list[str]) -> None:
    response = client.patch(f"{_CHECKS}/settings", json={"alert_emails": emails}, headers=_MASTER)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_nothing_runs_while_the_schedule_is_off(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    assert await _service(async_db, test_config, _Clock()).run_if_due(_offerings) is None


@pytest.mark.asyncio
async def test_a_run_names_the_provider_that_is_failing(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    await service.request_run()

    with patch(f"{_PROBE}.resolve_provider_selector", new=_resolve), patch(f"{_PROBE}.acompletion", new=_completion):
        run = await service.run_if_due(_offerings)

    assert run is not None
    # The reasoning model joins the first run; the embedding model is skipped, not counted.
    assert (run.total, run.passed, run.failed) == (5, 3, 2)
    status = await service.get_status()
    assert [(p.provider, p.status, p.answered, p.total) for p in status.providers] == [
        ("bedrock", "down", 0, 2),
        ("deepinfra", "ok", 2, 2),
        ("mistral", "ok", 1, 1),
    ]
    bedrock = status.providers[0]
    assert bedrock.top_error_count == 2
    assert bedrock.top_error == "[bedrock] model <model> is not available in this account"
    assert {r.selector: r.status_code for r in status.results if r.status == "fail"} == {
        "bedrock:claude": 404,
        "bedrock:nova": 404,
    }
    assert next(r for r in status.results if r.selector == "bedrock:titan-embed").status == "skip"
    assert status.settings.run_requested is False


@pytest.mark.asyncio
async def test_a_due_run_is_claimed_once(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    """The claim moves ``next_run_at`` forward, so the next poll, on this replica or another, finds nothing due."""
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    await service.update_settings(HealthCheckSettingsUpdate(enabled=True, interval_minutes=60))

    with patch(f"{_PROBE}.resolve_provider_selector", new=_resolve), patch(f"{_PROBE}.acompletion", new=_completion):
        first = await service.run_if_due(_offerings)
        clock.now += timedelta(minutes=30)
        second = await service.run_if_due(_offerings)
        clock.now += timedelta(minutes=31)
        third = await service.run_if_due(_offerings)

    assert first is not None
    assert second is None
    assert third is not None
    failing = {r.selector: (r.failing_since, r.failed_runs) for r in (await service.get_status()).results}
    assert failing["bedrock:claude"] == (datetime(2026, 10, 16, 9, 0, tzinfo=UTC), 2)
    assert failing["deepinfra:llama"] == (None, 0)
    # Reasoning models join every 6th run by default, so the second run left the mistral one out.
    assert (first.total, third.total) == (5, 4)
    status = await service.get_status()
    assert [len(p.history) for p in status.providers if p.provider == "deepinfra"] == [2]


@pytest.mark.asyncio
async def test_results_older_than_two_days_are_removed(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    clock = _Clock()
    service = _service(async_db, test_config, clock)

    with patch(f"{_PROBE}.resolve_provider_selector", new=_resolve), patch(f"{_PROBE}.acompletion", new=_completion):
        await service.request_run()
        old = await service.run_if_due(_offerings)
        clock.now += timedelta(hours=49)
        await service.request_run()
        await service.run_if_due(_offerings)

    status = await service.get_status()
    assert old is not None
    assert status.last_run is not None and status.last_run.id != old.id
    assert all(len(p.history) == 1 for p in status.providers)


async def _run_with(service: HealthCheckService, completion: Any, outbox: _Outbox) -> None:
    await service.request_run()
    with (
        patch(f"{_PROBE}.resolve_provider_selector", new=_resolve),
        patch(f"{_PROBE}.acompletion", new=completion),
        patch.object(Mailer, "send", new=outbox),
    ):
        await service.run_if_due(_offerings)


async def _all_answer(**_kwargs: Any) -> object:
    return object()


@pytest.mark.asyncio
async def test_the_alert_list_hears_when_a_provider_fails_and_when_it_is_back(
    async_db: AsyncSession, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mail_on(test_config, monkeypatch)
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    await service.update_settings(HealthCheckSettingsUpdate(alert_emails=["ops@example.com", "oncall@example.com"]))
    outbox = _Outbox()

    await _run_with(service, _completion, outbox)
    clock.now += timedelta(minutes=5)
    await _run_with(service, _completion, outbox)
    clock.now += timedelta(minutes=5)
    await _run_with(service, _all_answer, outbox)

    assert [to for to, _message in outbox.sent] == ["ops@example.com", "oncall@example.com"] * 2
    assert outbox.subjects[0] == "Otari health on otari.example.com: bedrock is down"
    # The second run found bedrock still down, which the list already knew.
    assert outbox.subjects[2] == "Otari health on otari.example.com: bedrock is answering again"
    assert "  bedrock:claude  failed for 10 minutes, 2 runs in a row" in outbox.sent[2][1].text
    assert "https://otari.example.com/#/health-checks" in outbox.sent[0][1].text


@pytest.mark.asyncio
async def test_an_alert_nobody_received_is_sent_again_on_the_next_run(
    async_db: AsyncSession, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mail_on(test_config, monkeypatch)
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    await service.update_settings(HealthCheckSettingsUpdate(alert_emails=["ops@example.com"]))
    outbox = _Outbox()
    outbox.delivered = False

    await _run_with(service, _completion, outbox)
    outbox.delivered = True
    clock.now += timedelta(minutes=5)
    await _run_with(service, _completion, outbox)

    assert outbox.subjects == ["Otari health on otari.example.com: bedrock is down"] * 2


@pytest.mark.asyncio
async def test_with_nobody_on_the_list_a_failure_is_remembered_not_saved_up(
    async_db: AsyncSession, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An address added later hears about the next change, not the history before it."""
    _mail_on(test_config, monkeypatch)
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    outbox = _Outbox()

    await _run_with(service, _completion, outbox)
    await service.update_settings(HealthCheckSettingsUpdate(alert_emails=["ops@example.com"]))
    clock.now += timedelta(minutes=5)
    await _run_with(service, _completion, outbox)
    clock.now += timedelta(minutes=5)
    await _run_with(service, _all_answer, outbox)

    assert outbox.subjects == ["Otari health on otari.example.com: bedrock is answering again"]


@pytest.mark.asyncio
async def test_a_run_that_breaks_before_probing_does_not_stay_running(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    service = _service(async_db, test_config, _Clock())
    await service.request_run()

    async def broken() -> Sequence[HealthCheckOffering]:
        msg = "catalog unavailable"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        await service.run_if_due(broken)

    status = await service.get_status()
    assert status.running is None
    assert status.last_run is None


@pytest.mark.asyncio
async def test_a_run_in_progress_holds_the_next_one_back(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    """A requested run waits for the one in progress, so two runs never email the same change."""
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    await service.request_run()

    async def overlapping() -> Sequence[HealthCheckOffering]:
        await service.request_run()
        assert await _service(async_db, test_config, clock).run_if_due(_offerings) is None
        return _OFFERINGS

    with patch(f"{_PROBE}.resolve_provider_selector", new=_resolve), patch(f"{_PROBE}.acompletion", new=_completion):
        assert await service.run_if_due(overlapping) is not None
        assert (await service.get_settings()).run_requested is True
        assert await service.run_if_due(_offerings) is not None


@pytest.mark.asyncio
async def test_a_reasoning_model_stays_on_the_page_between_the_runs_that_check_it(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    clock = _Clock()
    service = _service(async_db, test_config, clock)

    async def mistral_fails(*, model: str, provider: str, max_tokens: int, **_kwargs: Any) -> object:
        if provider == "mistral":
            msg = "overloaded"
            raise RuntimeError(msg)
        return object()

    with patch(f"{_PROBE}.resolve_provider_selector", new=_resolve), patch(f"{_PROBE}.acompletion", new=mistral_fails):
        await service.request_run()
        first = await service.run_if_due(_offerings)
        clock.now += timedelta(minutes=5)
        await service.request_run()
        second = await service.run_if_due(_offerings)

    assert first is not None and first.included_reasoning
    assert second is not None and not second.included_reasoning
    status = await service.get_status()
    magistral = next(r for r in status.results if r.selector == "mistral:magistral")
    assert magistral.status == "fail"
    assert [(p.provider, p.status) for p in status.providers if p.provider == "mistral"] == [("mistral", "down")]
