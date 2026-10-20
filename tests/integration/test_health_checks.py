"""``/api/v1/health-checks`` and the run the worker carries out: settings, the claim, and the results by provider."""

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.exceptions import ModelNotFoundError
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import build_hosted_provider_service
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.repositories.health_checks import (
    HealthCheckRecoveryRepository,
    HealthCheckResultRepository,
    HealthCheckRunRepository,
    HealthCheckSettingsRepository,
)
from gateway.schemas.health_checks import HealthCheckSettingsUpdate
from gateway.services.health_checks import HealthCheckOffering, HealthCheckService, list_hosted_offerings
from gateway.services.health_checks._service import HostedRoster
from gateway.services.mail import MailDelivery, Mailer, MailMessage
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_CHECKS = f"{API_ROOT}/health-checks"
_PROBE = "gateway.services.health_checks._probe"

_OFFERINGS = [
    HealthCheckOffering(
        "deepinfra:llama", "deepinfra", reasoning=False, chat=True, implementation="deepinfra", model="llama"
    ),
    HealthCheckOffering(
        "deepinfra:qwen", "deepinfra", reasoning=False, chat=True, implementation="deepinfra", model="qwen"
    ),
    HealthCheckOffering(
        "bedrock:claude", "bedrock", reasoning=False, chat=True, implementation="bedrock", model="claude"
    ),
    HealthCheckOffering("bedrock:nova", "bedrock", reasoning=False, chat=True, implementation="bedrock", model="nova"),
    HealthCheckOffering(
        "bedrock:titan-embed", "bedrock", reasoning=False, chat=False, implementation="bedrock", model="titan-embed"
    ),
    HealthCheckOffering(
        "mistral:magistral", "mistral", reasoning=True, chat=True, implementation="mistral", model="magistral"
    ),
]


async def _offerings() -> Sequence[HealthCheckOffering]:
    return _OFFERINGS


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


class _Roster:
    """Stands in for the hosted providers: serves every test offering until one is switched off."""

    def __init__(self) -> None:
        self.served: dict[str, set[str]] = defaultdict(set)
        for offering in _OFFERINGS:
            self.served[offering.provider].add(offering.model)
        self.switched_off: list[dict[str, set[str]]] = []

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]:
        return {name: frozenset(models) for name, models in self.served.items() if models}

    async def switch_off_models(self, models: Mapping[str, Collection[str]]) -> int:
        self.switched_off.append({name: set(names) for name, names in models.items()})
        before = sum(len(names) for names in self.served.values())
        for name, names in models.items():
            self.served[name] -= set(names)
        return before - sum(len(names) for names in self.served.values())


def _service(
    db: AsyncSession, config: GatewayConfig, clock: _Clock, roster: HostedRoster | None = None
) -> HealthCheckService:
    uow = UnitOfWork(db)
    return HealthCheckService(
        uow,
        HealthCheckSettingsRepository(uow),
        HealthCheckRunRepository(uow),
        HealthCheckResultRepository(uow),
        HealthCheckRecoveryRepository(uow),
        roster or _Roster(),
        config,
        clock=clock,
    )


def test_the_schedule_is_off_until_an_operator_turns_it_on(client: TestClient) -> None:
    response = client.get(f"{_CHECKS}/settings", headers=_MASTER)

    assert response.status_code == 200
    assert response.json()["schedule_enabled"] is False
    assert response.json()["next_run_at"] is None


def test_turning_the_schedule_on_makes_a_run_due_now(client: TestClient) -> None:
    before = datetime.now(UTC)

    response = client.patch(
        f"{_CHECKS}/settings", json={"schedule_enabled": True, "interval_minutes": 30}, headers=_MASTER
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schedule_enabled"] is True
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
    assert response.json()["schedule_enabled"] is False


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

    with patch(f"{_PROBE}.acompletion", new=_completion):
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
    await service.update_settings(HealthCheckSettingsUpdate(schedule_enabled=True, interval_minutes=60))

    with patch(f"{_PROBE}.acompletion", new=_completion):
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

    with patch(f"{_PROBE}.acompletion", new=_completion):
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

    with patch(f"{_PROBE}.acompletion", new=_completion):
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

    with patch(f"{_PROBE}.acompletion", new=mistral_fails):
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


@pytest.mark.asyncio
async def test_a_model_that_answers_again_leaves_the_failing_list(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    clock = _Clock()
    service = _service(async_db, test_config, clock)

    async def answers(**_kwargs: Any) -> object:
        return object()

    with patch(f"{_PROBE}.acompletion", new=_completion):
        await service.request_run()
        await service.run_if_due(_offerings)
    clock.now += timedelta(minutes=5)
    with patch(f"{_PROBE}.acompletion", new=answers):
        await service.request_run()
        await service.run_if_due(_offerings)

    status = await service.get_status()
    assert [r.selector for r in status.results if r.status == "fail"] == []
    assert {p.provider: p.status for p in status.providers} == {"bedrock": "ok", "deepinfra": "ok", "mistral": "ok"}


@pytest.mark.asyncio
async def test_a_reasoning_model_no_longer_served_leaves_the_page(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    """Only the reasoning models a run deferred are carried from the run that last checked them."""
    clock = _Clock()
    service = _service(async_db, test_config, clock)

    async def mistral_fails(*, model: str, provider: str, max_tokens: int, **_kwargs: Any) -> object:
        if provider == "mistral":
            msg = "overloaded"
            raise RuntimeError(msg)
        return object()

    async def without_mistral() -> Sequence[HealthCheckOffering]:
        return [offering for offering in _OFFERINGS if offering.provider != "mistral"]

    with patch(f"{_PROBE}.acompletion", new=mistral_fails):
        await service.request_run()
        await service.run_if_due(_offerings)
        clock.now += timedelta(minutes=5)
        await service.request_run()
        second = await service.run_if_due(without_mistral)

    assert second is not None and not second.included_reasoning
    status = await service.get_status()
    assert all(r.provider != "mistral" for r in status.results)
    assert all(p.provider != "mistral" for p in status.providers)


@pytest.mark.asyncio
async def test_a_run_checks_what_the_hosted_providers_serve_on_their_own_keys(
    async_db: AsyncSession, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model switched off, a provider switched off, and one whose key does not read are left out."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    async_db.add_all(
        [
            HostedProvider(
                name="openai-eu",
                provider="openai",
                encrypted_api_key=encrypt_secret("sk-hosted-eu-1234"),
                api_key_last4="1234",
                api_base="https://eu.example.com/v1",
            ),
            HostedProvider(
                name="mistral", provider="mistral", encrypted_api_key=encrypt_secret("sk-off-1234"), enabled=False
            ),
            HostedProvider(name="anthropic", provider="anthropic", encrypted_api_key="not-a-ciphertext"),
            HostedProviderModel(provider="openai-eu", model="gpt-test"),
            HostedProviderModel(provider="openai-eu", model="gpt-off", enabled=False),
            HostedProviderModel(provider="mistral", model="magistral"),
            HostedProviderModel(provider="anthropic", model="claude"),
        ]
    )
    await async_db.flush()

    offerings = await list_hosted_offerings(
        build_hosted_provider_service(UnitOfWork(async_db), config=test_config), test_config
    )

    assert [(o.selector, o.provider, o.implementation, o.model) for o in offerings] == [
        ("openai-eu:gpt-test", "openai-eu", "openai", "gpt-test")
    ]
    assert offerings[0].call_kwargs == {"api_key": "sk-hosted-eu-1234", "api_base": "https://eu.example.com/v1"}
    assert "sk-hosted" not in repr(offerings[0])

    sent: list[dict[str, Any]] = []

    async def record(**kwargs: Any) -> object:
        sent.append(kwargs)
        return object()

    service = _service(async_db, test_config, _Clock())
    await service.request_run()
    with patch(f"{_PROBE}.acompletion", new=record):
        run = await service.run_if_due(lambda: _listed(offerings))

    assert run is not None and run.passed == 1
    assert [(call["provider"], call["model"], call["api_key"], call["api_base"]) for call in sent] == [
        ("openai", "gpt-test", "sk-hosted-eu-1234", "https://eu.example.com/v1")
    ]


async def _listed(offerings: Sequence[HealthCheckOffering]) -> Sequence[HealthCheckOffering]:
    return offerings


@pytest.mark.asyncio
async def test_disable_all_switches_off_the_failing_models_and_they_leave_the_list(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    roster = _Roster()
    service = _service(async_db, test_config, _Clock(), roster)
    await service.request_run()
    with patch(f"{_PROBE}.acompletion", new=_completion):
        await service.run_if_due(_offerings)

    assert await service.disable_failing() == 2

    assert roster.switched_off == [{"bedrock": {"claude", "nova"}}]
    status = await service.get_status()
    assert [r.selector for r in status.results if r.status == "fail"] == []
    assert "bedrock" not in {p.provider for p in status.providers}
    assert await service.disable_failing() == 0


@pytest.mark.asyncio
async def test_an_outage_that_ends_is_kept_on_record(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    """A model that answers again leaves the failing list; how long it was down is kept past the results."""
    clock = _Clock()
    service = _service(async_db, test_config, clock)
    first_started = clock.now

    async def answers(**_kwargs: Any) -> object:
        return object()

    for completion in (_completion, _completion, answers, answers):
        await service.request_run()
        with patch(f"{_PROBE}.acompletion", new=completion):
            await service.run_if_due(_offerings)
        clock.now += timedelta(minutes=30)

    status = await service.get_status()
    assert [r.selector for r in status.results if r.status == "fail"] == []
    assert {p.provider: p.status for p in status.providers}["bedrock"] == "ok"
    uow = UnitOfWork(async_db)
    recoveries = HealthCheckRecoveryRepository(uow)
    async with uow:
        claude = await recoveries.for_selector("bedrock:claude")
        never_failed = await recoveries.for_selector("deepinfra:llama")
    assert [(r.provider, r.failing_since, r.recovered_at, r.failed_runs, r.status_code) for r in claude] == [
        ("bedrock", first_started, first_started + timedelta(minutes=60), 2, 404)
    ]
    assert claude[0].error is not None and "not available" in claude[0].error
    # A model that never failed has nothing to recover from.
    assert never_failed == []


@pytest.mark.asyncio
async def test_disable_all_switches_the_hosted_rows_off(
    async_db: AsyncSession, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    async_db.add_all(
        [
            HostedProvider(name="openai", provider="openai", encrypted_api_key=encrypt_secret("sk-hosted-1234")),
            HostedProviderModel(provider="openai", model="gpt-broken"),
            HostedProviderModel(provider="openai", model="gpt-fine"),
        ]
    )
    await async_db.flush()
    hosted = build_hosted_provider_service(UnitOfWork(async_db), config=test_config)
    service = _service(async_db, test_config, _Clock(), hosted)

    async def broken_only(*, model: str, **_kwargs: Any) -> object:
        if model == "gpt-broken":
            msg = "model gpt-broken does not exist"
            raise ModelNotFoundError(msg, provider_name="openai")
        return object()

    await service.request_run()
    with patch(f"{_PROBE}.acompletion", new=broken_only):
        await service.run_if_due(lambda: list_hosted_offerings(hosted, test_config))

    assert await service.disable_failing() == 1
    assert await hosted.serveable_models(organization_id=None) == {"openai": frozenset({"gpt-fine"})}
    assert [r.selector for r in (await service.get_status()).results] == ["openai:gpt-fine"]


def test_disable_failing_answers_with_the_count(client: TestClient) -> None:
    response = client.post(f"{_CHECKS}/disable-failing", headers=_MASTER)
    assert response.status_code == 200, response.text
    assert response.json() == {"disabled": 0}
    assert client.post(f"{_CHECKS}/disable-failing").status_code == 401


def test_switched_off_the_page_and_its_routes_are_gone(test_config: GatewayConfig) -> None:
    config = test_config.model_copy(update={"health_checks_enabled": False})
    for client in build_test_client(config):
        assert "health_checks" not in client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]
        assert client.get(_CHECKS, headers=_MASTER).status_code == 404
        assert client.post(f"{_CHECKS}/run", headers=_MASTER).status_code == 404


def test_switched_on_the_page_is_published(client: TestClient) -> None:
    assert "health_checks" in client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]


@pytest.mark.asyncio
async def test_a_cancelled_run_does_not_stay_running(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    """A shutdown mid-run must not hold every replica's next run back for an hour."""
    service = _service(async_db, test_config, _Clock())
    await service.request_run()

    async def cancelled() -> Sequence[HealthCheckOffering]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await service.run_if_due(cancelled)

    assert (await service.get_status()).running is None


@pytest.mark.asyncio
async def test_the_settings_row_survives_a_lost_insert_race(async_db: AsyncSession) -> None:
    """Two first reads at once both find no row; the one whose insert loses reads the other's."""
    uow = UnitOfWork(async_db)
    settings = HealthCheckSettingsRepository(uow)
    async with uow:
        await settings.get_or_create()

    real_get = async_db.get
    calls = 0

    async def first_read_misses(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return None if calls == 1 else await real_get(*args, **kwargs)

    with patch.object(async_db, "get", new=first_read_misses):
        async with uow:
            row = await settings.get_or_create()

    assert row.id == 1
    assert calls == 2
