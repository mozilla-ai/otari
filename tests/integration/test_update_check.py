"""``/api/v1/update-check`` and the check the worker sends: the operator's answer, the claim, and what is kept."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.update_check import UpdateCheckStateRepository
from gateway.schemas.update_check import UpdateCheckAdvisory, UpdateCheckAnswer, UpdateCheckReply
from gateway.services.update_check import UpdateCheckService
from gateway.services.update_check._client import UpdateCheckFailedError
from gateway.services.update_check._schedule import next_daily_check

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_CHECK = f"{API_ROOT}/update-check"
_ADVISORY = UpdateCheckAdvisory(
    id="GHSA-abcd-efgh-ijkl",
    summary="A key could be read back",
    severity="high",
    url="https://github.com/mozilla-ai/otari/security/advisories/GHSA-abcd-efgh-ijkl",
    fixed_in="0.30.2",
)


@pytest.fixture(autouse=True)
def _nothing_preset(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig) -> None:
    """CI sets ``CI``, which answers no; these tests decide the answer themselves."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setattr(test_config, "update_check", None)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class _Server:
    """Stands in for otari.ai, keeping every check it was sent."""

    def __init__(self, reply: UpdateCheckReply | None = None) -> None:
        self.sent: list[tuple[str, str, uuid.UUID]] = []
        self.reply = reply or UpdateCheckReply(latest_version="0.31.0", release_url="https://example/v0.31.0")
        self.failing = False

    async def __call__(self, url: str, version: str, install_id: uuid.UUID) -> UpdateCheckReply:
        self.sent.append((url, version, install_id))
        if self.failing:
            raise UpdateCheckFailedError("status 503")
        return self.reply


def _service(
    db: AsyncSession, config: GatewayConfig, server: _Server, clock: _Clock, version: str = "v0.30.1"
) -> UpdateCheckService:
    uow = UnitOfWork(db)
    return UpdateCheckService(uow, UpdateCheckStateRepository(uow), config, version=version, send=server, clock=clock)


def test_nobody_has_answered_on_a_new_install(client: TestClient) -> None:
    response = client.get(_CHECK, headers=_MASTER)
    assert response.status_code == 200
    body = response.json()
    assert (body["enabled"], body["answered_by"]) == (None, None)
    assert body["endpoint"] == "https://api.otari.ai/api/v1/update-checks"
    uuid.UUID(body["install_id"])


def test_the_install_id_does_not_change_between_reads(client: TestClient) -> None:
    first = client.get(_CHECK, headers=_MASTER).json()["install_id"]
    assert client.get(_CHECK, headers=_MASTER).json()["install_id"] == first


def test_an_operator_answers_yes_and_then_no(client: TestClient) -> None:
    yes = client.put(_CHECK, json={"enabled": True}, headers=_MASTER)
    assert yes.status_code == 200
    assert (yes.json()["enabled"], yes.json()["answered_by"]) == (True, "operator")

    no = client.put(_CHECK, json={"enabled": False}, headers=_MASTER)
    assert (no.json()["enabled"], no.json()["answered_by"]) == (False, "operator")


def test_the_dashboard_cannot_change_what_config_decided(
    client: TestClient, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_config, "update_check", False)
    assert client.get(_CHECK, headers=_MASTER).json()["answered_by"] == "config"
    response = client.put(_CHECK, json={"enabled": True}, headers=_MASTER)
    assert response.status_code == 409
    assert "update_check" in response.json()["detail"]


def test_do_not_track_answers_no(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    body = client.get(_CHECK, headers=_MASTER).json()
    assert (body["enabled"], body["answered_by"]) == (False, "config")


def test_a_caller_without_a_key_is_refused(client: TestClient) -> None:
    assert client.get(_CHECK).status_code == 401
    assert client.put(_CHECK, json={"enabled": True}).status_code == 401


def test_an_answer_carries_nothing_but_enabled(client: TestClient) -> None:
    response = client.put(_CHECK, json={"enabled": True, "install_id": "x"}, headers=_MASTER)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_nothing_is_sent_until_an_operator_answers(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock)
    assert await service.check_if_due() is False
    clock.now += timedelta(days=2)
    assert await service.check_if_due() is False
    assert server.sent == []


@pytest.mark.asyncio
async def test_nothing_is_sent_after_an_operator_says_no(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock)
    await service.answer(UpdateCheckAnswer(enabled=False))
    clock.now += timedelta(days=2)
    assert await service.check_if_due() is False
    assert server.sent == []


@pytest.mark.asyncio
async def test_a_development_build_never_checks(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock, version="0.0.0-dev")
    status = await service.answer(UpdateCheckAnswer(enabled=True))
    assert status.checkable is False
    assert await service.check_if_due() is False
    assert server.sent == []


@pytest.mark.asyncio
async def test_a_yes_is_checked_at_once_and_then_once_a_day(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock)
    status = await service.answer(UpdateCheckAnswer(enabled=True))

    assert await service.check_if_due() is True
    # Another replica polling a moment later finds the check already sent.
    clock.now += timedelta(minutes=1)
    assert await service.check_if_due() is False

    [(url, version, install_id)] = server.sent
    assert (url, version, str(install_id)) == (test_config.update_check_url, "0.30.1", status.install_id)

    after = await service.get_status()
    assert after.latest_version == "0.31.0"
    assert after.update_available is True
    assert after.next_check_at == next_daily_check(install_id, clock.now - timedelta(minutes=1))

    clock.now = after.next_check_at
    assert await service.check_if_due() is True
    assert len(server.sent) == 2


@pytest.mark.asyncio
async def test_config_turns_the_check_on_at_the_installs_own_time_not_at_startup(
    async_db: AsyncSession, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_config, "update_check", True)
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock)

    assert await service.check_if_due() is False
    assert server.sent == []
    status = await service.get_status()
    assert status.next_check_at == next_daily_check(uuid.UUID(status.install_id), clock.now)

    clock.now = status.next_check_at
    assert await service.check_if_due() is True


@pytest.mark.asyncio
async def test_a_failed_check_is_retried_and_keeps_what_the_last_one_heard(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock)
    await service.answer(UpdateCheckAnswer(enabled=True))
    await service.check_if_due()

    clock.now = (await service.get_status()).next_check_at or clock.now
    server.failing = True
    assert await service.check_if_due() is True

    status = await service.get_status()
    assert status.latest_version == "0.31.0"
    assert status.next_check_at is not None
    assert clock.now < status.next_check_at <= clock.now + timedelta(minutes=8)


@pytest.mark.asyncio
async def test_advisories_for_the_running_version_are_shown_until_it_is_upgraded(
    async_db: AsyncSession, test_config: GatewayConfig
) -> None:
    server, clock = _Server(UpdateCheckReply(latest_version="0.30.2", advisories=[_ADVISORY])), _Clock()
    service = _service(async_db, test_config, server, clock)
    await service.answer(UpdateCheckAnswer(enabled=True))
    await service.check_if_due()

    assert [advisory.id for advisory in (await service.get_status()).advisories] == [_ADVISORY.id]
    upgraded = _service(async_db, test_config, server, clock, version="v0.30.2")
    status = await upgraded.get_status()
    assert status.advisories == []
    assert status.update_available is False


@pytest.mark.asyncio
async def test_saying_no_forgets_what_the_checks_heard(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    server, clock = _Server(), _Clock()
    service = _service(async_db, test_config, server, clock)
    await service.answer(UpdateCheckAnswer(enabled=True))
    await service.check_if_due()

    status = await service.answer(UpdateCheckAnswer(enabled=False))
    assert (status.latest_version, status.last_checked_at, status.next_check_at) == (None, None, None)
