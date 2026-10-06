"""Signup, resend-verification and password-reset schedule their mail send
instead of awaiting it, so response time does not reveal which branch ran.

A wall-clock assertion would be flaky, so these intercept
``BackgroundTasks.add_task`` and check that nothing was mailed during the request.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from httpx2 import Response

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.services.mail import Mailer

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret


def _config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'signup-bg-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        mail_transport="console",
        public_base_url="https://gw.example.com",
    )


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_config(tmp_path)))


def _add_member(client: TestClient, *, email: str) -> None:
    response = client.post(
        "/api/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text


def _intercept_add_task(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, tuple[object, ...], dict[str, object]]]:
    """Capture every scheduled task without running it."""
    scheduled: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def spy(self: BackgroundTasks, func: object, *args: object, **kwargs: object) -> None:
        scheduled.append((func, args, kwargs))

    monkeypatch.setattr(BackgroundTasks, "add_task", spy)
    return scheduled


def _post_with_logs(client: TestClient, caplog: pytest.LogCaptureFixture, call: Callable[[], Response]) -> Response:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    try:
        return call()
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _assert_mail_scheduled_not_sent(
    scheduled: list[tuple[object, tuple[object, ...], dict[str, object]]],
    caplog: pytest.LogCaptureFixture,
    *,
    to: str,
) -> None:
    assert not any("[mail:console]" in record.message for record in caplog.records), (
        "mail was logged during the request instead of being scheduled for after it"
    )
    assert len(scheduled) == 1, scheduled
    func, _args, kwargs = scheduled[0]
    assert func.__func__ is Mailer.send  # type: ignore[attr-defined]
    assert kwargs["to"] == to


def test_signup_schedules_verification_mail_instead_of_awaiting_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled = _intercept_add_task(monkeypatch)
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")

        response = _post_with_logs(
            client,
            caplog,
            lambda: client.post("/api/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled_not_sent(scheduled, caplog, to="ada@example.com")


def test_resend_verification_schedules_mail_instead_of_awaiting_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com")
        # Leaves the identity claimed and unverified; this send is not intercepted.
        signup = client.post("/api/v1/auth/signup", json={"email": "grace@example.com", "password": PASSWORD})
        assert signup.status_code == 200, signup.text

        scheduled = _intercept_add_task(monkeypatch)
        response = _post_with_logs(
            client,
            caplog,
            lambda: client.post("/api/v1/auth/resend-verification", json={"email": "grace@example.com"}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled_not_sent(scheduled, caplog, to="grace@example.com")


def test_password_reset_request_schedules_mail_instead_of_awaiting_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="hopper@example.com")
        signup = client.post("/api/v1/auth/signup", json={"email": "hopper@example.com", "password": PASSWORD})
        assert signup.status_code == 200, signup.text

        scheduled = _intercept_add_task(monkeypatch)
        response = _post_with_logs(
            client,
            caplog,
            lambda: client.post("/api/v1/auth/password/reset", json={"email": "hopper@example.com"}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled_not_sent(scheduled, caplog, to="hopper@example.com")
