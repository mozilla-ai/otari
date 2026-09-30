"""Signup, resend-verification and password-reset each hand their mail send to
``BackgroundTasks`` instead of awaiting it inline in the request path.

Enumeration-safety on these three endpoints depends on the eligible and
ineligible branches taking indistinguishable wall-clock time (see each
service function's own docstring in ``user_service.py``). An awaited SMTP
round-trip on the eligible branch alone reopens that gap; a real wall-clock
assertion would be flaky, so this intercepts ``BackgroundTasks.add_task``
instead and checks that the send is scheduled, never run inline, by asserting
the console transport has not logged anything by the time the response comes
back.
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
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text


def _intercept_add_task(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, tuple[object, ...], dict[str, object]]]:
    """Capture every scheduled task without running it.

    Not calling through to the real ``add_task`` is deliberate: if a send were
    still awaited inline, the console transport would already have logged by
    the time this spy is even reached, so the request-time log check below
    would fail regardless of whether this spy also runs the task.
    """
    scheduled: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def spy(self: BackgroundTasks, func: object, *args: object, **kwargs: object) -> None:
        scheduled.append((func, args, kwargs))

    monkeypatch.setattr(BackgroundTasks, "add_task", spy)
    return scheduled


def _post_with_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture, call: Callable[[], Response]
) -> Response:
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
            lambda: client.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled_not_sent(scheduled, caplog, to="ada@example.com")


def test_resend_verification_schedules_mail_instead_of_awaiting_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com")
        # Claim the identity first (unverified after this), unrelated to the
        # backgrounding under test, so the send here is not intercepted.
        signup = client.post("/v1/auth/signup", json={"email": "grace@example.com", "password": PASSWORD})
        assert signup.status_code == 200, signup.text

        scheduled = _intercept_add_task(monkeypatch)
        response = _post_with_logs(
            client,
            caplog,
            lambda: client.post("/v1/auth/resend-verification", json={"email": "grace@example.com"}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled_not_sent(scheduled, caplog, to="grace@example.com")


def test_password_reset_request_schedules_mail_instead_of_awaiting_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="hopper@example.com")
        signup = client.post("/v1/auth/signup", json={"email": "hopper@example.com", "password": PASSWORD})
        assert signup.status_code == 200, signup.text

        scheduled = _intercept_add_task(monkeypatch)
        response = _post_with_logs(
            client,
            caplog,
            lambda: client.post("/v1/auth/password/reset", json={"email": "hopper@example.com"}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled_not_sent(scheduled, caplog, to="hopper@example.com")
