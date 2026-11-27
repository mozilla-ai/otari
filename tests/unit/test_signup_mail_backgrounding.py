"""Signup, resend-verification and password-reset schedule their mail send
instead of awaiting it, so response time does not reveal which branch ran.

A wall-clock assertion would be flaky, so these intercept
``BackgroundTasks.add_task`` and check that nothing was mailed during the request.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from httpx2 import Response

from _mail_support import MAIL_CONFIGURED, RecordingTransport
from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.services.mail import Mailer

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret


def _config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'signup-bg-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        **MAIL_CONFIGURED,  # type: ignore[arg-type]
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


def _request_sending_nothing(mail: RecordingTransport, call: Callable[[], Response]) -> Response:
    """Make the request, holding it to mailing nothing while it runs.

    NOTE: The baseline is read per request rather than assumed empty, because
    two of these tests sign an identity up first and that send is a real one.
    """
    before = len(mail.sent)
    response = call()
    during = len(mail.sent) - before
    assert during == 0, f"the request sent {during} messages instead of scheduling them"
    return response


def _assert_mail_scheduled(
    scheduled: list[tuple[object, tuple[object, ...], dict[str, object]]],
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_mail: RecordingTransport,
) -> None:
    scheduled = _intercept_add_task(monkeypatch)
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")

        response = _request_sending_nothing(
            recorded_mail,
            lambda: client.post("/api/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled(scheduled, to="ada@example.com")


def test_resend_verification_schedules_mail_instead_of_awaiting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_mail: RecordingTransport,
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com")
        # Leaves the identity claimed and unverified; this send is not intercepted.
        signup = client.post("/api/v1/auth/signup", json={"email": "grace@example.com", "password": PASSWORD})
        assert signup.status_code == 200, signup.text

        scheduled = _intercept_add_task(monkeypatch)
        response = _request_sending_nothing(
            recorded_mail,
            lambda: client.post("/api/v1/auth/resend-verification", json={"email": "grace@example.com"}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled(scheduled, to="grace@example.com")


def test_password_reset_request_schedules_mail_instead_of_awaiting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_mail: RecordingTransport,
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="hopper@example.com")
        signup = client.post("/api/v1/auth/signup", json={"email": "hopper@example.com", "password": PASSWORD})
        assert signup.status_code == 200, signup.text

        scheduled = _intercept_add_task(monkeypatch)
        response = _request_sending_nothing(
            recorded_mail,
            lambda: client.post("/api/v1/auth/password/reset", json={"email": "hopper@example.com"}),
        )

    assert response.status_code == 200, response.text
    _assert_mail_scheduled(scheduled, to="hopper@example.com")
