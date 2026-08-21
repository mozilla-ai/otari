"""Verifying an address, and resending the link that verifies it.

Unit rather than integration, the same reasoning ``test_password_sign_in.py``
gives. Mail runs on the console transport so a verification link can be read
back out of the log, the same pattern ``test_invitations_api.py`` uses.
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _config(tmp_path: Path, **overrides: object) -> GatewayConfig:
    fields: dict[str, object] = {
        "database_url": f"sqlite:///{tmp_path / 'verification-test.db'}",
        "master_key": MASTER_KEY,
        "require_pricing": False,
        "mail_transport": "console",
        "public_base_url": "https://gw.example.com",
    }
    fields.update(overrides)
    return GatewayConfig(**fields)  # type: ignore[arg-type]


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(_config(tmp_path, **overrides)))


def _add_member(client: TestClient, *, email: str) -> None:
    response = client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text


def _signed_up(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> str:
    """Add the identity, sign it up, and return the verification token from the mail log."""
    _add_member(client, email=email)
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        response = client.post("/v1/auth/signup", json={"email": email, "password": PASSWORD})
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert response.status_code == 200, response.text
    match = _TOKEN_IN_LINK.search(caplog.text)
    assert match, caplog.text
    return match.group(1)


def _resend(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> tuple[int, str]:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    try:
        response = client.post("/v1/auth/resend-verification", json={"email": email})
    finally:
        gateway_logger.removeHandler(caplog.handler)
    return response.status_code, caplog.text


def test_a_valid_token_verifies_and_the_identity_can_sign_in(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, caplog, email="ada@example.com")

        response = client.post("/v1/auth/verify-email", json={"token": token})
        assert response.status_code == 200, response.text

        signed_in = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert signed_in.status_code == 200


def test_a_reused_token_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The concrete fix over the platform's reset token: a consumed token cannot be replayed."""
    with _client(tmp_path) as client:
        token = _signed_up(client, caplog, email="ada@example.com")

        first = client.post("/v1/auth/verify-email", json={"token": token})
        assert first.status_code == 200

        second = client.post("/v1/auth/verify-email", json={"token": token})
        assert second.status_code == 400


def test_an_unknown_token_is_refused(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/v1/auth/verify-email", json={"token": "not-a-real-token"})
        assert response.status_code == 400


def test_an_expired_token_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, caplog, email="ada@example.com")

    engine = create_engine(f"sqlite:///{tmp_path / 'verification-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "user" SET email_verification_token_expires_at = :expired WHERE email = :email'),
            {"expired": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "email": "ada@example.com"},
        )

    with _client(tmp_path) as client:
        response = client.post("/v1/auth/verify-email", json={"token": token})
        assert response.status_code == 400


def test_a_token_is_refused_once_the_identity_is_deactivated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A live token must not outlive the account it was issued to.

    Matches ``authenticate``'s own ``is_active`` check: deactivating someone
    has to end every road back in, not just sign-in.
    """
    with _client(tmp_path) as client:
        token = _signed_up(client, caplog, email="ada@example.com")

    engine = create_engine(f"sqlite:///{tmp_path / 'verification-test.db'}")
    with engine.begin() as connection:
        connection.execute(text('UPDATE "user" SET is_active = 0 WHERE email = :email'), {"email": "ada@example.com"})

    with _client(tmp_path) as client:
        response = client.post("/v1/auth/verify-email", json={"token": token})
        assert response.status_code == 400


def test_resend_for_an_unknown_address_sends_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        status_code, text_seen = _resend(client, caplog, email="grace@example.com")

        assert status_code == 200
        assert "mail:console" not in text_seen


def test_resend_for_an_address_that_never_signed_up_sends_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com")

        status_code, text_seen = _resend(client, caplog, email="grace@example.com")

        assert status_code == 200
        assert "mail:console" not in text_seen


def test_resend_for_an_already_verified_address_sends_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, caplog, email="grace@example.com")
        assert client.post("/v1/auth/verify-email", json={"token": token}).status_code == 200

        status_code, text_seen = _resend(client, caplog, email="grace@example.com")

        assert status_code == 200
        assert "mail:console" not in text_seen


def test_resend_sends_a_fresh_link_for_a_genuinely_unverified_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        first_token = _signed_up(client, caplog, email="ada@example.com")

        status_code, text_seen = _resend(client, caplog, email="ada@example.com")
        assert status_code == 200
        match = _TOKEN_IN_LINK.search(text_seen)
        assert match, text_seen
        second_token = match.group(1)

        assert second_token != first_token
        # The old link stops working the moment a new one is issued.
        assert client.post("/v1/auth/verify-email", json={"token": first_token}).status_code == 400
        assert client.post("/v1/auth/verify-email", json={"token": second_token}).status_code == 200


def test_resend_without_mail_configured_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _signed_up(client, caplog, email="ada@example.com")

    with _client(tmp_path, mail_transport="none", public_base_url=None) as client:
        response = client.post("/v1/auth/resend-verification", json={"email": "ada@example.com"})
        assert response.status_code == 503
        # Not the central tenancy handler's generic 5xx body: this refusal has
        # to name what is missing, the same as GET /v1/settings/mail's own.
        assert "mail_transport" in response.json()["detail"]


def test_repeated_resend_calls_get_throttled(tmp_path: Path) -> None:
    with TestClient(
        create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))
    ) as client:
        for _ in range(2):
            response = client.post("/v1/auth/resend-verification", json={"email": "nobody@example.com"})
            assert response.status_code == 200

        throttled = client.post("/v1/auth/resend-verification", json={"email": "nobody@example.com"})
        assert throttled.status_code == 429
