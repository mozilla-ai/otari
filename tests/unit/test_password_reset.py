"""Password reset: request a link, then complete it with the token it carries.

Unit rather than integration, the same reasoning ``test_password_sign_in.py``
gives. Mail runs on the console transport so a reset link can be read back out
of the log, the same pattern ``test_invitations_api.py`` uses.
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.models.entities import DashboardSession

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret
NEW_PASSWORD = "a-recovered-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _config(tmp_path: Path, **overrides: object) -> GatewayConfig:
    fields: dict[str, object] = {
        "database_url": f"sqlite:///{tmp_path / 'reset-test.db'}",
        "master_key": MASTER_KEY,
        "require_pricing": False,
        "mail_transport": "console",
        "public_base_url": "https://gw.example.com",
    }
    fields.update(overrides)
    return GatewayConfig(**fields)


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(_config(tmp_path, **overrides)))


def _with_logs(client: TestClient, caplog: pytest.LogCaptureFixture, call):  # noqa: ANN001, ANN202
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    try:
        return call()
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _claimed_and_verified(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> None:
    """Get an identity onto the roster, signed up, and verified, ready to sign in."""
    assert client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    ).status_code == 201

    signup = _with_logs(
        client, caplog, lambda: client.post("/v1/auth/signup", json={"email": email, "password": PASSWORD})
    )
    assert signup.status_code == 200, signup.text
    token = _TOKEN_IN_LINK.search(caplog.text).group(1)
    assert client.post("/v1/auth/verify-email", json={"token": token}).status_code == 200


def _request_reset(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> tuple[int, str]:
    response = _with_logs(client, caplog, lambda: client.post("/v1/auth/password/reset", json={"email": email}))
    return response.status_code, caplog.text


def _sessions(tmp_path: Path, db_name: str) -> list[DashboardSession]:
    engine = create_engine(f"sqlite:///{tmp_path / db_name}")
    with sessionmaker(bind=engine)() as session:
        return list(session.execute(select(DashboardSession)).scalars().all())


def test_reset_round_trips_end_to_end(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        token = _TOKEN_IN_LINK.search(log_text).group(1)

        confirmed = client.post(
            "/v1/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert confirmed.status_code == 204, confirmed.text

        assert client.post(
            "/v1/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD}
        ).status_code == 200
        assert client.post(
            "/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
        ).status_code == 401


def test_reset_works_before_the_address_is_verified(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Forgetting a password predates ever verifying it."""
    with _client(tmp_path) as client:
        assert client.post(
            "/v1/organizations/me/members",
            json={"email": "ada@example.com", "role": "member"},
            headers={"Otari-Key": MASTER_KEY},
        ).status_code == 201
        signup = _with_logs(
            client,
            caplog,
            lambda: client.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD}),
        )
        assert signup.status_code == 200

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        token = _TOKEN_IN_LINK.search(log_text).group(1)

        confirmed = client.post(
            "/v1/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert confirmed.status_code == 204, confirmed.text


def test_reset_revokes_the_identity_s_other_sessions(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        assert client.post(
            "/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
        ).status_code == 200

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        token = _TOKEN_IN_LINK.search(log_text).group(1)
        client.post("/v1/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD})

    assert _sessions(tmp_path, "reset-test.db") == []


def test_a_reused_reset_token_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The concrete fix over the platform's stateless reset JWT: single use, enforced."""
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        token = _TOKEN_IN_LINK.search(log_text).group(1)

        first = client.post("/v1/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD})
        assert first.status_code == 204

        second = client.post(
            "/v1/auth/password/reset/confirm", json={"token": token, "new_password": "yet-another-password"}
        )
        assert second.status_code == 400


def test_an_unknown_reset_token_is_refused(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/v1/auth/password/reset/confirm", json={"token": "not-a-real-token", "new_password": NEW_PASSWORD}
        )
        assert response.status_code == 400


def test_an_expired_reset_token_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        token = _TOKEN_IN_LINK.search(log_text).group(1)

    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "user" SET password_reset_token_expires_at = :expired WHERE email = :email'),
            {"expired": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "email": "ada@example.com"},
        )

    with _client(tmp_path) as client:
        response = client.post(
            "/v1/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert response.status_code == 400


def test_request_reset_is_enumeration_safe(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")

        known_status, known_text = _request_reset(client, caplog, email="ada@example.com")
        unknown_status, unknown_text = _request_reset(client, caplog, email="nobody@example.com")

        assert known_status == unknown_status == 200
        assert "mail:console" in known_text
        assert "mail:console" not in unknown_text


def test_request_reset_without_mail_configured_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")

    with _client(tmp_path, mail_transport="none", public_base_url=None) as client:
        response = client.post("/v1/auth/password/reset", json={"email": "ada@example.com"})
        assert response.status_code == 503
        # Not the central tenancy handler's generic 5xx body: this refusal has
        # to name what is missing, the same as GET /v1/settings/mail's own.
        assert "mail_transport" in response.json()["detail"]


def test_repeated_reset_requests_get_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))) as client:
        for _ in range(2):
            response = client.post("/v1/auth/password/reset", json={"email": "nobody@example.com"})
            assert response.status_code == 200

        throttled = client.post("/v1/auth/password/reset", json={"email": "nobody@example.com"})
        assert throttled.status_code == 429
