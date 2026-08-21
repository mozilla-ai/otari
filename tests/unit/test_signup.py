"""Signup: claiming an identity ``organization_service`` already put on the roster.

Unit rather than integration, the same reasoning ``test_password_sign_in.py``
gives: everything under test is route, service and identity behavior that runs
unchanged on the SQLite file each test stands up. Mail runs on the console
transport, which logs the rendered message (including the verification link)
rather than delivering it, the same pattern ``test_invitations_api.py`` uses to
observe an emailed link without a real SMTP server.
"""

import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import create_engine, text

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _config(tmp_path: Path, *, mail_ready: bool = True) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'signup-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        mail_transport="console" if mail_ready else "none",
        public_base_url="https://gw.example.com" if mail_ready else None,
    )


def _client(tmp_path: Path, *, mail_ready: bool = True) -> TestClient:
    return TestClient(create_app(_config(tmp_path, mail_ready=mail_ready)))


def _add_member(client: TestClient, *, email: str, role: str = "member") -> None:
    """Put a password-less, unclaimed identity on the roster, as an admin would."""
    response = client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": role},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text


def _signup(client: TestClient, *, email: str, password: str = PASSWORD, **extra: object) -> Response:
    return client.post("/v1/auth/signup", json={"email": email, "password": password, **extra})


def _captured_verification_link(caplog: pytest.LogCaptureFixture, client: TestClient, **kwargs: object) -> str:
    """Sign up, expecting success, and pull the emailed link's token out of the console log."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        response = _signup(client, **kwargs)  # type: ignore[arg-type]
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert response.status_code == 200, response.text
    match = _TOKEN_IN_LINK.search(caplog.text)
    assert match, caplog.text
    return match.group(1)


def test_signup_claims_a_roster_identity_and_sends_a_verification_link(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")

        token = _captured_verification_link(
            caplog, client, email="ada@example.com", full_name="Ada Lovelace"
        )

        # Unverified: the hard-block refuses the very password just set.
        response = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert response.status_code == 403

        verified = client.post("/v1/auth/verify-email", json={"token": token})
        assert verified.status_code == 200, verified.text
        assert verified.json()["email"] == "ada@example.com"

        response = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert response.status_code == 200


def test_signup_preserves_the_existing_membership_and_organization(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com", role="admin")
        headers = {"Otari-Key": MASTER_KEY}

        def _roster_row() -> dict[str, object]:
            members = client.get("/v1/organizations/me/members", headers=headers).json()["data"]
            return next(row for row in members if row["email"] == "grace@example.com")

        before = _roster_row()
        _captured_verification_link(caplog, client, email="grace@example.com")
        after = _roster_row()

        assert after["role"] == before["role"] == "admin"
        assert after["organization_member_id"] == before["organization_member_id"]


def test_signup_on_an_untouched_address_is_enumeration_safe(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An earlier version answered 404 here, letting a caller enumerate the roster."""
    with _client(tmp_path) as client:
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            response = _signup(client, email="nobody@example.com")
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert response.status_code == 200
        assert "mail:console" not in caplog.text


def test_signup_on_an_already_completed_address_is_enumeration_safe(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An earlier version answered 409 here, letting a caller enumerate signup progress."""
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        _captured_verification_link(caplog, client, email="ada@example.com")

        caplog.clear()
        again = _signup(client, email="ada@example.com", password="a-different-password")

        assert again.status_code == 200
        assert again.json() == _signup(client, email="nobody@example.com").json()
        # Nothing was re-sent, and the original password is untouched.
        assert "mail:console" not in caplog.text
        assert client.post(
            "/v1/auth/session", json={"email": "ada@example.com", "password": "a-different-password"}
        ).status_code == 401


def test_signup_password_policy_is_enforced_before_any_enumeration_check(tmp_path: Path) -> None:
    """A policy-violating password answers the same 400 whether or not the address exists.

    Checked first, ahead of the address lookup, so the shape of the failure
    never depends on account state: only the password itself is being judged.
    Longer than bcrypt's 72-byte ceiling rather than merely short, so the
    schema's own ``min_length=8`` does not intercept it as a 422 first.
    """
    with _client(tmp_path) as client:
        too_long = "a" * 100
        unknown = _signup(client, email="nobody@example.com", password=too_long)
        assert unknown.status_code == 400

        _add_member(client, email="ada@example.com")
        known = _signup(client, email="ada@example.com", password=too_long)
        assert known.status_code == 400
        assert known.json() == unknown.json()


def test_signup_without_mail_configured_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    with _client(tmp_path, mail_ready=False) as client:
        _add_member(client, email="ada@example.com")

        response = _signup(client, email="ada@example.com")
        assert response.status_code == 503
        # Not the central tenancy handler's generic 5xx body: this refusal has
        # to name what is missing, the same as GET /v1/settings/mail's own.
        assert "mail_transport" in response.json()["detail"]

    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        row = (
            connection.execute(
                text('SELECT hashed_password, email_verification_token_hash FROM "user" WHERE email = :email'),
                {"email": "ada@example.com"},
            )
            .mappings()
            .one()
        )
    assert row["hashed_password"] is None
    assert row["email_verification_token_hash"] is None
