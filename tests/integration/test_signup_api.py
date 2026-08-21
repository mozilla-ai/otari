"""Signup and email verification, end to end against PostgreSQL.

The unit suite (``test_signup.py``, ``test_email_verification.py``) covers the
edge cases against SQLite; what is under test here is the same happy path and
the sign-in gate running against the real engine. Throttling is not exercised
here: ``test_invitation_rate_limit.py``'s own docstring explains why the
shared, Postgres-backed ``client``/``test_config`` fixtures cannot pin a low
rate limit (the limiter is built into the app at boot from a session-scoped
config, before a test's own ``monkeypatch`` call runs).
``test_email_verification.py::test_repeated_resend_calls_get_throttled``
covers it instead, against SQLite, the same way ``test_invitation_rate_limit.py``
does for the invitation routes; it exercises the same
``_public_auth.throttle_public_auth`` this module's own signup and
verify-email routes share.
"""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger

PASSWORD = "a-real-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _add_member(client: TestClient, master_key_header: dict[str, str], *, email: str) -> None:
    response = client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers=master_key_header,
    )
    assert response.status_code == 201, response.text


def _signed_up_token(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    *,
    email: str,
) -> str:
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


def test_signup_then_verify_then_sign_in(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")

    _add_member(client, master_key_header, email="ada@example.com")
    token = _signed_up_token(client, caplog, email="ada@example.com")

    unverified = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert unverified.status_code == 403

    verified = client.post("/v1/auth/verify-email", json={"token": token})
    assert verified.status_code == 200, verified.text
    assert verified.json()["email"] == "ada@example.com"

    signed_in = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text


def test_signup_on_an_untouched_address_is_enumeration_safe(
    client: TestClient,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 with nothing written, not a 404: see ``test_signup.py`` for the reasoning."""
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")

    response = client.post("/v1/auth/signup", json={"email": "nobody@example.com", "password": PASSWORD})
    assert response.status_code == 200
