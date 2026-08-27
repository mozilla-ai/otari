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


def test_invited_member_accepts_then_claims_the_invited_address(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The path an invitee actually walks, which the test above does not cover.

    ``_add_member`` lands a membership ``active`` directly; an invitation lands
    it ``invited`` and the identity stays password-less until the accept page
    hands the recipient to signup (otari#835). Both ends existed and were
    tested; the join between them was not, so nothing failed if the address the
    accept page previews stopped being the address signup can claim.
    """
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")

    invited = client.post(
        "/v1/organizations/me/member-invitations",
        json={"email": "grace@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert invited.status_code == 201, invited.text
    accept_token = _TOKEN_IN_LINK.search(invited.json()["accept_link"])
    assert accept_token, invited.json()["accept_link"]

    # The address the claim is bound to is the one the preview publishes, which
    # is what lets the accept page prefill it without a second endpoint.
    preview = client.post("/v1/invitations/validate", json={"token": accept_token.group(1)})
    assert preview.status_code == 200, preview.text
    previewed_email = preview.json()["email"]
    assert previewed_email == "grace@example.com"

    accepted = client.post("/v1/invitations/accept", json={"token": accept_token.group(1)})
    assert accepted.status_code == 200, accepted.text

    # Spending the token is what the accept page's branch order protects: the
    # preview refuses from here on, over a membership the visitor already holds.
    spent = client.post("/v1/invitations/validate", json={"token": accept_token.group(1)})
    assert spent.status_code == 400, spent.text

    verification_token = _signed_up_token(client, caplog, email=previewed_email)
    verified = client.post("/v1/auth/verify-email", json={"token": verification_token})
    assert verified.status_code == 200, verified.text

    signed_in = client.post("/v1/auth/session", json={"email": previewed_email, "password": PASSWORD})
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
