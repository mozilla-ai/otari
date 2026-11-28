"""Password reset, end to end against PostgreSQL.

The unit suite (``test_password_reset.py``) covers the edge cases against
SQLite; what is under test here is the same round trip running against the
real engine. Throttling is not exercised here for the same reason
``test_invitation_rate_limit.py`` gives: the shared, Postgres-backed
``client``/``test_config`` fixtures build the rate limiter into the app at
boot from a session-scoped config, before a test's own ``monkeypatch`` call
runs. ``test_password_reset.py::test_repeated_reset_requests_get_throttled``
covers it instead, against SQLite.
"""

import re

import pytest
from fastapi.testclient import TestClient

from _mail_support import RecordingTransport, configure_mail, token_in
from gateway.core.config import API_ROOT, GatewayConfig

PASSWORD = "a-real-password"  # pragma: allowlist secret
NEW_PASSWORD = "a-recovered-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _claimed_and_verified(
    client: TestClient, master_key_header: dict[str, str], mail: RecordingTransport, *, email: str
) -> None:
    assert (
        client.post(
            f"{API_ROOT}/organizations/me/members",
            json={"email": email, "role": "member"},
            headers=master_key_header,
        ).status_code
        == 201
    )

    signup = client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": PASSWORD})
    assert signup.status_code == 200, signup.text
    token = token_in(mail.last)
    assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200


def test_reset_request_then_confirm_then_sign_in(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    recorded_mail: RecordingTransport,
) -> None:
    configure_mail(monkeypatch, test_config)
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")

    _claimed_and_verified(client, master_key_header, recorded_mail, email="ada@example.com")

    requested = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "ada@example.com"})
    assert requested.status_code == 200, requested.text
    token = token_in(recorded_mail.last)

    confirmed = client.post(
        f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
    )
    assert confirmed.status_code == 204, confirmed.text

    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD})
    assert signed_in.status_code == 200, signed_in.text

    stale = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert stale.status_code == 401
