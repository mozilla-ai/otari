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

import logging
import re
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger

PASSWORD = "a-real-password"  # pragma: allowlist secret
NEW_PASSWORD = "a-recovered-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _with_logs(caplog: pytest.LogCaptureFixture, call: Callable[[], Response]) -> Response:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    try:
        return call()
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _extract_token(text: str) -> str:
    """Pull the token out of a link the console transport logged, asserting it is there."""
    match = _TOKEN_IN_LINK.search(text)
    assert match, text
    return match.group(1)


def _claimed_and_verified(
    client: TestClient, master_key_header: dict[str, str], caplog: pytest.LogCaptureFixture, *, email: str
) -> None:
    assert client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers=master_key_header,
    ).status_code == 201

    signup = _with_logs(caplog, lambda: client.post("/v1/auth/signup", json={"email": email, "password": PASSWORD}))
    assert signup.status_code == 200, signup.text
    token = _extract_token(caplog.text)
    assert client.post("/v1/auth/verify-email", json={"token": token}).status_code == 200


def test_reset_request_then_confirm_then_sign_in(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")

    _claimed_and_verified(client, master_key_header, caplog, email="ada@example.com")

    requested = _with_logs(
        caplog, lambda: client.post("/v1/auth/password/reset", json={"email": "ada@example.com"})
    )
    assert requested.status_code == 200, requested.text
    token = _extract_token(caplog.text)

    confirmed = client.post(
        "/v1/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
    )
    assert confirmed.status_code == 204, confirmed.text

    signed_in = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD})
    assert signed_in.status_code == 200, signed_in.text

    stale = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert stale.status_code == 401
