"""Checking a mailed reset link on arrival, end to end against PostgreSQL.

The unit suite (``test_mailed_link_status.py``) covers expiry, deactivation,
the verification link and throttling against SQLite; what is under test here is
that a check reads the token without using it on the real engine.
"""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger

PASSWORD = "a-real-password"  # pragma: allowlist secret
NEW_PASSWORD = "a-recovered-password"  # pragma: allowlist secret
EMAIL = "ada@example.com"

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _mailed_token(caplog: pytest.LogCaptureFixture, client: TestClient, path: str, body: dict[str, str]) -> str:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    try:
        response = client.post(f"{API_ROOT}{path}", json=body)
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert response.status_code == 200, response.text
    match = _TOKEN_IN_LINK.search(caplog.text)
    assert match, caplog.text
    return match.group(1)


def test_checking_links_never_uses_them(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")
    added = client.post(
        f"{API_ROOT}/organizations/me/members", json={"email": EMAIL, "role": "member"}, headers=master_key_header
    )
    assert added.status_code == 201, added.text

    verification = _mailed_token(caplog, client, "/auth/signup", {"email": EMAIL, "password": PASSWORD})
    check = client.post(f"{API_ROOT}/auth/verify-email/validate", json={"token": verification})
    assert check.json() == {"status": "valid"}
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": verification, "password": PASSWORD})
    assert verified.status_code == 200, verified.text

    reset = _mailed_token(caplog, client, "/auth/password/reset", {"email": EMAIL})
    assert client.post(f"{API_ROOT}/auth/password/reset/validate", json={"token": reset}).json() == {"status": "valid"}
    confirmed = client.post(
        f"{API_ROOT}/auth/password/reset/confirm", json={"token": reset, "new_password": NEW_PASSWORD}
    )
    assert confirmed.status_code == 204, confirmed.text

    assert client.post(f"{API_ROOT}/auth/password/reset/validate", json={"token": reset}).json() == {
        "status": "invalid"
    }
    assert client.post(f"{API_ROOT}/auth/verify-email/validate", json={"token": verification}).json() == {
        "status": "invalid"
    }
