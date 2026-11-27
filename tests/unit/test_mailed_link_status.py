"""Checking a mailed reset or verification link on arrival, without using it."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import create_engine, text

from _mail_support import MAIL_CONFIGURED, RecordingTransport, token_in
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret
NEW_PASSWORD = "a-recovered-password"  # pragma: allowlist secret
EMAIL = "ada@example.com"

_DB_NAME = "link-status-test.db"
_RESET = f"{API_ROOT}/auth/password/reset/validate"
_VERIFY = f"{API_ROOT}/auth/verify-email/validate"


def _config(tmp_path: Path, **overrides: object) -> GatewayConfig:
    fields: dict[str, object] = {
        "database_url": f"sqlite:///{tmp_path / _DB_NAME}",
        "master_key": MASTER_KEY,
        "require_pricing": False,
        **MAIL_CONFIGURED,
        "public_base_url": "https://gw.example.com",
    }
    fields.update(overrides)
    return GatewayConfig(**fields)  # type: ignore[arg-type]


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(_config(tmp_path, **overrides)))


def _mailed_token(mail: RecordingTransport, call: Callable[[], Response]) -> str:
    """The token in the message a call sent, read from the message the deployment handed its transport."""
    before = len(mail.sent)
    response = call()
    assert response.status_code == 200, response.text
    assert len(mail.sent) == before + 1, f"expected one message, got {len(mail.sent) - before}"
    return token_in(mail.last)


def _signed_up(client: TestClient, mail: RecordingTransport) -> str:
    """Put ``EMAIL`` on the roster and sign it up, returning its verification token."""
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": EMAIL, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert added.status_code == 201, added.text
    return _mailed_token(
        mail, lambda: client.post(f"{API_ROOT}/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    )


def _reset_token(client: TestClient, mail: RecordingTransport) -> str:
    verification = _signed_up(client, mail)
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": verification, "password": PASSWORD})
    assert verified.status_code == 200, verified.text
    return _mailed_token(mail, lambda: client.post(f"{API_ROOT}/auth/password/reset", json={"email": EMAIL}))


def _update_user(tmp_path: Path, statement: str, **params: object) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / _DB_NAME}")
    with engine.begin() as connection:
        connection.execute(text(statement), {"email": EMAIL, **params})
    engine.dispose()


def _expired() -> str:
    return (datetime.now(UTC) - timedelta(hours=1)).isoformat()


def test_a_reset_link_is_valid_until_it_is_used_and_checking_it_does_not_use_it(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        token = _reset_token(client, recorded_mail)

        assert client.post(_RESET, json={"token": token}).json() == {"status": "valid"}
        assert client.post(_RESET, json={"token": token}).json() == {"status": "valid"}

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert confirmed.status_code == 204, confirmed.text
        assert client.post(_RESET, json={"token": token}).json() == {"status": "invalid"}


def test_an_unknown_link_is_invalid(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.post(_RESET, json={"token": "not-a-real-token"}).json() == {"status": "invalid"}
        assert client.post(_VERIFY, json={"token": "not-a-real-token"}).json() == {"status": "invalid"}


def test_an_expired_reset_link_says_so(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        token = _reset_token(client, recorded_mail)

    _update_user(
        tmp_path, 'UPDATE "user" SET password_reset_token_expires_at = :at WHERE email = :email', at=_expired()
    )

    with _client(tmp_path) as client:
        assert client.post(_RESET, json={"token": token}).json() == {"status": "expired"}


def test_a_deactivated_account_s_link_is_invalid(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        token = _reset_token(client, recorded_mail)

    _update_user(tmp_path, 'UPDATE "user" SET is_active = 0 WHERE email = :email')

    with _client(tmp_path) as client:
        assert client.post(_RESET, json={"token": token}).json() == {"status": "invalid"}


def test_a_verification_link_is_valid_until_it_is_used(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail)

        assert client.post(_VERIFY, json={"token": token}).json() == {"status": "valid"}

        verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert verified.status_code == 200, verified.text
        assert client.post(_VERIFY, json={"token": token}).json() == {"status": "invalid"}


def test_an_expired_verification_link_says_so(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail)

    _update_user(
        tmp_path, 'UPDATE "user" SET email_verification_token_expires_at = :at WHERE email = :email', at=_expired()
    )

    with _client(tmp_path) as client:
        assert client.post(_VERIFY, json={"token": token}).json() == {"status": "expired"}


def test_a_shared_password_link_is_checked_like_a_mailed_one(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    """The link an operator hands over where mail is off opens the same reset page."""
    with _client(tmp_path) as client:
        _reset_token(client, recorded_mail)
        accounts = client.get(f"{API_ROOT}/admin/users", headers={"Otari-Key": MASTER_KEY}).json()["data"]
        account_id = str(next(row["id"] for row in accounts if row["email"] == EMAIL))

    with _client(tmp_path, mail_transport="none", public_base_url=None) as client:
        issued = client.post(f"{API_ROOT}/admin/users/{account_id}/password-link", headers={"Otari-Key": MASTER_KEY})
        assert issued.status_code == 201, issued.text
        token = issued.json()["link"].split("token=", 1)[1]

        assert client.post(_RESET, json={"token": token}).json() == {"status": "valid"}

    _update_user(
        tmp_path, 'UPDATE "user" SET password_reset_token_expires_at = :at WHERE email = :email', at=_expired()
    )

    with _client(tmp_path, mail_transport="none", public_base_url=None) as client:
        assert client.post(_RESET, json={"token": token}).json() == {"status": "expired"}


def test_checking_links_is_throttled_like_the_other_public_auth_routes(tmp_path: Path) -> None:
    with _client(tmp_path, dashboard_login_rate_limit_per_minute=2) as client:
        assert client.post(_RESET, json={"token": "one"}).status_code == 200
        assert client.post(_VERIFY, json={"token": "two"}).status_code == 200

        assert client.post(_RESET, json={"token": "three"}).status_code == 429
