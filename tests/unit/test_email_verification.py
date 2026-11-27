"""Verifying an address, and resending the link that verifies it.

Unit rather than integration, the same reasoning ``test_password_sign_in.py``
gives. Mail is recorded rather than delivered, so a verification link is read
from the message the deployment sent.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from _mail_support import MAIL_CONFIGURED, RecordingTransport, token_in
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
from gateway.services.tenancy import user_service

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret


def _config(tmp_path: Path, **overrides: object) -> GatewayConfig:
    fields: dict[str, object] = {
        "database_url": f"sqlite:///{tmp_path / 'verification-test.db'}",
        "master_key": MASTER_KEY,
        "require_pricing": False,
        **MAIL_CONFIGURED,
        "public_base_url": "https://gw.example.com",
    }
    fields.update(overrides)
    return GatewayConfig(**fields)  # type: ignore[arg-type]


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(_config(tmp_path, **overrides)))


def _add_member(client: TestClient, *, email: str) -> None:
    response = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text


def _signed_up(client: TestClient, mail: RecordingTransport, *, email: str) -> str:
    """Add the identity, sign it up, and return the verification token it was sent."""
    _add_member(client, email=email)
    response = client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return token_in(mail.last)


def _resend(client: TestClient, mail: RecordingTransport, *, email: str) -> tuple[int, int]:
    """The resend's status, and how many messages it sent."""
    before = len(mail.sent)
    response = client.post(f"{API_ROOT}/auth/resend-verification", json={"email": email})
    return response.status_code, len(mail.sent) - before


def test_a_valid_token_verifies_and_the_identity_can_sign_in(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")

        response = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert response.status_code == 200, response.text

        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert signed_in.status_code == 200


def test_a_reused_token_is_refused(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    """The concrete fix over the platform's reset token: a consumed token cannot be replayed."""
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")

        first = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert first.status_code == 200

        second = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert second.status_code == 400


def test_an_unknown_token_is_refused(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            f"{API_ROOT}/auth/verify-email", json={"token": "not-a-real-token", "password": PASSWORD}
        )
        assert response.status_code == 400


def test_an_expired_token_is_refused(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")

    engine = create_engine(f"sqlite:///{tmp_path / 'verification-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "user" SET email_verification_token_expires_at = :expired WHERE email = :email'),
            {"expired": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "email": "ada@example.com"},
        )

    with _client(tmp_path) as client:
        response = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert response.status_code == 400


def test_a_token_is_refused_once_the_identity_is_deactivated(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    """A live token must not outlive the account it was issued to.

    Matches ``authenticate``'s own ``is_active`` check: deactivating someone
    has to end every road back in, not just sign-in.
    """
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")

    engine = create_engine(f"sqlite:///{tmp_path / 'verification-test.db'}")
    with engine.begin() as connection:
        connection.execute(text('UPDATE "user" SET is_active = 0 WHERE email = :email'), {"email": "ada@example.com"})

    with _client(tmp_path) as client:
        response = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert response.status_code == 400


def test_resend_for_an_unknown_address_sends_nothing(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        status_code, sent = _resend(client, recorded_mail, email="grace@example.com")

        assert status_code == 200
        assert sent == 0


def test_resend_for_an_address_that_never_signed_up_sends_nothing(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com")

        status_code, sent = _resend(client, recorded_mail, email="grace@example.com")

        assert status_code == 200
        assert sent == 0


def test_resend_for_an_already_verified_address_sends_nothing(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="grace@example.com")
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )

        status_code, sent = _resend(client, recorded_mail, email="grace@example.com")

        assert status_code == 200
        assert sent == 0


def test_resend_for_a_deactivated_identity_sends_nothing(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    """Deactivation closes this road too, the same as the redemption it would lead to."""
    with _client(tmp_path) as client:
        _signed_up(client, recorded_mail, email="ada@example.com")

        engine = create_engine(f"sqlite:///{tmp_path / 'verification-test.db'}")
        with engine.begin() as connection:
            connection.execute(
                text('UPDATE "user" SET is_active = 0 WHERE email = :email'), {"email": "ada@example.com"}
            )
        engine.dispose()

        status_code, sent = _resend(client, recorded_mail, email="ada@example.com")

        assert status_code == 200
        assert sent == 0


def test_resend_sends_a_fresh_link_for_a_genuinely_unverified_identity(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        first_token = _signed_up(client, recorded_mail, email="ada@example.com")

        status_code, sent = _resend(client, recorded_mail, email="ada@example.com")
        assert (status_code, sent) == (200, 1)
        second_token = token_in(recorded_mail.last)

        assert second_token != first_token
        # The old link stops working the moment a new one is issued.
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": first_token, "password": PASSWORD}).status_code
            == 400
        )
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": second_token, "password": PASSWORD}).status_code
            == 200
        )


def test_resend_without_mail_configured_is_refused(tmp_path: Path, recorded_mail: RecordingTransport) -> None:
    with _client(tmp_path) as client:
        _signed_up(client, recorded_mail, email="ada@example.com")

    with _client(tmp_path, mail_transport="none", public_base_url=None) as client:
        response = client.post(f"{API_ROOT}/auth/resend-verification", json={"email": "ada@example.com"})
        assert response.status_code == 503
        # Not the central tenancy handler's generic 5xx body: this refusal has
        # to name what is missing, the same as GET /api/v1/settings/mail's own.
        assert "mail_transport" in response.json()["detail"]


def test_repeated_resend_calls_get_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))) as client:
        for _ in range(2):
            response = client.post(f"{API_ROOT}/auth/resend-verification", json={"email": "nobody@example.com"})
            assert response.status_code == 200

        throttled = client.post(f"{API_ROOT}/auth/resend-verification", json={"email": "nobody@example.com"})
        assert throttled.status_code == 429


def test_a_link_verifies_only_with_the_password_chosen_at_signup(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")

        wrong = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": "not-the-password"})
        assert wrong.status_code == 400
        # A mismatch consumes nothing, so a typo does not cost the link, and the refusal says the link still works.
        assert "This link still works" in wrong.json()["detail"]
        right = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert right.status_code == 200, right.text


def test_a_dead_link_is_refused_without_mentioning_the_password(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )

        spent = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": "not-the-password"})
        unknown = client.post(f"{API_ROOT}/auth/verify-email", json={"token": "not-a-real-token", "password": PASSWORD})

        assert spent.status_code == unknown.status_code == 400
        assert (
            spent.json() == unknown.json() == {"detail": "This verification link is invalid, expired, or already used"}
        )


def test_a_link_minted_before_deactivation_stays_dead_after_reactivation(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    """Deactivating an account clears its pending links, so reactivating it does not bring one back."""
    with _client(tmp_path) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")
        accounts = client.get(f"{API_ROOT}/admin/users", headers={"Otari-Key": MASTER_KEY}).json()["data"]
        account_id = next(row["id"] for row in accounts if row["email"] == "ada@example.com")
        for is_active in (False, True):
            changed = client.patch(
                f"{API_ROOT}/admin/users/{account_id}", json={"is_active": is_active}, headers={"Otari-Key": MASTER_KEY}
            )
            assert changed.status_code == 200, changed.text

        response = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})

        assert response.status_code == 400


def test_a_stranger_who_signs_up_an_invited_address_cannot_have_its_owner_verify_them(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    """The owner receives the link but not the stranger's password, and takes the account back with a reset."""
    stranger_password = "chosen-by-a-stranger"  # pragma: allowlist secret
    owner_password = "chosen-by-the-owner"  # pragma: allowlist secret
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        _, signup_mail = recorded_mail.sent_during(
            lambda: client.post(
                f"{API_ROOT}/auth/signup", json={"email": "ada@example.com", "password": stranger_password}
            )
        )
        assert (
            client.post(
                f"{API_ROOT}/auth/verify-email",
                json={"token": token_in(signup_mail), "password": owner_password},
            ).status_code
            == 400
        )

        client.post(f"{API_ROOT}/auth/password/reset", json={"email": "ada@example.com"})
        reset_token = token_in(recorded_mail.last)
        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": reset_token, "new_password": owner_password}
        )
        assert confirmed.status_code == 204, confirmed.text

        owner = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": owner_password})
        assert owner.status_code == 200, owner.text
        client.cookies.clear()
        stranger = client.post(
            f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": stranger_password}
        )
        assert stranger.status_code == 401


def _invite(client: TestClient, *, email: str) -> str:
    response = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["invitation_id"])


def _revoke(client: TestClient, invitation_id: str) -> None:
    response = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{invitation_id}", headers={"Otari-Key": MASTER_KEY}
    )
    assert response.status_code == 200, response.text


def _signed_up_invitee(client: TestClient, mail: RecordingTransport, *, email: str) -> tuple[str, str]:
    """Invite the address and sign it up; return the invitation id and the verification token."""
    invitation_id = _invite(client, email=email)
    response, envelope = mail.sent_during(
        lambda: client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": PASSWORD})
    )
    assert response.status_code == 200, response.text
    return invitation_id, token_in(envelope)


def test_a_link_stops_verifying_once_the_invitation_behind_it_is_revoked(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    """Closed signup's rule holds when the link is used, not only when it was mailed."""
    with _client(tmp_path) as client:
        invitation_id, token = _signed_up_invitee(client, recorded_mail, email="eve@example.com")
        _revoke(client, invitation_id)

        response = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})

        assert response.status_code == 400


def test_a_link_still_verifies_a_revoked_invitee_where_signup_is_open(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        invitation_id, token = _signed_up_invitee(client, recorded_mail, email="eve@example.com")
        _revoke(client, invitation_id)

        response = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})

        assert response.status_code == 200, response.text


def test_resend_sends_nothing_once_the_invitation_behind_it_is_revoked(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with _client(tmp_path) as client:
        invitation_id, _ = _signed_up_invitee(client, recorded_mail, email="eve@example.com")
        _revoke(client, invitation_id)

        status_code, sent = _resend(client, recorded_mail, email="eve@example.com")

        assert status_code == 200
        assert sent == 0


def test_failed_sign_ins_do_not_lock_anyone_out_of_signing_up_or_verifying(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    with TestClient(create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")
        for _ in range(3):
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": "a-typo-1234"})
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": "a-typo-1234"}
            ).status_code
            == 429
        )

        assert client.post(f"{API_ROOT}/auth/resend-verification", json={"email": "x@example.com"}).status_code == 200
        verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert verified.status_code == 200, verified.text


def test_failed_verifications_are_throttled_before_the_password_is_checked(
    tmp_path: Path, recorded_mail: RecordingTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held link cannot be used to guess its password, nor to make the gateway hash at will."""
    with TestClient(create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))) as client:
        token = _signed_up(client, recorded_mail, email="ada@example.com")
        for _ in range(2):
            wrong = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": "a-wrong-guess"})
            assert wrong.status_code == 400

        checked: list[str] = []

        async def recording_verify(password: str, hashed: str) -> bool:
            checked.append(password)
            return False

        monkeypatch.setattr(user_service, "verify_password_async", recording_verify)
        throttled = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert throttled.status_code == 429
        assert checked == []


def test_a_successful_verification_spends_none_of_the_failure_budget(
    tmp_path: Path, recorded_mail: RecordingTransport
) -> None:
    def verify(client: TestClient, token: str, password: str) -> int:
        return client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": password}).status_code

    with TestClient(create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))) as client:
        first = _signed_up(client, recorded_mail, email="ada@example.com")
        second = _signed_up(client, recorded_mail, email="grace@example.com")

        assert verify(client, first, "a-typo-1234") == 400
        assert verify(client, first, PASSWORD) == 200
        # One failure is left: the success above was not counted against it.
        assert verify(client, second, "a-typo-1234") == 400
        assert verify(client, second, "a-typo-1234") == 429
