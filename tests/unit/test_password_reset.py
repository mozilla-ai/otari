"""Password reset: request a link, then complete it with the token it carries.

Unit rather than integration, the same reasoning ``test_password_sign_in.py``
gives. Mail runs on the console transport so a reset link can be read back out
of the log, the same pattern ``test_invitations_api.py`` uses.
"""

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.models.tenancy import DashboardSession, User, WebAuthnCredential

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
    return GatewayConfig(**fields)  # type: ignore[arg-type]


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(_config(tmp_path, **overrides)))


def _with_logs(client: TestClient, caplog: pytest.LogCaptureFixture, call: Callable[[], Response]) -> Response:
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


def _claimed_and_verified(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> None:
    """Get an identity onto the roster, signed up, and verified, ready to sign in."""
    assert (
        client.post(
            f"{API_ROOT}/organizations/me/members",
            json={"email": email, "role": "member"},
            headers={"Otari-Key": MASTER_KEY},
        ).status_code
        == 201
    )

    signup = _with_logs(
        client, caplog, lambda: client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": PASSWORD})
    )
    assert signup.status_code == 200, signup.text
    token = _extract_token(caplog.text)
    assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200


def _request_reset(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> tuple[int, str]:
    response = _with_logs(client, caplog, lambda: client.post(f"{API_ROOT}/auth/password/reset", json={"email": email}))
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
        token = _extract_token(log_text)

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert confirmed.status_code == 204, confirmed.text

        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD}
            ).status_code
            == 200
        )
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
            == 401
        )


def test_reset_works_before_the_address_is_verified(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Forgetting a password predates ever verifying it."""
    with _client(tmp_path) as client:
        assert (
            client.post(
                f"{API_ROOT}/organizations/me/members",
                json={"email": "ada@example.com", "role": "member"},
                headers={"Otari-Key": MASTER_KEY},
            ).status_code
            == 201
        )
        signup = _with_logs(
            client,
            caplog,
            lambda: client.post(f"{API_ROOT}/auth/signup", json={"email": "ada@example.com", "password": PASSWORD}),
        )
        assert signup.status_code == 200

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        token = _extract_token(log_text)

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert confirmed.status_code == 204, confirmed.text


def test_reset_revokes_the_identity_s_other_sessions(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
            == 200
        )

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        token = _extract_token(log_text)
        client.post(f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD})

    assert _sessions(tmp_path, "reset-test.db") == []


def test_a_reused_reset_token_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The concrete fix over the platform's stateless reset JWT: single use, enforced."""
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        token = _extract_token(log_text)

        first = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert first.status_code == 204

        second = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": "yet-another-password"}
        )
        assert second.status_code == 400


def test_an_unknown_reset_token_is_refused(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": "not-a-real-token", "new_password": NEW_PASSWORD}
        )
        assert response.status_code == 400


def test_an_expired_reset_token_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        token = _extract_token(log_text)

    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "user" SET password_reset_token_expires_at = :expired WHERE email = :email'),
            {"expired": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "email": "ada@example.com"},
        )

    with _client(tmp_path) as client:
        response = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert response.status_code == 400


def test_a_reset_token_is_refused_once_the_identity_is_deactivated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A live token must not outlive the account it was issued to.

    Matches ``authenticate``'s own ``is_active`` check: deactivating someone
    has to end every road back in, not just sign-in.
    """
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        token = _extract_token(log_text)

    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        connection.execute(text('UPDATE "user" SET is_active = 0 WHERE email = :email'), {"email": "ada@example.com"})

    with _client(tmp_path) as client:
        response = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert response.status_code == 400


def test_a_password_change_elsewhere_invalidates_an_outstanding_reset_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A reset token minted before a password change has no reason to survive it.

    Without this, a reset link generated and then overtaken by a change
    through another channel (the operator recovers the account before the
    link is opened, an ordinary self-service change) would still work,
    letting whoever holds it undo the change.
    """
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        token = _extract_token(log_text)

        # The account owner changes their password through the ordinary,
        # session-authenticated channel while the reset link is still live.
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert signed_in.status_code == 200
        changed = client.put(
            f"{API_ROOT}/auth/password",
            json={"current_password": PASSWORD, "new_password": "an-unrelated-change"},
        )
        assert changed.status_code == 200, changed.text

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert confirmed.status_code == 400


def test_request_reset_is_enumeration_safe(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")

        known_status, known_text = _request_reset(client, caplog, email="ada@example.com")
        unknown_status, unknown_text = _request_reset(client, caplog, email="nobody@example.com")

        assert known_status == unknown_status == 200
        assert "mail:console" in known_text
        assert "mail:console" not in unknown_text


def test_request_reset_for_a_deactivated_identity_sends_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deactivation closes this road too, matching the refusal ``reset_password`` already makes."""
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")

        engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
        with engine.begin() as connection:
            connection.execute(
                text('UPDATE "user" SET is_active = 0 WHERE email = :email'), {"email": "ada@example.com"}
            )
        engine.dispose()

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")

        assert status_code == 200
        assert "mail:console" not in log_text


def test_request_reset_for_a_verified_identity_with_no_password_sends_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        assert (
            client.post(
                f"{API_ROOT}/organizations/me/members",
                json={"email": "ada@example.com", "role": "member"},
                headers={"Otari-Key": MASTER_KEY},
            ).status_code
            == 201
        )
        engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
        with engine.begin() as connection:
            connection.execute(
                text('UPDATE "user" SET email_verified_at = CURRENT_TIMESTAMP WHERE email = :email'),
                {"email": "ada@example.com"},
            )
        engine.dispose()

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")

        assert status_code == 200
        assert "mail:console" not in log_text

    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        token_hash = connection.execute(
            text('SELECT password_reset_token_hash FROM "user" WHERE email = :email'), {"email": "ada@example.com"}
        ).scalar_one()
    engine.dispose()
    assert token_hash is None


def test_request_reset_without_mail_configured_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")

    with _client(tmp_path, mail_transport="none", public_base_url=None) as client:
        response = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "ada@example.com"})
        assert response.status_code == 503
        # Not the central tenancy handler's generic 5xx body: this refusal has
        # to name what is missing, the same as GET /api/v1/settings/mail's own.
        assert "mail_transport" in response.json()["detail"]


def test_repeated_reset_requests_get_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path, dashboard_login_rate_limit_per_minute=2))) as client:
        for _ in range(2):
            response = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "nobody@example.com"})
            assert response.status_code == 200

        throttled = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "nobody@example.com"})
        assert throttled.status_code == 429


def _reset_token_hash(tmp_path: Path, *, email: str) -> str | None:
    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        token_hash: str | None = connection.execute(
            text('SELECT password_reset_token_hash FROM "user" WHERE email = :email'), {"email": email}
        ).scalar_one()
    engine.dispose()
    return token_hash


def _complete_reset(client: TestClient, log_text: str) -> None:
    confirmed = client.post(
        f"{API_ROOT}/auth/password/reset/confirm",
        json={"token": _extract_token(log_text), "new_password": NEW_PASSWORD},
    )
    assert confirmed.status_code == 204, confirmed.text


def _invite(client: TestClient, *, email: str) -> str:
    """Invite an address, as an admin would, and return the invitation's id."""
    response = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["invitation_id"])


def test_reset_claims_an_identity_an_admin_added_and_nobody_signed_in_to(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Forgot password is what someone added by an admin tries first, so it mails a link that claims the account."""
    with _client(tmp_path) as client:
        added = client.post(
            f"{API_ROOT}/organizations/me/members",
            json={"email": "ada@example.com", "role": "member"},
            headers={"Otari-Key": MASTER_KEY},
        )
        assert added.status_code == 201, added.text

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        _complete_reset(client, log_text)

        # No separate verification: the reset link already proved the mailbox.
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD})
        assert signed_in.status_code == 200, signed_in.text
        assert client.get(f"{API_ROOT}/organizations/me").status_code == 200


def test_reset_claims_an_invitee_who_then_finds_the_invitation_waiting(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _invite(client, email="ada@example.com")

        status_code, log_text = _request_reset(client, caplog, email="ada@example.com")
        assert status_code == 200
        _complete_reset(client, log_text)

        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD})
        assert signed_in.status_code == 200, signed_in.text
        pending = client.get(f"{API_ROOT}/organizations/me/pending-memberships")
        assert pending.json()["count"] == 1


def test_reset_before_verification_lets_the_identity_sign_in(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A lost verification email is recovered by a reset, whose link proves the same mailbox."""
    with _client(tmp_path) as client:
        client.post(
            f"{API_ROOT}/organizations/me/members",
            json={"email": "ada@example.com", "role": "member"},
            headers={"Otari-Key": MASTER_KEY},
        )
        _with_logs(
            client,
            caplog,
            lambda: client.post(f"{API_ROOT}/auth/signup", json={"email": "ada@example.com", "password": PASSWORD}),
        )

        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        _complete_reset(client, log_text)

        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": NEW_PASSWORD}
            ).status_code
            == 200
        )


def test_reset_sends_nothing_for_a_revoked_invitee_where_signup_is_closed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The rule the signup form and a provider sign-in follow, so a reset is not a third way round it."""
    with _client(tmp_path) as client:
        invitation_id = _invite(client, email="eve@example.com")
        revoked = client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation_id}", headers={"Otari-Key": MASTER_KEY}
        )
        assert revoked.status_code == 200, revoked.text

        status_code, log_text = _request_reset(client, caplog, email="eve@example.com")

        assert status_code == 200
        assert "mail:console" not in log_text
    assert _reset_token_hash(tmp_path, email="eve@example.com") is None


def test_reset_link_stops_working_once_the_invitation_is_revoked_where_signup_is_closed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A link mailed while the invitation was live does not claim the identity after the roster drops it."""
    with _client(tmp_path) as client:
        invitation_id = _invite(client, email="eve@example.com")
        _, log_text = _request_reset(client, caplog, email="eve@example.com")
        revoked = client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation_id}", headers={"Otari-Key": MASTER_KEY}
        )
        assert revoked.status_code == 200, revoked.text

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm",
            json={"token": _extract_token(log_text), "new_password": NEW_PASSWORD},
        )
        assert confirmed.status_code == 400, confirmed.text

        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "eve@example.com", "password": NEW_PASSWORD})
        assert signed_in.status_code != 200


def test_reset_claims_a_revoked_invitee_where_signup_is_open(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Open signup would let the same address sign up, so refusing the reset would protect nothing."""
    with _client(tmp_path, open_signup=True) as client:
        invitation_id = _invite(client, email="eve@example.com")
        client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation_id}", headers={"Otari-Key": MASTER_KEY}
        )

        status_code, log_text = _request_reset(client, caplog, email="eve@example.com")

        assert status_code == 200
        assert "mail:console" in log_text


def test_reset_request_says_the_same_thing_for_a_first_password_and_a_forgotten_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _invite(client, email="ida@example.com")

        messages: set[str] = set()
        for email in ("nobody@example.com", "ada@example.com", "ida@example.com"):
            reply = client.post(f"{API_ROOT}/auth/password/reset", json={"email": email})
            messages.add(reply.json()["message"])
        _, mailed = _request_reset(client, caplog, email="ida@example.com")

        assert len(messages) == 1
        assert "set or reset" in messages.pop()
        assert "Set or reset your Otari password" in mailed


def test_a_reset_link_stops_working_once_the_invitation_behind_it_is_revoked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Closed signup's rule holds when the link is used, not only when it was mailed."""
    with _client(tmp_path) as client:
        invitation_id = _invite(client, email="eve@example.com")
        _, log_text = _request_reset(client, caplog, email="eve@example.com")
        revoked = client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation_id}", headers={"Otari-Key": MASTER_KEY}
        )
        assert revoked.status_code == 200, revoked.text

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm",
            json={"token": _extract_token(log_text), "new_password": NEW_PASSWORD},
        )

        assert confirmed.status_code == 400
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "eve@example.com", "password": NEW_PASSWORD})
        assert signed_in.status_code == 401


TERMS_URL = "https://example.com/terms"


def _terms_accepted_at(tmp_path: Path, *, email: str) -> object:
    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        accepted = connection.execute(
            text('SELECT terms_accepted_at FROM "user" WHERE email = :email'), {"email": email}
        ).scalar_one()
    engine.dispose()
    return accepted


def test_a_reset_that_claims_an_account_requires_the_published_terms(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, terms_url=TERMS_URL) as client:
        _invite(client, email="ida@example.com")
        _, log_text = _request_reset(client, caplog, email="ida@example.com")
        token = _extract_token(log_text)

        refused = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD}
        )
        assert refused.status_code == 400, refused.text
        assert _terms_accepted_at(tmp_path, email="ida@example.com") is None

        # The refusal spent nothing, so the same link completes once the terms are accepted.
        accepted = client.post(
            f"{API_ROOT}/auth/password/reset/confirm",
            json={"token": token, "new_password": NEW_PASSWORD, "terms_accepted": True},
        )
        assert accepted.status_code == 204, accepted.text
    assert _terms_accepted_at(tmp_path, email="ida@example.com") is not None


def test_a_reset_does_not_ask_again_for_terms_already_accepted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, terms_url=TERMS_URL) as client:
        client.post(
            f"{API_ROOT}/organizations/me/members",
            json={"email": "ada@example.com", "role": "member"},
            headers={"Otari-Key": MASTER_KEY},
        )
        _with_logs(
            client,
            caplog,
            lambda: client.post(
                f"{API_ROOT}/auth/signup",
                json={"email": "ada@example.com", "password": PASSWORD, "terms_accepted": True},
            ),
        )
        _, log_text = _request_reset(client, caplog, email="ada@example.com")

        _complete_reset(client, log_text)


def _add_passkey(tmp_path: Path, *, email: str) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with sessionmaker(bind=engine)() as session:
        identity = session.execute(select(User).where(col(User.email) == email)).scalar_one()
        session.add(
            WebAuthnCredential(
                user_id=identity.id,
                name="Laptop",
                credential_id=f"credential-{identity.id}",
                public_key="public-key",
                rp_id="gw.example.com",
            )
        )
        session.commit()
    engine.dispose()


def _passkey_count(tmp_path: Path) -> int:
    engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
    with engine.begin() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM webauthn_credential")).scalar_one()
    engine.dispose()
    return int(count)


def test_a_reset_that_replaces_a_vouched_address_removes_the_passkeys_registered_under_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        engine = create_engine(f"sqlite:///{tmp_path / 'reset-test.db'}")
        with engine.begin() as connection:
            connection.execute(
                text('UPDATE "user" SET email_vouched_at = CURRENT_TIMESTAMP WHERE email = :email'),
                {"email": "ada@example.com"},
            )
        engine.dispose()
        _add_passkey(tmp_path, email="ada@example.com")

        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        _complete_reset(client, log_text)

    assert _passkey_count(tmp_path) == 0


def test_an_ordinary_reset_keeps_the_passkeys(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        _claimed_and_verified(client, caplog, email="ada@example.com")
        _add_passkey(tmp_path, email="ada@example.com")

        _, log_text = _request_reset(client, caplog, email="ada@example.com")
        _complete_reset(client, log_text)

    assert _passkey_count(tmp_path) == 1
