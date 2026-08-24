"""Password sign-in, and the master key retiring as the dashboard login.

The shape mozilla-ai/otari-ai#1716 settled and issue #649 implements: the master
key bootstraps a standalone deployment, an operator claims the deployment by
setting an address and a password, and from that moment email and password is
the dashboard login while the master key stays the credential for the management
API. Both halves of that sentence are load-bearing and both are asserted here,
because retiring the login and retiring the credential are one edit apart and
the second would break every self-hoster's automation and the OSS smoke gate.

Unit rather than integration: everything under test is route, service and
identity behavior that runs unchanged on the SQLite file each test stands up.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.models.entities import DashboardSession
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.services.password_service import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH, hash_password
from gateway.services.tenancy.provisioning_service import BOOTSTRAP_IDENTITY_KEY
from gateway.services.tenancy.user_service import _is_email_conflict

MASTER_KEY = "sk-test-master"
EMAIL = "operator@example.com"
PASSWORD = "first-password"  # pragma: allowlist secret
NEW_PASSWORD = "second-password"  # pragma: allowlist secret


def _config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'password-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
    )


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_config(tmp_path)))


def _sign_in_with_master_key(client: TestClient) -> dict[str, object]:
    response = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _claim(client: TestClient, *, email: str = EMAIL, password: str = PASSWORD) -> None:
    """Claim the deployment with the master key in a header, as an operator would."""
    response = client.put(
        "/v1/auth/password",
        json={"email": email, "new_password": password},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 200, response.text


def _sessions(tmp_path: Path) -> list[DashboardSession]:
    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with sessionmaker(bind=engine)() as session:
        return list(session.execute(select(DashboardSession)).scalars().all())


# =============================================================================
# First boot: unchanged
# =============================================================================


def test_first_boot_still_signs_in_with_the_master_key(tmp_path: Path) -> None:
    """The bootstrap path #647 built, which this change must not disturb."""
    with _client(tmp_path) as client:
        body = _sign_in_with_master_key(client)

        assert SESSION_COOKIE_NAME in client.cookies
        assert body["user_id"] and body["active_organization_id"]
        assert client.get("/v1/settings").status_code == 200


def test_an_unclaimed_deployment_advertises_the_master_key(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["master_key"]


def test_a_password_sign_in_before_anyone_has_one_is_refused(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD})

        assert response.status_code == 401
        assert SESSION_COOKIE_NAME not in client.cookies


# =============================================================================
# Claiming the deployment
# =============================================================================


def test_claiming_sets_the_address_and_retires_master_key_sign_in(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"email": EMAIL, "new_password": PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 200, response.text
        assert response.json() == {"email": EMAIL, "master_key_sign_in_retired": True}
        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["password"]


def test_the_claimed_address_is_normalized(tmp_path: Path) -> None:
    """The unique index is case-sensitive, so the stored form has to be settled here."""
    with _client(tmp_path) as client:
        _claim(client, email="  Operator@Example.COM  ")

        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200


def test_claiming_without_an_address_is_refused(tmp_path: Path) -> None:
    """The operator identity first boot provisions has none, so one must be supplied."""
    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"new_password": PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 400
        assert "email address" in response.json()["detail"]


def test_an_operator_signed_in_on_the_cookie_can_claim(tmp_path: Path) -> None:
    """The dashboard's own path: the session was master-key minted, so it is proof enough."""
    with _client(tmp_path) as client:
        _sign_in_with_master_key(client)

        response = client.put("/v1/auth/password", json={"email": EMAIL, "new_password": PASSWORD})

        assert response.status_code == 200, response.text
        # The claiming session survives; the operator is not signed out mid-claim.
        assert client.get("/v1/settings").status_code == 200


def test_claiming_needs_a_credential(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.put("/v1/auth/password", json={"email": EMAIL, "new_password": PASSWORD})

        assert response.status_code == 401


def test_changing_an_address_that_already_exists_is_refused(tmp_path: Path) -> None:
    """Address changes belong to the verification flow (#650), not to this endpoint."""
    with _client(tmp_path) as client:
        _claim(client)

        response = client.put(
            "/v1/auth/password",
            json={"email": "someone.else@example.com", "new_password": NEW_PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]


def test_claiming_an_address_another_identity_holds_is_refused(tmp_path: Path) -> None:
    """The column is unique and the address is what sign-in matches on."""
    with _client(tmp_path) as client:
        headers = {"Otari-Key": MASTER_KEY}
        added = client.post("/v1/organizations/me/members", json={"email": EMAIL}, headers=headers)
        assert added.status_code == 201, added.text

        response = client.put(
            "/v1/auth/password",
            json={"email": EMAIL, "new_password": PASSWORD},
            headers=headers,
        )

        assert response.status_code == 409, response.text
        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["master_key"]


def test_the_email_conflict_detector_reads_sqlite_and_ignores_other_constraints(tmp_path: Path) -> None:
    """The unique-index race is mapped only when it really is the email index.

    Provoked against a real SQLite engine rather than a synthesized exception,
    because the whole point of the detector is that the two engines word this
    differently and only the engine knows how. PostgreSQL's wording is covered
    by ``tests/integration/test_tenancy_races.py``; SQLite's is covered here,
    since that is what the unit suite runs on.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'conflict.db'}")
    with engine.begin() as connection:
        # Named "user" on purpose: SQLite reports the *table* and column, so a
        # stand-in name would prove the detector matches a string this codebase
        # never produces.
        connection.execute(text('CREATE TABLE "user" (id INTEGER PRIMARY KEY, email TEXT)'))
        connection.execute(text('CREATE UNIQUE INDEX ix_user_email ON "user" (email)'))
        connection.execute(text("INSERT INTO \"user\" (id, email) VALUES (1, 'taken@example.com')"))

    with pytest.raises(IntegrityError) as duplicate_email:
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO \"user\" (id, email) VALUES (2, 'taken@example.com')"))
    assert _is_email_conflict(duplicate_email.value)

    with pytest.raises(IntegrityError) as duplicate_id:
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO \"user\" (id, email) VALUES (1, 'other@example.com')"))
    # A different constraint must keep its own error rather than being reported
    # to the caller as "that address is taken".
    assert not _is_email_conflict(duplicate_id.value)


def test_resubmitting_the_same_address_with_a_new_password_is_not_a_change(tmp_path: Path) -> None:
    """A client keeping one form for claiming and changing must not 400 on the second use."""
    with _client(tmp_path) as client:
        _claim(client)

        response = client.put(
            "/v1/auth/password",
            json={"email": EMAIL.upper(), "new_password": NEW_PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 200, response.text
        assert response.json()["email"] == EMAIL
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": NEW_PASSWORD}).status_code == 200


def test_claiming_an_identity_that_already_has_an_address_stamps_it_verified(tmp_path: Path) -> None:
    """The adopted-tenancy case: no email is supplied, so nothing else would stamp it.

    #650 turns on the verification gate, and its premise is that every identity
    able to sign in by then already passes. An operator identity adopted from an
    existing tenancy arrives *with* an address, so its claim supplies no email;
    without this the column stays NULL and that operator is locked out the day
    the gate lands.
    """
    with _client(tmp_path) as client:
        _sign_in_with_master_key(client)

    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        # Stand in for the adopted identity: an address, and nothing else.
        connection.execute(
            text('UPDATE "user" SET email = :email, email_verified_at = NULL'),
            {"email": EMAIL},
        )

    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"new_password": PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )
        assert response.status_code == 200, response.text

    with engine.begin() as connection:
        verified = connection.execute(text('SELECT email_verified_at FROM "user"')).scalar_one()
    assert verified is not None


def test_an_ordinary_password_change_does_not_stamp_the_address_verified(tmp_path: Path) -> None:
    """Proving the current password says the caller owns the account, not the address.

    The session is established *before* ``email_verified_at`` is nulled out, and
    on purpose: #650's sign-in gate would otherwise refuse the very sign-in this
    test uses to reach the change. A cookie already resolved does not re-check
    verification on every request (``resolve_dashboard_session`` checks only
    ``is_active``), so the established session still reaches the password
    change, which is what is actually under test here.
    """
    with _client(tmp_path) as client:
        _claim(client)

        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200

        engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
        with engine.begin() as connection:
            connection.execute(text('UPDATE "user" SET email_verified_at = NULL'))

        assert (
            client.put(
                "/v1/auth/password",
                json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
            ).status_code
            == 200
        )

    with engine.begin() as connection:
        verified = connection.execute(text('SELECT email_verified_at FROM "user"')).scalar_one()
    assert verified is None


def test_a_second_identity_without_a_password_cannot_be_given_one_by_anyone(tmp_path: Path) -> None:
    """Pins the assumption the claim-from-cookie branch rests on.

    ``set_password`` skips the current-password proof partly because a session on
    an unclaimed deployment can only have been minted by the master key. Once
    #650 mints sessions another way that stops being self-evident, so the other
    half of the guarantee is pinned here: the endpoint only ever acts on the
    caller's own identity, so a member added by address is not reachable through
    it at all, whoever is calling.
    """
    with _client(tmp_path) as client:
        headers = {"Otari-Key": MASTER_KEY}
        added = client.post("/v1/organizations/me/members", json={"email": "member@example.com"}, headers=headers)
        assert added.status_code == 201, added.text
        _claim(client)

        # The master key acts as the bootstrap operator, never as the member, so
        # this sets the operator's own password and leaves the member's NULL.
        assert (
            client.put(
                "/v1/auth/password",
                json={"new_password": NEW_PASSWORD},
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/auth/session", json={"email": "member@example.com", "password": NEW_PASSWORD}
            ).status_code
            == 401
        )


@pytest.mark.parametrize(
    ("password", "expected_status"),
    [
        # The floor is the request schema's, so it never reaches a hash.
        ("a" * (MIN_PASSWORD_LENGTH - 1), 422),
        # The ceiling is bcrypt's and is counted in bytes, so it is the service's.
        ("a" * (MAX_PASSWORD_BYTES + 1), 400),
        ("é" * (MAX_PASSWORD_BYTES // 2 + 1), 400),
    ],
    ids=["too-short", "too-long", "too-long-in-bytes"],
)
def test_a_password_outside_the_published_bounds_is_refused(
    tmp_path: Path, password: str, expected_status: int
) -> None:
    """A multi-byte password hits the ceiling sooner than its character count reads."""
    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"email": EMAIL, "new_password": password},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == expected_status, response.text
        # Nothing was stored, so the deployment is still on its bootstrap credential.
        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["master_key"]


def test_an_address_longer_than_the_column_is_refused_by_the_schema(tmp_path: Path) -> None:
    """255 is the column's width, so a longer address has to be a 422, not a driver error.

    SQLite does not enforce the width, so what this pins is the request
    schema's own ceiling: without it PostgreSQL answers 500 from
    ``StringDataRightTruncationError`` while this test still passed.
    """
    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"email": "a" * 250 + "@example.com", "new_password": PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 422, response.text
        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["master_key"]


# =============================================================================
# The steady state: password in, master key out (of the login only)
# =============================================================================


def test_a_claimed_deployment_signs_in_with_email_and_password(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _claim(client)

        response = client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD})

        assert response.status_code == 200, response.text
        assert SESSION_COOKIE_NAME in client.cookies
        assert response.json()["user_id"]
        assert response.json()["active_organization_id"]
        # The cookie alone opens the management API, exactly as a master-key session did.
        assert client.get("/v1/settings").status_code == 200


def test_master_key_sign_in_is_refused_once_the_deployment_is_claimed(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _claim(client)

        response = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})

        assert response.status_code == 403
        assert "email and password" in response.json()["detail"]
        assert SESSION_COOKIE_NAME not in client.cookies


def test_a_wrong_master_key_is_still_a_401_on_a_claimed_deployment(tmp_path: Path) -> None:
    """The 403 says "this use is over"; an unknown key must not borrow that answer."""
    with _client(tmp_path) as client:
        _claim(client)

        assert client.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 401


def test_the_master_key_still_authenticates_the_management_api(tmp_path: Path) -> None:
    """The half of #1716 that is not a login. Breaking this breaks every self-hoster."""
    with _client(tmp_path) as client:
        _claim(client)
        headers = {"Otari-Key": MASTER_KEY}

        assert client.get("/v1/keys", headers=headers).status_code == 200
        assert client.get("/v1/users", headers=headers).status_code == 200
        assert client.get("/v1/budgets", headers=headers).status_code == 200
        assert client.get("/v1/settings", headers=headers).status_code == 200
        created = client.post("/v1/keys", json={"key_name": "after-claim"}, headers=headers)
        assert created.status_code in (200, 201), created.text


@pytest.mark.parametrize(
    "credentials",
    [
        {"email": EMAIL, "password": "wrong"},
        {"email": "nobody@example.com", "password": PASSWORD},
        {"email": EMAIL.upper(), "password": "wrong"},
    ],
    ids=["wrong-password", "unknown-address", "known-address-wrong-password-other-case"],
)
def test_every_failed_password_sign_in_answers_the_same(tmp_path: Path, credentials: dict[str, str]) -> None:
    """An unauthenticated caller must not be able to tell which addresses exist."""
    with _client(tmp_path) as client:
        _claim(client)

        response = client.post("/v1/auth/session", json=credentials)

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect email or password"
        assert SESSION_COOKIE_NAME not in client.cookies


def test_signing_in_is_case_insensitive_in_the_address(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _claim(client)

        assert client.post("/v1/auth/session", json={"email": EMAIL.upper(), "password": PASSWORD}).status_code == 200


def test_a_deactivated_identity_cannot_sign_in(tmp_path: Path) -> None:
    """Same rule the session cookie already follows: deactivation ends access now."""
    with _client(tmp_path) as client:
        _claim(client)

    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        connection.execute(text('UPDATE "user" SET is_active = 0'))

    with _client(tmp_path) as client:
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 401


def test_an_unverified_identity_with_the_right_password_is_refused(tmp_path: Path) -> None:
    """#650's hard-block: the right password is not enough until the address is verified.

    Distinct from every other sign-in failure: the password has already been
    proven correct by the time this is checked, so the caller is told what is
    actually wrong rather than folded into ``InvalidCredentialsError``.
    """
    with _client(tmp_path) as client:
        _claim(client)

    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        connection.execute(text('UPDATE "user" SET email_verified_at = NULL'))

    with _client(tmp_path) as client:
        response = client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD})
        assert response.status_code == 403, response.text
        assert "verify your email" in response.json()["detail"].lower()
        assert SESSION_COOKIE_NAME not in client.cookies


def test_a_verified_identity_signs_in_normally(tmp_path: Path) -> None:
    """Claiming the deployment stamps ``email_verified_at``, so sign-in is unaffected by #650's gate."""
    with _client(tmp_path) as client:
        _claim(client)
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"master_key": MASTER_KEY, "email": EMAIL, "password": PASSWORD},
        {"email": EMAIL},
        {"password": PASSWORD},
    ],
    ids=["neither", "both", "address-only", "password-only"],
)
def test_a_sign_in_body_must_carry_exactly_one_credential(tmp_path: Path, body: dict[str, str]) -> None:
    with _client(tmp_path) as client:
        assert client.post("/v1/auth/session", json=body).status_code == 422


# =============================================================================
# Changing a password
# =============================================================================


def test_a_signed_in_operator_changes_their_password_with_the_current_one(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _claim(client)
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200

        response = client.put(
            "/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert response.status_code == 200, response.text
        # Still signed in on the session the change was made from.
        assert client.get("/v1/settings").status_code == 200

    with _client(tmp_path) as fresh:
        assert fresh.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 401
        assert fresh.post("/v1/auth/session", json={"email": EMAIL, "password": NEW_PASSWORD}).status_code == 200


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [
        ({"new_password": NEW_PASSWORD}, "current password is required"),
        ({"current_password": "wrong", "new_password": NEW_PASSWORD}, "Current password is incorrect"),
        ({"current_password": PASSWORD, "new_password": PASSWORD}, "cannot be the same"),
    ],
    ids=["no-current", "wrong-current", "unchanged"],
)
def test_a_password_change_from_a_session_is_refused_without_the_right_proof(
    tmp_path: Path, body: dict[str, str], expected_detail: str
) -> None:
    with _client(tmp_path) as client:
        _claim(client)
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200

        response = client.put("/v1/auth/password", json=body)

        assert response.status_code == 400, response.text
        assert expected_detail in response.json()["detail"]
        # The stored password is untouched.
        assert client.get("/v1/settings").status_code == 200


def test_the_master_key_resets_a_forgotten_password_without_the_old_one(tmp_path: Path) -> None:
    """Why this change cannot strand an operator: the deployment credential still works."""
    with _client(tmp_path) as client:
        _claim(client)

        response = client.put(
            "/v1/auth/password",
            json={"new_password": NEW_PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 200, response.text
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": NEW_PASSWORD}).status_code == 200


def test_a_password_change_revokes_the_identity_s_other_sessions(tmp_path: Path) -> None:
    """A cookie minted under the old password must not outlive it."""
    with _client(tmp_path) as elsewhere, _client(tmp_path) as here:
        _claim(here)
        assert elsewhere.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200
        assert here.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200
        assert len(_sessions(tmp_path)) == 2

        assert (
            here.put(
                "/v1/auth/password",
                json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
            ).status_code
            == 200
        )

        assert here.get("/v1/settings").status_code == 200
        assert elsewhere.get("/v1/settings").status_code == 401
        assert len(_sessions(tmp_path)) == 1


def test_a_master_key_reset_revokes_every_session_including_the_browser_s(tmp_path: Path) -> None:
    """A header caller has no session of its own to spare, and a reset is a recovery."""
    with _client(tmp_path) as browser:
        _claim(browser)
        assert browser.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD}).status_code == 200

        assert (
            browser.put(
                "/v1/auth/password",
                json={"new_password": NEW_PASSWORD},
                headers={"Otari-Key": MASTER_KEY},
            ).status_code
            == 200
        )

        assert browser.get("/v1/settings").status_code == 401
        assert _sessions(tmp_path) == []


def test_a_session_that_outlived_its_expiry_does_not_authenticate_a_change(tmp_path: Path) -> None:
    """The claim path is gated on a *valid* session, not on the presence of a cookie."""
    with _client(tmp_path) as client:
        _sign_in_with_master_key(client)

    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE dashboard_sessions SET expires_at = :past"),
            {"past": (datetime.now(UTC) - timedelta(hours=1)).isoformat(sep=" ")},
        )

    with _client(tmp_path) as client:
        response = client.put("/v1/auth/password", json={"email": EMAIL, "new_password": PASSWORD})

        assert response.status_code == 401


def test_a_refused_sign_in_body_is_not_echoed_back(tmp_path: Path) -> None:
    """A 422 must not carry the credential that caused it.

    Pydantic puts the rejected value on every error entry and FastAPI's default
    handler serializes it, so an over-length password came back in full and the
    dashboard rendered the whole ``detail`` into its error banner. It is also
    what makes the reply as large as the request on an endpoint anyone can post
    to. `gateway.main` strips ``input`` and ``ctx``; this is what says so.
    """
    with _client(tmp_path) as client:
        oversized = "p" * (MAX_PASSWORD_BYTES + 1)

        too_long = client.post("/v1/auth/session", json={"email": EMAIL, "password": oversized})
        both = client.post(
            "/v1/auth/session",
            json={"master_key": MASTER_KEY, "email": EMAIL, "password": PASSWORD},
        )

        assert too_long.status_code == 422, too_long.text
        assert both.status_code == 422, both.text
        for response in (too_long, both):
            assert oversized not in response.text
            assert MASTER_KEY not in response.text
            assert PASSWORD not in response.text
            # The reason still reaches the caller; only the value is dropped.
            assert response.json()["detail"][0]["msg"]
            assert set(response.json()["detail"][0]) == {"type", "loc", "msg"}


def test_an_address_stored_with_other_casing_is_still_the_identity_s_own(tmp_path: Path) -> None:
    """Resubmitting the held address must not depend on how the row was written.

    ``user.email`` is only lower-cased when this tree wrote it. A row from the
    M4 re-parenting backfill, or from an operator's own SQL, can carry any
    casing, which ``UserRepository.get_by_email`` already matches
    case-insensitively. Comparing a normalized candidate against the raw column
    refused the identity its own address with "changing it is not supported",
    which is the one thing the field documents as accepted and ignored.
    """
    with _client(tmp_path) as client:
        _claim(client)

    # Rewritten by address rather than with a bare ``UPDATE "user"``: a fixture
    # that grows a second identity would otherwise have its address clobbered
    # too, and the unique index would fail the statement rather than the
    # assertion.
    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "user" SET email = :stored WHERE email = :claimed'),
            {"stored": "Operator@Example.COM", "claimed": EMAIL},
        )
    engine.dispose()

    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"email": EMAIL, "new_password": NEW_PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 200, response.text
        assert client.post("/v1/auth/session", json={"email": EMAIL, "password": NEW_PASSWORD}).status_code == 200


def test_a_malformed_stored_address_is_not_reported_as_the_caller_s_mistake(tmp_path: Path) -> None:
    """The stored side is normalized, not shape-checked.

    Running ``validated_email`` over a column value raises ``InvalidEmailError``
    for a stored address that cannot pass the shape check, so a caller who
    submitted a perfectly good address would be told theirs is invalid, with
    the stored one quoted back at them. Both refusals are 400, so the message
    is what separates them. ``get_by_email`` normalizes with ``.strip().lower()``
    and this comparison matches it.
    """
    with _client(tmp_path) as client:
        _claim(client)

    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "user" SET email = :stored WHERE email = :claimed'),
            {"stored": "legacy operator", "claimed": EMAIL},
        )
    engine.dispose()

    with _client(tmp_path) as client:
        response = client.put(
            "/v1/auth/password",
            json={"email": EMAIL, "new_password": NEW_PASSWORD},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert "changing it is not supported" in detail, detail
        assert "legacy operator" not in detail, detail


# =============================================================================
# Whose password retires the master key (#702)
# =============================================================================
#
# "The deployment has been claimed" is a fact about the *operator* identity, and
# the switch is read off that identity alone. It used to be read off any row
# holding a ``hashed_password``, which was the same answer only while claiming
# was the sole writer of that column. #650 and the re-parenting backfill
# (otari-ai#1644) are each another writer, and each pulls the two apart in a
# different direction; both are pinned here.


def _add_a_password(tmp_path: Path, email: str, password: str) -> None:
    """Give an existing identity a usable password, the way signup does.

    Written straight to the row rather than through ``POST /v1/auth/signup``,
    which needs a configured mailer: what these tests are about is the column,
    not the flow that fills it. ``email_verified_at`` is stamped alongside so
    the identity can actually sign in, which is the state signup leaves it in
    once the verification link is opened.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
    with engine.begin() as connection:
        updated = connection.execute(
            text(
                'UPDATE "user" SET hashed_password = :hashed, email_verified_at = :verified '
                "WHERE lower(email) = :email"
            ),
            {"hashed": hash_password(password), "verified": datetime.now(UTC), "email": email},
        )
    assert updated.rowcount == 1


def test_a_member_s_password_leaves_an_unclaimed_deployment_on_the_master_key(tmp_path: Path) -> None:
    """The #650 direction: somebody else's password is not the operator's claim.

    Without this, the first member to complete signup retires the dashboard
    login of an operator who never set a password, by an act of their own and
    with no way back through the UI.
    """
    with _client(tmp_path) as client:
        headers = {"Otari-Key": MASTER_KEY}
        _sign_in_with_master_key(client)  # provisions the operator identity
        added = client.post("/v1/organizations/me/members", json={"email": "member@example.com"}, headers=headers)
        assert added.status_code == 201, added.text

        _add_a_password(tmp_path, "member@example.com", PASSWORD)

        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["master_key"]
        client.cookies.clear()
        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 200


def test_a_member_changing_their_password_is_told_the_master_key_still_signs_in(tmp_path: Path) -> None:
    """``master_key_sign_in_retired`` is read back, not asserted from the act.

    The dashboard turns the master-key affordances off on this field alone
    (`web/src/shared/hooks/useDeployment.tsx`), so a member changing their own
    password on an unclaimed deployment must not be the thing that reports it.
    """
    with _client(tmp_path) as client:
        headers = {"Otari-Key": MASTER_KEY}
        _sign_in_with_master_key(client)
        added = client.post("/v1/organizations/me/members", json={"email": "member@example.com"}, headers=headers)
        assert added.status_code == 201, added.text
        _add_a_password(tmp_path, "member@example.com", PASSWORD)

        client.cookies.clear()
        signed_in = client.post("/v1/auth/session", json={"email": "member@example.com", "password": PASSWORD})
        assert signed_in.status_code == 200, signed_in.text

        changed = client.put(
            "/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert changed.status_code == 200, changed.text
        assert changed.json() == {"email": "member@example.com", "master_key_sign_in_retired": False}


def test_an_identity_that_arrived_with_a_password_does_not_claim_the_deployment(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The otari-ai#1644 direction: a re-parented database is not a claimed one.

    Platform identities carry ``hashed_password`` already, so a backfilled
    deployment would otherwise read as claimed before anyone had claimed it:
    ``/v1/bootstrap`` would publish ``["password"]`` and the sign-in screen
    would ask the operator holding the master key for credentials that are not
    theirs.
    """
    with _client(tmp_path) as client:
        # The shape a backfill leaves behind: rows this gateway never
        # provisioned, so no ``tenancy_bootstrap_user_id`` marker names any of
        # them, and the identity carries a password from the platform.
        engine = create_engine(f"sqlite:///{tmp_path / 'password-test.db'}")
        organization_id = uuid.uuid4()
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO organization (id, name, slug, created_at, updated_at) "
                    "VALUES (:id, :name, :slug, :now, :now)"
                ),
                {"id": organization_id.hex, "name": "Acme", "slug": "acme-a1b2", "now": datetime.now(UTC)},
            )
            connection.execute(
                text(
                    'INSERT INTO "user" (id, email, hashed_password, is_active, is_superuser, '
                    "active_organization_id, created_at, updated_at) "
                    "VALUES (:id, :email, :hashed, 1, 0, :organization_id, :now, :now)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "email": "imported@example.com",
                    "hashed": hash_password(PASSWORD),
                    "organization_id": organization_id.hex,
                    "now": datetime.now(UTC),
                },
            )

        assert client.get("/v1/bootstrap").json()["sign_in_methods"] == ["master_key"]

        # And the master key is not refused as a retired login. It cannot mint a
        # session on this database either, but for the reason that actually
        # applies: ``ForeignTenancyError``, which names the marker to point.
        # A 403 here would be the bug, since it would send an operator holding
        # the only credential there is off to find a password.
        #
        # Pinned through the log and not the body: ``_tenancy_error_handler``
        # renders every 5xx tenancy error as a bare "Internal server error", so
        # the status alone would pass for a broken fixture or any unrelated 500.
        # The ``gateway`` logger does not propagate, hence the handler.
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.ERROR, logger="gateway")
        try:
            refused = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert refused.status_code == 500
        assert refused.json() == {"detail": "Internal server error"}
        failure = "\n".join(record.getMessage() for record in caplog.records)
        assert "Tenancy request failed" in failure, failure
        assert BOOTSTRAP_IDENTITY_KEY in failure, failure
        assert "acme-a1b2" in failure, failure
