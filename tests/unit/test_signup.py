"""Signup: claiming a roster identity, and registering one where the deployment allows it.

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

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.services.tenancy.provisioning_service import (
    DEFAULT_ORGANIZATION_SLUG,
    DEFAULT_WORKSPACE_NAME,
)

MASTER_KEY = "sk-test-master"
PASSWORD = "a-real-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


def _config(tmp_path: Path, *, mail_ready: bool = True, open_signup: bool = False) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'signup-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        mail_transport="console" if mail_ready else "none",
        public_base_url="https://gw.example.com" if mail_ready else None,
        open_signup=open_signup,
    )


def _client(tmp_path: Path, *, mail_ready: bool = True, open_signup: bool = False) -> TestClient:
    return TestClient(create_app(_config(tmp_path, mail_ready=mail_ready, open_signup=open_signup)))


def _identity_count(tmp_path: Path) -> int:
    """How many identities the deployment holds, operator included."""
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        count = connection.execute(text('SELECT COUNT(*) FROM "user"')).scalar_one()
    engine.dispose()
    return int(count)


def _sign_in_with_master_key(client: TestClient) -> None:
    """Provision the operator identity, so an identity count has a stable baseline."""
    assert client.post(f"{API_ROOT}/auth/session", json={"master_key": MASTER_KEY}).status_code == 200


def _add_member(client: TestClient, *, email: str, role: str = "member") -> None:
    """Put a password-less, unclaimed identity on the roster, as an admin would."""
    response = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": email, "role": role},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text


def _signup(client: TestClient, *, email: str, password: str = PASSWORD, **extra: object) -> Response:
    return client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": password, **extra})


def _deactivate(tmp_path: Path, *, email: str) -> None:
    """Flip an identity inactive, as the M5 backfill does for a soft-deleted gateway user."""
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        connection.execute(text('UPDATE "user" SET is_active = 0 WHERE email = :email'), {"email": email})
    engine.dispose()


def _credentials(tmp_path: Path, *, email: str) -> tuple[str | None, str | None]:
    """The identity's stored password hash and verification token hash."""
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        row = connection.execute(
            text('SELECT hashed_password, email_verification_token_hash FROM "user" WHERE email = :email'),
            {"email": email},
        ).one()
    engine.dispose()
    return (row[0], row[1])


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
        response = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert response.status_code == 403

        verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token})
        assert verified.status_code == 200, verified.text
        assert verified.json()["email"] == "ada@example.com"

        response = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert response.status_code == 200


def test_signup_preserves_the_existing_membership_and_organization(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="grace@example.com", role="admin")
        headers = {"Otari-Key": MASTER_KEY}

        def _roster_row() -> dict[str, object]:
            members = client.get(f"{API_ROOT}/organizations/me/members", headers=headers).json()["data"]
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
            f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": "a-different-password"}
        ).status_code == 401


def test_signup_on_a_deactivated_identity_writes_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A deactivated identity is not claimable, the same way its token is not redeemable.

    ``verify_email`` and ``reset_password`` both refuse a deactivated identity,
    and ``authenticate`` refuses to sign one in. Without the same check here, an
    address deactivated before it ever signed up could still have a password set
    and a live verification token minted on it, both waiting to become usable
    the moment an operator reactivated the identity.
    """
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        _deactivate(tmp_path, email="ada@example.com")

        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            response = _signup(client, email="ada@example.com")
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert response.status_code == 200
        assert response.json() == _signup(client, email="nobody@example.com").json()
        assert "mail:console" not in caplog.text

    assert _credentials(tmp_path, email="ada@example.com") == (None, None)


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
        # to name what is missing, the same as GET /api/v1/settings/mail's own.
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


# =============================================================================
# Open signup: registering an address nobody put on the roster
# =============================================================================


def test_closed_signup_creates_no_identity_for_an_unknown_address(tmp_path: Path) -> None:
    """The default posture, asserted on the table rather than on the response.

    ``test_signup_on_an_untouched_address_is_enumeration_safe`` already asserts
    the response says nothing. This asserts the other half, which is the one a
    self-hoster is relying on: nobody who can reach the dashboard can join the
    deployment's single tenant by submitting the form.
    """
    with _client(tmp_path) as client:
        _sign_in_with_master_key(client)
        before = _identity_count(tmp_path)

        assert _signup(client, email="stranger@example.com").status_code == 200

        assert _identity_count(tmp_path) == before


def test_open_signup_registers_an_unknown_address_and_verification_lets_it_sign_in(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        token = _captured_verification_link(
            caplog, client, email="ada@example.com", full_name="Ada Lovelace"
        )

        # Unverified, so the hard-block refuses the password that was just set,
        # exactly as it does on the claim path.
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
            ).status_code
            == 403
        )
        assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token}).status_code == 200
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
            ).status_code
            == 200
        )


def test_open_signup_lands_the_new_account_in_an_organization_it_owns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A tenant of its own, with the workspace every organization needs to hold anything.

    Not the deployment's default organization: a registration that joined it
    would put a stranger inside the operator's tenant, which is the thing the
    closed default exists to prevent.
    """
    with _client(tmp_path, open_signup=True) as client:
        token = _captured_verification_link(
            caplog, client, email="ada@example.com", full_name="Ada Lovelace"
        )
        assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token}).status_code == 200
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
            ).status_code
            == 200
        )

        context = client.get(f"{API_ROOT}/organizations/me").json()
        assert context["role"] == "owner"
        assert context["organization"]["name"] == "Ada Lovelace's organization"
        assert context["organization"]["slug"] != DEFAULT_ORGANIZATION_SLUG

        memberships = client.get(f"{API_ROOT}/organizations/me/memberships").json()["data"]
        assert [row["role"] for row in memberships] == ["owner"]

        workspaces = client.get(f"{API_ROOT}/workspaces").json()["data"]
        assert [workspace["name"] for workspace in workspaces] == [DEFAULT_WORKSPACE_NAME]


def test_open_signup_names_the_organization_from_the_address_when_no_name_is_given(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        token = _captured_verification_link(caplog, client, email="ada@example.com")
        assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token}).status_code == 200
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
            ).status_code
            == 200
        )

        context = client.get(f"{API_ROOT}/organizations/me").json()
        assert context["organization"]["name"] == "ada's organization"


def test_open_signup_fits_a_long_name_into_the_organization_name_column(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 255-character name plus the suffix would overflow a 255-character column.

    ``SignupRequest.full_name`` admits exactly what ``Organization.name`` holds,
    so the wrapper has to come out of the name rather than off the end of the
    row: untruncated, a perfectly valid signup fails its flush.
    """
    with _client(tmp_path, open_signup=True) as client:
        token = _captured_verification_link(
            caplog, client, email="ada@example.com", full_name="A" * 255
        )
        assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token}).status_code == 200
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}
            ).status_code
            == 200
        )

        name = client.get(f"{API_ROOT}/organizations/me").json()["organization"]["name"]
        assert len(name) == 255
        assert name.endswith("'s organization")


def test_open_signup_claims_a_roster_address_rather_than_registering_a_second_tenant(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The setting widens what an *unknown* address does and changes nothing else.

    An address an admin added is still claimed in place, keeping the membership
    and the organization they were added to. Registering a fresh tenant for them
    instead would quietly undo the admin's act.
    """
    with _client(tmp_path, open_signup=True) as client:
        _add_member(client, email="grace@example.com", role="admin")
        headers = {"Otari-Key": MASTER_KEY}
        before = client.get(f"{API_ROOT}/organizations/me/members", headers=headers).json()["data"]

        _captured_verification_link(caplog, client, email="grace@example.com")

        after = client.get(f"{API_ROOT}/organizations/me/members", headers=headers).json()["data"]
        assert [row["organization_member_id"] for row in after] == [
            row["organization_member_id"] for row in before
        ]
        assert next(row for row in after if row["email"] == "grace@example.com")["role"] == "admin"


def test_open_signup_on_an_already_registered_address_is_enumeration_safe(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Registration does not become an oracle for which addresses are taken."""
    with _client(tmp_path, open_signup=True) as client:
        _captured_verification_link(caplog, client, email="ada@example.com")
        before = _identity_count(tmp_path)

        caplog.clear()
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            again = _signup(client, email="ada@example.com", password="a-different-password")
            unknown = _signup(client, email="nobody-else@example.com", password=PASSWORD)
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert again.status_code == 200
        assert again.json() == unknown.json()
        # The second submission wrote nothing: no mail, no second identity for
        # the taken address, and the original password still signs in.
        assert _identity_count(tmp_path) == before + 1  # the unknown address registered
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": "a-different-password"}
            ).status_code
            == 401
        )


def test_open_signup_without_mail_configured_registers_nobody(tmp_path: Path) -> None:
    """The refusal that already guards the claim path guards registration too.

    A registered account that can never be verified is a row nobody can sign in
    as, so the 503 has to come before the identity is written.
    """
    with _client(tmp_path, mail_ready=False, open_signup=True) as client:
        _sign_in_with_master_key(client)
        before = _identity_count(tmp_path)

        response = _signup(client, email="ada@example.com")

        assert response.status_code == 503
        assert _identity_count(tmp_path) == before
