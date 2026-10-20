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
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import create_engine, text

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.exceptions.identity_exceptions import ResetTokenInvalidError
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.models.tenancy import User
from gateway.services.password_service import hash_password_async
from gateway.services.tenancy.organization_service import OrganizationService, _default_organization_name
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


def _mark_verified_by_a_provider(tmp_path: Path, *, email: str) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE \"user\" SET email_verified_at = CURRENT_TIMESTAMP, oauth_provider = 'google' "
                "WHERE email = :email"
            ),
            {"email": email},
        )
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

        token = _captured_verification_link(caplog, client, email="ada@example.com", full_name="Ada Lovelace")

        # Unverified: the hard-block refuses the very password just set.
        response = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert response.status_code == 403

        verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
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


def test_signup_on_an_untouched_address_is_enumeration_safe(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
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
        assert (
            client.post(
                f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": "a-different-password"}
            ).status_code
            == 401
        )


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


def test_signup_cannot_set_a_password_on_a_verified_identity(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        _mark_verified_by_a_provider(tmp_path, email="ada@example.com")

        assert _signup(client, email="ada@example.com").status_code == 200
        response = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})

    assert response.status_code == 401


def test_signup_on_a_verified_identity_writes_nothing_and_tells_the_owner_how_it_signs_in(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The response is the one any address gets; the mailbox learns it already has an account."""
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        _mark_verified_by_a_provider(tmp_path, email="ada@example.com")

        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            response = _signup(client, email="ada@example.com")
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert response.status_code == 200
        assert response.json() == _signup(client, email="nobody@example.com").json()
        assert "You already have an Otari account" in caplog.text
        assert "signs in with Google" in caplog.text
        assert "token=" not in caplog.text

    assert _credentials(tmp_path, email="ada@example.com") == (None, None)


def test_signup_on_a_verified_password_account_mails_its_owner_a_notice(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        token = _captured_verification_link(caplog, client, email="ada@example.com")
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )

        caplog.clear()
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            again = _signup(client, email="ada@example.com", password="a-different-password")
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert again.json() == _signup(client, email="nobody@example.com").json()
        assert "You already have an Otari account" in caplog.text
        assert "Forgot your password?" in caplog.text
        assert "token=" not in caplog.text


def test_signup_writes_nothing_when_the_address_is_verified_after_its_check(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def verify_then_hash(password: str) -> str:
        _mark_verified_by_a_provider(tmp_path, email="ada@example.com")
        return await hash_password_async(password)

    with _client(tmp_path) as client:
        _add_member(client, email="ada@example.com")
        # Signup hashes the password between its check and its write.
        monkeypatch.setattr("gateway.services.tenancy.user_service.hash_password_async", verify_then_hash)

        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            response = _signup(client, email="ada@example.com")
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert response.status_code == 200
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


def _invite(client: TestClient, *, email: str) -> dict[str, object]:
    response = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


def _lapse_invitations(tmp_path: Path) -> None:
    """Put every invitation's deadline in the past without anyone presenting its token, as an unopened link."""
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        connection.execute(text("UPDATE invitation SET expires_at = '2000-01-01 00:00:00'"))
    engine.dispose()


def test_closed_signup_claims_an_identity_with_a_pending_invitation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _invite(client, email="ada@example.com")

        token = _captured_verification_link(caplog, client, email="ada@example.com")

        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )


@pytest.mark.parametrize("ending", ["revoked", "lapsed", "removed"])
def test_closed_signup_refuses_an_identity_the_roster_no_longer_holds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ending: str
) -> None:
    """Inviting or adding an address creates its identity, and ending that leaves the identity behind.

    Claiming it would let somebody an admin revoked or removed sign in to a
    deployment that keeps signup closed, and from there create an organization.
    """
    headers = {"Otari-Key": MASTER_KEY}
    with _client(tmp_path) as client:
        if ending == "removed":
            added = client.post(
                f"{API_ROOT}/organizations/me/members",
                json={"email": "eve@example.com", "role": "member"},
                headers=headers,
            )
            assert added.status_code == 201, added.text
            removed = client.delete(
                f"{API_ROOT}/organizations/me/members/{added.json()['organization_member_id']}", headers=headers
            )
            assert removed.status_code == 200, removed.text
        else:
            invitation = _invite(client, email="eve@example.com")
            if ending == "revoked":
                revoked = client.delete(
                    f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}", headers=headers
                )
                assert revoked.status_code == 200, revoked.text
            else:
                _lapse_invitations(tmp_path)

        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            response = _signup(client, email="eve@example.com")
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert response.status_code == 200
        assert response.json() == _signup(client, email="nobody@example.com").json()
        assert "mail:console" not in caplog.text
        assert _credentials(tmp_path, email="eve@example.com") == (None, None)


def test_open_signup_registers_an_unknown_address_and_verification_lets_it_sign_in(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        token = _captured_verification_link(caplog, client, email="ada@example.com", full_name="Ada Lovelace")

        # Unverified, so the hard-block refuses the password that was just set,
        # exactly as it does on the claim path.
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
            == 403
        )
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
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
        token = _captured_verification_link(caplog, client, email="ada@example.com", full_name="Ada Lovelace")
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
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
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
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
        token = _captured_verification_link(caplog, client, email="ada@example.com", full_name="A" * 255)
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )
        assert (
            client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD}).status_code
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
        assert [row["organization_member_id"] for row in after] == [row["organization_member_id"] for row in before]
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


def _signed_in_member(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> str:
    """Add, claim, verify and sign in a member; return their organization membership id."""
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": email, "role": "member"},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert added.status_code == 201, added.text
    token = _captured_verification_link(caplog, client, email=email)
    assert client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
    assert client.post(f"{API_ROOT}/auth/session", json={"email": email, "password": PASSWORD}).status_code == 200
    return str(added.json()["organization_member_id"])


def _remove(client: TestClient, organization_member_id: str) -> None:
    """Remove a member as the operator, leaving the member's own session cookie in place."""
    session = dict(client.cookies)
    client.cookies.clear()
    removed = client.delete(
        f"{API_ROOT}/organizations/me/members/{organization_member_id}", headers={"Otari-Key": MASTER_KEY}
    )
    assert removed.status_code == 200, removed.text
    client.cookies.update(session)


def test_a_member_can_create_an_organization_where_signup_is_closed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        _signed_in_member(client, caplog, email="ada@example.com")

        created = client.post(f"{API_ROOT}/organizations", json={"name": "Ada Corp"})

        assert created.status_code == 201, created.text


def test_a_removed_member_cannot_create_an_organization_where_signup_is_closed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Removal ends the membership but not the identity or its password, so the session still signs in.

    Creating an organization would make that identity a tenant of a deployment
    that admits nobody it was not invited into.
    """
    with _client(tmp_path) as client:
        membership_id = _signed_in_member(client, caplog, email="eve@example.com")
        _remove(client, membership_id)

        created = client.post(f"{API_ROOT}/organizations", json={"name": "Eve Corp"})

        assert created.status_code == 403, created.text
        memberships = client.get(f"{API_ROOT}/organizations/me/memberships")
        assert memberships.status_code == 200, memberships.text
        assert memberships.json()["count"] == 0


def test_a_removed_member_can_create_an_organization_where_signup_is_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Open signup already gives any address an organization of its own, so refusing here would protect nothing."""
    with _client(tmp_path, open_signup=True) as client:
        membership_id = _signed_in_member(client, caplog, email="eve@example.com")
        _remove(client, membership_id)

        created = client.post(f"{API_ROOT}/organizations", json={"name": "Eve Corp"})

        assert created.status_code == 201, created.text


def test_signup_requires_the_terms_where_the_deployment_publishes_them(tmp_path: Path) -> None:
    config = _config(tmp_path, open_signup=True).model_copy(update={"terms_url": "https://example.com/terms"})
    with TestClient(create_app(config)) as client:
        _sign_in_with_master_key(client)
        before = _identity_count(tmp_path)

        refused = _signup(client, email="new@example.com")
        assert refused.status_code == 400, refused.text
        assert _identity_count(tmp_path) == before

        accepted = _signup(client, email="new@example.com", terms_accepted=True)
        assert accepted.status_code == 200, accepted.text
        assert _identity_count(tmp_path) == before + 1


def _name_and_organization(tmp_path: Path, *, email: str) -> tuple[str | None, str]:
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        row = connection.execute(
            text(
                'SELECT "user".full_name, organization.name FROM "user" '
                'JOIN organization ON organization.id = "user".active_organization_id WHERE "user".email = :email'
            ),
            {"email": email},
        ).one()
    engine.dispose()
    return (row[0], row[1])


@pytest.mark.parametrize(
    ("submitted", "stored", "organization"),
    [
        ("   ", None, "ada's organization"),
        ("  Ada   Lovelace ", "Ada Lovelace", "Ada Lovelace's organization"),
    ],
    ids=["blank", "padded"],
)
def test_open_signup_stores_the_name_the_way_the_account_page_does(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, submitted: str, stored: str | None, organization: str
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        _captured_verification_link(caplog, client, email="ada@example.com", full_name=submitted)

    assert _name_and_organization(tmp_path, email="ada@example.com") == (stored, organization)


def _organization_ids(tmp_path: Path, *, email: str) -> tuple[str, str | None]:
    engine = create_engine(f"sqlite:///{tmp_path / 'signup-test.db'}")
    with engine.begin() as connection:
        row = connection.execute(
            text('SELECT active_organization_id, default_organization_id FROM "user" WHERE email = :email'),
            {"email": email},
        ).one()
    engine.dispose()
    return (str(uuid.UUID(str(row[0]))), str(uuid.UUID(str(row[1]))) if row[1] is not None else None)


@pytest.mark.parametrize("ending", ["revoked", "removed"])
def test_open_signup_gives_an_identity_nobody_still_wants_an_organization_of_its_own(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ending: str
) -> None:
    """What a new address gets, rather than an account pointed at an organization that let it go."""
    with _client(tmp_path, open_signup=True) as client:
        if ending == "revoked":
            invitation = _invite(client, email="ada@example.com")
            client.delete(
                f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}",
                headers={"Otari-Key": MASTER_KEY},
            )
        else:
            added = client.post(
                f"{API_ROOT}/organizations/me/members",
                json={"email": "ada@example.com", "role": "member"},
                headers={"Otari-Key": MASTER_KEY},
            )
            _remove(client, str(added.json()["organization_member_id"]))
        inviting_organization, _ = _organization_ids(tmp_path, email="ada@example.com")

        token = _captured_verification_link(caplog, client, email="ada@example.com")
        assert (
            client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD}).status_code == 200
        )
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert signed_in.status_code == 200, signed_in.text

        own = signed_in.json()["active_organization_id"]
        assert own is not None
        assert own != inviting_organization
        assert _organization_ids(tmp_path, email="ada@example.com") == (own, own)
        context = client.get(f"{API_ROOT}/organizations/me")
        assert context.status_code == 200, context.text
        assert context.json()["role"] == "owner"


def test_open_signup_gives_an_identity_nobody_still_wants_its_own_organization_through_a_reset_too(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The third way to claim an address follows the same rule as the signup form and a provider."""
    with _client(tmp_path, open_signup=True) as client:
        invitation = _invite(client, email="ada@example.com")
        client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}",
            headers={"Otari-Key": MASTER_KEY},
        )
        inviting_organization, _ = _organization_ids(tmp_path, email="ada@example.com")
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            requested = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "ada@example.com"})
        finally:
            gateway_logger.removeHandler(caplog.handler)
        assert requested.status_code == 200, requested.text
        match = _TOKEN_IN_LINK.search(caplog.text)
        assert match, caplog.text

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": match.group(1), "new_password": PASSWORD}
        )
        assert confirmed.status_code == 204, confirmed.text
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert signed_in.status_code == 200, signed_in.text

        own = signed_in.json()["active_organization_id"]
        assert own is not None
        assert own != inviting_organization
        assert _organization_ids(tmp_path, email="ada@example.com") == (own, own)
        assert client.get(f"{API_ROOT}/organizations/me").json()["role"] == "owner"


def test_a_reset_claim_whose_link_expires_while_its_organization_is_founded_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The link is checked in full again once the organization's commit has released the lock."""
    real = OrganizationService.provision_tenancy_for_claim

    async def provision_then_expire(self: OrganizationService, identity: User, **claim: Any) -> bool:
        founded = await real(self, identity, **claim)
        identity.password_reset_token_expires_at = datetime.now(UTC) - timedelta(minutes=1)
        await self.db.commit()
        return founded

    monkeypatch.setattr(OrganizationService, "provision_tenancy_for_claim", provision_then_expire)
    with _client(tmp_path, open_signup=True) as client:
        invitation = _invite(client, email="ada@example.com")
        client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}",
            headers={"Otari-Key": MASTER_KEY},
        )
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            client.post(f"{API_ROOT}/auth/password/reset", json={"email": "ada@example.com"})
        finally:
            gateway_logger.removeHandler(caplog.handler)
        match = _TOKEN_IN_LINK.search(caplog.text)
        assert match, caplog.text

        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": match.group(1), "new_password": PASSWORD}
        )

        assert confirmed.status_code == 400, confirmed.text
        assert confirmed.json()["detail"] == ResetTokenInvalidError().message
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
        assert signed_in.status_code == 401, signed_in.text


def test_open_signup_leaves_a_pending_invitee_where_the_invitation_is(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        _invite(client, email="ada@example.com")
        before = _organization_ids(tmp_path, email="ada@example.com")

        _captured_verification_link(caplog, client, email="ada@example.com")

        assert _organization_ids(tmp_path, email="ada@example.com") == before


def test_signing_in_after_removal_names_no_organization(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path) as client:
        membership_id = _signed_in_member(client, caplog, email="eve@example.com")
        _remove(client, membership_id)
        client.cookies.clear()

        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "eve@example.com", "password": PASSWORD})

        assert signed_in.status_code == 200, signed_in.text
        assert signed_in.json()["active_organization_id"] is None


def test_signing_in_moves_a_stale_pointer_to_an_organization_the_member_still_belongs_to(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        membership_id = _signed_in_member(client, caplog, email="ada@example.com")
        created = client.post(f"{API_ROOT}/organizations", json={"name": "Ada Corp"})
        assert created.status_code == 201, created.text
        _remove(client, membership_id)
        client.cookies.clear()

        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})

        assert signed_in.json()["active_organization_id"] == created.json()["id"]
        assert client.get(f"{API_ROOT}/workspaces").status_code == 200


@pytest.mark.parametrize(
    ("full_name", "expected"),
    [
        ("Ada\nLovelace", "Ada Lovelace's organization"),
        ("Ada\r\n\tLovelace", "Ada Lovelace's organization"),
        ("Ada Lovelace", "Ada Lovelace's organization"),
        ("Ada\x07", "Ada's organization"),
        ("\x07\n", "ada's organization"),
    ],
)
def test_a_default_organization_name_is_one_line_however_the_name_was_typed(full_name: str, expected: str) -> None:
    """The name is quoted in invitation emails, where a line break would start a line the deployment did not write."""
    assert _default_organization_name("ada@example.com", full_name) == expected


def _reset_token(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> str:
    """Ask for a reset link and pull its token out of the console log."""
    caplog.clear()
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        requested = client.post(f"{API_ROOT}/auth/password/reset", json={"email": email})
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert requested.status_code == 200, requested.text
    match = _TOKEN_IN_LINK.search(caplog.text)
    assert match, caplog.text
    return match.group(1)


def test_open_signup_gives_an_identity_whose_invitation_was_revoked_before_verifying_its_own_organization(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Signup claimed the address while the invitation waited; verifying it is when nobody wants it any more."""
    with _client(tmp_path, open_signup=True) as client:
        invitation = _invite(client, email="ada@example.com")
        inviting_organization, _ = _organization_ids(tmp_path, email="ada@example.com")
        token = _captured_verification_link(caplog, client, email="ada@example.com")
        assert _organization_ids(tmp_path, email="ada@example.com")[0] == inviting_organization
        revoked = client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}",
            headers={"Otari-Key": MASTER_KEY},
        )
        assert revoked.status_code == 200, revoked.text

        verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token, "password": PASSWORD})
        assert verified.status_code == 200, verified.text
        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})

        own = signed_in.json()["active_organization_id"]
        assert own is not None
        assert own != inviting_organization
        assert _organization_ids(tmp_path, email="ada@example.com") == (own, own)
        assert client.get(f"{API_ROOT}/organizations/me").json()["role"] == "owner"


def test_open_signup_gives_a_removed_vouched_member_its_own_organization_when_a_reset_proves_the_address(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A password chosen from an invitation link only vouches for the address; the reset is what claims it."""
    with _client(tmp_path, open_signup=True) as client:
        gateway_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger="gateway")
        try:
            invitation = _invite(client, email="ada@example.com")
        finally:
            gateway_logger.removeHandler(caplog.handler)
        match = _TOKEN_IN_LINK.search(caplog.text)
        assert match, caplog.text
        accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": match.group(1), "password": PASSWORD})
        assert accepted.status_code == 200, accepted.text
        inviting_organization, _ = _organization_ids(tmp_path, email="ada@example.com")
        _remove(client, str(invitation["organization_member_id"]))

        token = _reset_token(client, caplog, email="ada@example.com")
        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": "another-password"}
        )
        assert confirmed.status_code == 204, confirmed.text
        signed_in = client.post(
            f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": "another-password"}
        )

        own = signed_in.json()["active_organization_id"]
        assert own is not None
        assert own != inviting_organization
        assert client.get(f"{API_ROOT}/organizations/me").json()["role"] == "owner"


def test_open_signup_leaves_a_proven_member_who_was_removed_without_an_organization(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An address is claimed once; afterwards the person creates an organization from the empty state."""
    with _client(tmp_path, open_signup=True) as client:
        membership_id = _signed_in_member(client, caplog, email="eve@example.com")
        _remove(client, membership_id)
        client.cookies.clear()

        signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "eve@example.com", "password": PASSWORD})

        assert signed_in.status_code == 200, signed_in.text
        assert signed_in.json()["active_organization_id"] is None


def test_a_reset_that_takes_back_a_squatted_address_drops_the_strangers_name_and_renames_their_organization(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, open_signup=True) as client:
        _captured_verification_link(caplog, client, email="ada@example.com", full_name="Mallory Squat")
        assert _name_and_organization(tmp_path, email="ada@example.com") == (
            "Mallory Squat",
            "Mallory Squat's organization",
        )

        token = _reset_token(client, caplog, email="ada@example.com")
        confirmed = client.post(
            f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": "owner-password"}
        )
        assert confirmed.status_code == 204, confirmed.text

    assert _name_and_organization(tmp_path, email="ada@example.com") == (None, "ada's organization")


def test_the_context_of_a_caller_in_no_organization_says_so_without_naming_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        membership_id = _signed_in_member(client, caplog, email="eve@example.com")
        former, _ = _organization_ids(tmp_path, email="eve@example.com")
        _remove(client, membership_id)

        context = client.get(f"{API_ROOT}/organizations/me")

        assert context.status_code == 404, context.text
        assert context.json()["detail"] == "You are not an active member of any organization"
        assert former not in context.text


def test_removal_moves_the_active_organization_to_one_the_member_still_belongs_to(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path) as client:
        membership_id = _signed_in_member(client, caplog, email="ada@example.com")
        created = client.post(f"{API_ROOT}/organizations", json={"name": "Ada Corp"})
        assert created.status_code == 201, created.text

        _remove(client, membership_id)

        assert _organization_ids(tmp_path, email="ada@example.com")[0] == created.json()["id"]
