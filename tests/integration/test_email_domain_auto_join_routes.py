"""Auto-join over the sign-in routes themselves, not the service beneath them.

`test_organization_domains.py` calls ``auto_join_for_user`` directly, which
proves the rule and nothing about the wiring: delete any one of the three call
sites and that suite stays green while the feature dies on that route. These
tests sign in over HTTP instead, once per credential, so a missing call is a
failure rather than a silence.

The shape is the same each time. Ada is a member of the organization the
deployment boots (call it A) because every sign-in route refuses an address no
identity holds. The claim is on a *second* organization (B), so a membership
appearing in B is auto-join's doing and nothing else's, and Ada staying pointed
at A is the "never moves the pointer" rule observed from outside.
"""

import logging
import re
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.tenancy import DOMAIN_VERIFICATION_TXT_PREFIX, OrganizationMember
from gateway.services.tenancy import organization_domain_service as domain_service

from .webauthn_helpers import SoftwareAuthenticator, challenge_of

ORIGIN = "http://testserver"
RP_ID = "testserver"
PASSWORD = "a-real-password"  # pragma: allowlist secret
ADDRESS = "ada@acme.example"
DOMAIN = "acme.example"

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


@pytest.fixture
def claiming_organization(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """A second organization that has claimed and proven ``DOMAIN``. Returns its id."""
    created = client.post("/v1/organizations", json={"name": "Beta"}, headers=master_key_header)
    assert created.status_code == 201, created.text
    beta = str(created.json()["id"])

    switched = client.post(
        "/v1/organizations/me/switch",
        json={"organization_id": beta},
        headers=master_key_header,
    )
    assert switched.status_code == 200, switched.text

    claim = client.post("/v1/organizations/me/domains", json={"domain": DOMAIN}, headers=master_key_header)
    assert claim.status_code == 201, claim.text
    record = claim.json()["verification_record"]
    assert record.startswith(DOMAIN_VERIFICATION_TXT_PREFIX)

    async def _resolve(domain: str) -> list[str]:
        return [record] if domain == DOMAIN else []

    monkeypatch.setattr(domain_service, "resolve_txt_records", _resolve)
    verified = client.post(
        f"/v1/organizations/me/domains/{claim.json()['id']}/verify",
        headers=master_key_header,
    )
    assert verified.status_code == 200, verified.text
    assert verified.json()["verified_at"] is not None
    return beta


def _rostered_identity(client: TestClient, master_key_header: dict[str, str]) -> str:
    """Put ADDRESS on the booted organization's roster, which every route requires."""
    added = client.post(
        "/v1/organizations/me/members",
        json={"email": ADDRESS, "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    return str(added.json()["user_id"])


def _claim_password(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sign up on the rostered address and verify it, so a password sign-in works."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        signed_up = client.post("/v1/auth/signup", json={"email": ADDRESS, "password": PASSWORD})
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert signed_up.status_code == 200, signed_up.text
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    assert client.post("/v1/auth/verify-email", json={"token": token.group(1)}).status_code == 200


def _memberships(client: TestClient) -> list[dict[str, Any]]:
    """The organizations the currently-signed-in caller belongs to."""
    response = client.get("/v1/organizations/me/memberships")
    assert response.status_code == 200, response.text
    data: list[dict[str, Any]] = response.json()["data"]
    return data


def _assert_joined(client: TestClient, signed_in: Any, *, beta: str, home: str) -> None:
    assert signed_in.status_code == 200, signed_in.text
    joined = {row["organization"]["id"] for row in _memberships(client)}
    assert beta in joined, f"auto-join did not run on this route; memberships were {joined}"
    # The pointer is untouched: auto-join adds a membership, it does not decide
    # where the caller lands.
    assert signed_in.json()["active_organization_id"] == home


def test_a_password_sign_in_joins_the_organization_that_proved_the_domain(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    claiming_organization: str,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    _rostered_identity(client, master_key_header)
    _claim_password(client, caplog)
    home = client.get("/v1/organizations/me", headers=master_key_header).json()["organization"]["id"]
    client.cookies.clear()

    signed_in = client.post("/v1/auth/session", json={"email": ADDRESS, "password": PASSWORD})

    _assert_joined(client, signed_in, beta=claiming_organization, home=home)


def test_an_oauth_sign_in_joins_the_organization_that_proved_the_domain(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    claiming_organization: str,
) -> None:
    from gateway.services.oauth_service import OAuthIdentity

    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")

    async def _exchange(_config: GatewayConfig, provider: str, *, code: str) -> OAuthIdentity:
        return OAuthIdentity(provider=provider, email=ADDRESS, full_name="Ada", email_verified=True)

    monkeypatch.setattr("gateway.api.routes.auth_oauth.exchange_code", _exchange)
    _rostered_identity(client, master_key_header)
    home = client.get("/v1/organizations/me", headers=master_key_header).json()["organization"]["id"]
    client.cookies.clear()

    signed_in = client.post("/v1/auth/oauth/google/callback", json={"code": "the-code"})

    _assert_joined(client, signed_in, beta=claiming_organization, home=home)


def test_a_passkey_sign_in_joins_the_organization_that_proved_the_domain(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    claiming_organization: str,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    _rostered_identity(client, master_key_header)
    _claim_password(client, caplog)
    home = client.get("/v1/organizations/me", headers=master_key_header).json()["organization"]["id"]
    client.cookies.clear()

    # Ada registers a passkey against her own session, then signs in with it
    # alone: the passkey route is the one under test, not the password.
    assert client.post("/v1/auth/session", json={"email": ADDRESS, "password": PASSWORD}).status_code == 200
    authenticator = SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN)
    options = client.post("/v1/auth/webauthn/register/options")
    assert options.status_code == 200, options.text
    registered = client.post(
        "/v1/auth/webauthn/register",
        json={"credential": authenticator.register(challenge_of(options.json()))},
    )
    assert registered.status_code == 201, registered.text
    client.cookies.clear()

    challenge = client.post("/v1/auth/webauthn/authenticate/options")
    assert challenge.status_code == 200, challenge.text
    signed_in = client.post(
        "/v1/auth/webauthn/authenticate",
        json={"credential": authenticator.authenticate(challenge_of(challenge.json()))},
    )

    _assert_joined(client, signed_in, beta=claiming_organization, home=home)


def test_a_membership_does_not_survive_a_sign_in_that_fails_after_it_is_staged(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    test_db: Session,
    claiming_organization: str,
) -> None:
    """The transactional claim, which nothing else exercises.

    Auto-join stages the membership and the sign-in commits it. If the session
    row fails after that, the membership has to go with it: a caller who was
    told their sign-in failed must not have quietly gained access to a tenant.
    """
    monkeypatch.setattr(test_config, "mail_transport", "console")
    user_id = _rostered_identity(client, master_key_header)
    _claim_password(client, caplog)
    client.cookies.clear()

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise SQLAlchemyError("the session row could not be written")

    monkeypatch.setattr("gateway.api.routes.auth_session.create_dashboard_session", _explode)
    refused = client.post("/v1/auth/session", json={"email": ADDRESS, "password": PASSWORD})
    assert refused.status_code == 500, refused.text

    # Read on a connection of its own, so what is asserted is what committed
    # rather than anything the request's own session still held.
    stranded = test_db.execute(
        select(OrganizationMember).where(
            col(OrganizationMember.user_id) == uuid.UUID(user_id),
            col(OrganizationMember.organization_id) == uuid.UUID(claiming_organization),
        )
    ).scalars().all()
    assert list(stranded) == [], "the failed sign-in left a membership behind"

    # And the same sign-in, once it can complete, does create it: the assertion
    # above is about the rollback, not about auto-join being broken here.
    monkeypatch.undo()
    monkeypatch.setattr(test_config, "mail_transport", "console")
    healthy = client.post("/v1/auth/session", json={"email": ADDRESS, "password": PASSWORD})
    assert healthy.status_code == 200, healthy.text
    assert claiming_organization in {row["organization"]["id"] for row in _memberships(client)}
