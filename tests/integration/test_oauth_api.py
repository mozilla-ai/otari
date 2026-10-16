"""OAuth sign-in end to end, with only the provider's own endpoints stubbed.

Everything on this side of the exchange is real: the routes, the container, the
``IdentityProviderPort`` adapter the base build binds, the roster it resolves
against, and the session cookie a sign-in mints. What is replaced is
apron-auth's outbound half, because a test cannot complete a consent screen.

That replacement is the reason
``tests/integration/test_oauth_live_provider.py`` exists: it is the check that
the request shape apron-auth actually sends is one Google and GitHub accept, and
nothing here can stand in for it.
"""

import logging
import re
import uuid
from base64 import urlsafe_b64encode
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth import OAuthClient
from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlmodel import col, delete

from gateway.api.routes import auth_oauth
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.exceptions.identity_exceptions import OAuthIdentityLinkConflictError, OAuthProviderUnavailableError
from gateway.log_config import logger as gateway_logger
from gateway.models.identity import EmailAssertion, OAuthIdentityLink, ProviderIdentity
from gateway.models.tenancy import Organization, User, WebAuthnCredential
from gateway.models.tools import WorkspaceCodeExecutionPolicy
from gateway.repositories.identity import DashboardSessionRepository
from gateway.repositories.tenancy import UserRepository
from gateway.services import oauth_service
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.services.identity import IdentityService
from gateway.services.oauth_service import FLOW_COOKIE_NAME, FLOW_COOKIE_PATH
from gateway.services.tenancy.organization_service import OrganizationService, SignupRegistration

ORIGIN = "http://testserver"
PASSWORD = "a-real-password"  # pragma: allowlist secret
UNPROVEN_PASSWORD = "set-before-verification"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


@pytest.fixture
def oauth_configured(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register both providers on the deployment TestClient serves."""
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")
    monkeypatch.setattr(test_config, "oauth_github_client_id", "github-id")
    monkeypatch.setattr(test_config, "oauth_github_client_secret", "github-secret")


def stub_exchange(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subject: str = "ada-subject",
    email: str | None = "ada@example.com",
    full_name: str | None = "Ada Lovelace",
    email_verified: EmailAssertion = EmailAssertion.VERIFIED,
) -> list[str]:
    """Replace the provider round trip, and record every code it was handed.

    Patched on the route module's own reference, so the substitution is visible
    from the handler rather than depending on how it imported the service.
    """
    spent: list[str] = []

    async def _exchange(
        _config: GatewayConfig, provider: str, *, code: str, state: str, flow_secret: str | None, db: Any
    ) -> ProviderIdentity:
        spent.append(code)
        return ProviderIdentity(
            provider=provider,
            subject=subject,
            email=email,
            full_name=full_name,
            email_verified=email_verified,
        )

    monkeypatch.setattr("gateway.api.routes.auth_oauth.exchange_code", _exchange)
    return spent


def _identity(db_session: Session, email: str) -> User:
    """The tenancy identity holding ``email``, read outside the test client."""
    identity = db_session.execute(select(User).where(col(User.email) == email)).scalar_one()
    return identity


def add_member(client: TestClient, master_key_header: dict[str, str], *, email: str) -> str:
    """Put an address on the roster, the way an operator does, and return its id."""
    response = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": email, "role": "member"},
        headers=master_key_header,
    )
    assert response.status_code == 201, response.text
    member: dict[str, Any] = response.json()
    return str(member["user_id"])


# ---------- what the deployment publishes ----------


def test_the_bootstrap_offers_no_provider_until_one_is_configured(client: TestClient) -> None:
    # The default. This is what makes the sign-in screen carry no OAuth
    # affordance out of the box rather than a pair of dead buttons.
    bootstrap = client.get(f"{API_ROOT}/bootstrap")

    assert bootstrap.status_code == 200, bootstrap.text
    assert bootstrap.json()["oauth_providers"] == []


def test_the_bootstrap_names_the_providers_an_operator_configured(client: TestClient, oauth_configured: None) -> None:
    assert client.get(f"{API_ROOT}/bootstrap").json()["oauth_providers"] == ["github", "google"]


def test_a_provider_missing_its_secret_is_not_published(
    client: TestClient, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")

    assert client.get(f"{API_ROOT}/bootstrap").json()["oauth_providers"] == []


# ---------- starting the flow ----------


@pytest.mark.parametrize("provider", ["google", "github"])
def test_authorize_hands_back_a_consent_url_and_a_fresh_state(
    client: TestClient, oauth_configured: None, provider: str
) -> None:
    first = client.get(f"{API_ROOT}/auth/oauth/{provider}/authorize")
    second = client.get(f"{API_ROOT}/auth/oauth/{provider}/authorize")

    assert first.status_code == 200, first.text
    query = parse_qs(urlsplit(first.json()["authorization_url"]).query)
    assert query["redirect_uri"] == [f"{ORIGIN}/auth/{provider}/callback"]
    assert query["state"] == [first.json()["state"]]
    # A fresh value per request: only the one the browser kept is the one it
    # will compare against.
    assert first.json()["state"] != second.json()["state"]


def test_authorize_needs_no_credential(client: TestClient, oauth_configured: None) -> None:
    # It is how somebody who holds nothing starts signing in, so requiring a
    # credential would be circular.
    assert client.get(f"{API_ROOT}/auth/oauth/google/authorize").status_code == 200


def test_the_flow_cookie_reaches_every_route_that_reads_it(client: TestClient, oauth_configured: None) -> None:
    """A path-scoped cookie is only sent to paths under its own path.

    Read off the wire and compared against the routing table, so neither side
    is a literal that can drift from the other. Scope the cookie more narrowly
    than the routes it guards and the exchange refuses every sign-in as though
    the state had expired, which is a routing fault wearing an expiry's face.
    """
    response = client.get(f"{API_ROOT}/auth/oauth/google/authorize")
    assert response.status_code == 200, response.text

    jar: SimpleCookie = SimpleCookie()
    for header in response.headers.get_list("set-cookie"):
        jar.load(header)
    cookie_path = jar[FLOW_COOKIE_NAME]["path"]
    assert cookie_path, "the flow cookie is unscoped, so this check would hold vacuously"

    app = cast(FastAPI, client.app)
    served = [
        route.path
        for route in iter_route_contexts(app.routes)
        if route.path
        and isinstance(route.original_route, APIRoute)
        and route.endpoint.__module__ == auth_oauth.__name__
    ]
    assert served, "no OAuth routes are mounted, so this check would hold vacuously"
    withheld = [path for path in served if not (path == cookie_path or path.startswith(f"{cookie_path}/"))]
    assert not withheld, f"the flow cookie is scoped to {cookie_path}, so the browser withholds it from: {withheld}"


def test_authorize_refuses_an_unconfigured_provider_and_names_the_settings(
    client: TestClient,
) -> None:
    response = client.get(f"{API_ROOT}/auth/oauth/google/authorize")

    assert response.status_code == 503
    assert "oauth_google_client_id" in response.json()["detail"]


def test_a_provider_this_deployment_could_never_configure_is_not_a_route(
    client: TestClient, oauth_configured: None
) -> None:
    # The path parameter is bounded by the config vocabulary, so an unknown
    # segment is refused by the framework rather than by a handler.
    assert client.get(f"{API_ROOT}/auth/oauth/not-a-provider/authorize").status_code == 422


# ---------- finishing the flow ----------


def test_a_rostered_member_signs_in_and_gets_the_same_session_a_password_would(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    spent = stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "the-code", "state": "s"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_id"] == user_id
    assert body["active_organization_id"]
    # The token travels only in the cookie, exactly as the password and passkey
    # sign-ins do.
    assert SESSION_COOKIE_NAME in response.cookies
    assert "token" not in body
    assert spent == ["the-code"]

    # And the cookie authenticates the management API on its own, with no header
    # credential: the provider minted the same session a password would have.
    assert client.cookies.get(SESSION_COOKIE_NAME)
    membership = client.get(f"{API_ROOT}/organizations/me")
    assert membership.status_code == 200, membership.text
    assert membership.json()["organization"]["id"] == body["active_organization_id"]


def test_the_provider_is_recorded_on_the_identity_it_signed_in(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # Read from the column rather than an endpoint: `user.oauth_provider` is
    # carried for schema parity with the platform and no route publishes it, so
    # the link is only observable here.
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    signed_in = client.post(f"{API_ROOT}/auth/oauth/github/callback", json={"code": "c", "state": "s"})
    assert signed_in.status_code == 200, signed_in.text

    identity = _identity(db_session, "ada@example.com")
    assert identity.oauth_provider == "github"
    # And the provider's assertion lifted the local verification gate.
    assert identity.email_verified_at is not None


def test_a_verified_provider_address_lifts_the_local_verification_gate(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A member an operator added has never verified their address here, and the
    # password sign-in hard-blocks that. The provider's assertion is a stronger
    # proof of the same fact, so this is how a deployment that cannot send mail
    # still lets a member in.
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200


def test_a_provider_sign_in_reports_that_it_holds_no_password_and_can_set_one(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mozilla-ai/otari-ai#2099: the account page has to be able to tell, and the way out has to exist.

    Somebody who signs in with Google or GitHub holds no password, so the change
    form asks them for a current one they can never supply. The dashboard picks
    its form from ``caller.has_password`` rather than from the deployment-wide
    ``sign_in_methods``, and this pins both halves: the flag is what this
    identity actually holds, and a cookie-authenticated caller who holds none
    sets a first password with nothing further to prove.
    """
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200

    before = client.get(f"{API_ROOT}/organizations/me")
    assert before.status_code == 200, before.text
    assert before.json()["caller"]["has_password"] is False
    # A member's first password leaves an unclaimed deployment on the master key.
    assert before.json()["caller"]["claims_deployment"] is False

    set_password = client.put(f"{API_ROOT}/auth/password", json={"new_password": PASSWORD})

    assert set_password.status_code == 200, set_password.text
    assert set_password.json() == {"email": "ada@example.com", "master_key_sign_in_retired": False}
    after = client.get(f"{API_ROOT}/organizations/me")
    assert after.json()["caller"]["has_password"] is True


def test_a_second_change_from_that_session_asks_for_the_password_it_now_holds(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other side of the flag: once it is true the change form is the right
    # one, and the gateway refuses the shape that was correct a moment earlier.
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert client.put(f"{API_ROOT}/auth/password", json={"new_password": PASSWORD}).status_code == 200

    again = client.put(f"{API_ROOT}/auth/password", json={"new_password": "another-password"})

    assert again.status_code == 400
    assert "current password" in again.json()["detail"].lower()


def test_an_address_nobody_put_on_the_roster_is_refused_rather_than_provisioned(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With open_signup disabled (the default), a verified Google address alone must not create an account.
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert "not registered on this gateway" in response.json()["detail"]
    assert SESSION_COOKIE_NAME not in response.cookies


def _invite(client: TestClient, master_key_header: dict[str, str], *, email: str) -> dict[str, Any]:
    response = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers=master_key_header,
    )
    assert response.status_code == 201, response.text
    invitation: dict[str, Any] = response.json()
    return invitation


def test_an_invitee_signs_in_with_a_provider_before_accepting(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed signup still admits an address with an invitation it can accept, which then waits in its inbox."""
    _invite(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 200, response.text
    pending = client.get(f"{API_ROOT}/organizations/me/pending-memberships")
    assert pending.status_code == 200, pending.text
    assert pending.json()["count"] == 1


@pytest.mark.parametrize("ending", ["revoked", "removed"])
def test_a_provider_cannot_claim_an_identity_the_roster_no_longer_holds(
    client: TestClient,
    db_session: Session,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
) -> None:
    """Inviting or adding an address creates its identity, and ending that leaves the identity behind.

    The signup form refuses to claim it where signup is closed, and a provider
    sign-in is the same claim by another route.
    """
    if ending == "revoked":
        invitation = _invite(client, master_key_header, email="ada@example.com")
        revoked = client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}",
            headers=master_key_header,
        )
        assert revoked.status_code == 200, revoked.text
    else:
        added = client.post(
            f"{API_ROOT}/organizations/me/members",
            json={"email": "ada@example.com", "role": "member"},
            headers=master_key_header,
        )
        assert added.status_code == 201, added.text
        removed = client.delete(
            f"{API_ROOT}/organizations/me/members/{added.json()['organization_member_id']}",
            headers=master_key_header,
        )
        assert removed.status_code == 200, removed.text
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert "not registered on this gateway" in response.json()["detail"]
    assert SESSION_COOKIE_NAME not in response.cookies
    identity = _identity(db_session, "ada@example.com")
    db_session.refresh(identity)
    assert identity.email_verified_at is None
    assert identity.oauth_provider is None


def test_a_removed_member_who_already_signed_in_with_a_provider_still_can(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule is about claiming an account, not signing in to one: a password sign-in is not refused here either."""
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "ada@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    stub_exchange(monkeypatch)
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200
    client.cookies.clear()
    removed = client.delete(
        f"{API_ROOT}/organizations/me/members/{added.json()['organization_member_id']}",
        headers=master_key_header,
    )
    assert removed.status_code == 200, removed.text

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": "s"})

    assert response.status_code == 200, response.text


@pytest.fixture
def signup_open(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable open_signup on the deployment that TestClient serves."""
    monkeypatch.setattr(test_config, "open_signup", True)


def test_open_signup_registers_an_address_nobody_holds_and_signs_it_in(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 200, response.text
    assert SESSION_COOKIE_NAME in response.cookies
    identity = _identity(db_session, "stranger@example.com")
    assert str(identity.id) == response.json()["user_id"]


def test_the_registered_identity_lands_the_tenancy_the_signup_form_lands(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read the tenancy through the new session, as the dashboard does.
    stub_exchange(monkeypatch, email="stranger@example.com", full_name="A Stranger")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert response.status_code == 200, response.text

    membership = client.get(f"{API_ROOT}/organizations/me")
    assert membership.status_code == 200, membership.text
    body = membership.json()
    assert body["organization"]["id"] == response.json()["active_organization_id"]
    assert body["role"] == "owner"
    workspaces = client.get(f"{API_ROOT}/workspaces")
    assert workspaces.status_code == 200, workspaces.text
    assert workspaces.json()["data"], "the registered tenant has no workspace"


def test_a_hosted_open_signup_starts_its_workspace_with_code_execution_on(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """A hosted control plane reads no policy as off, so the tenant this founds starts with one turned on."""
    monkeypatch.setattr(test_config, "mode", "hosted")
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert response.status_code == 200, response.text

    workspaces = {uuid.UUID(row["id"]) for row in client.get(f"{API_ROOT}/workspaces").json()["data"]}
    policies = {
        row.workspace_id: row.enabled for row in db_session.execute(select(WorkspaceCodeExecutionPolicy)).scalars()
    }
    assert workspaces
    assert all(policies.get(workspace) is True for workspace in workspaces)


def test_a_registered_identity_holds_the_provider_and_a_verified_address_and_no_password(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    stub_exchange(monkeypatch, email="stranger@example.com", full_name="A Stranger")

    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200

    identity = _identity(db_session, "stranger@example.com")
    assert identity.oauth_provider == "google"
    assert identity.email_verified_at is not None
    assert identity.hashed_password is None
    assert identity.full_name == "A Stranger"


def test_a_second_sign_in_on_a_registered_address_founds_no_second_tenant(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_exchange(monkeypatch, email="stranger@example.com")

    first = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    client.cookies.clear()
    second = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": "s"})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["user_id"] == first.json()["user_id"]
    assert second.json()["active_organization_id"] == first.json()["active_organization_id"]


def test_open_signup_registers_nobody_from_an_address_the_provider_would_not_verify(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # Creating an account for an unverified address would let anyone hold an account on another person's address.
    stub_exchange(monkeypatch, email="stranger@example.com", email_verified=EmailAssertion.UNVERIFIED)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert db_session.execute(select(User).where(col(User.email) == "stranger@example.com")).first() is None


def test_open_signup_does_not_register_a_deactivated_identity_afresh(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # A new account here would undo the deactivation.
    add_member(client, master_key_header, email="ada@example.com")
    identity = _identity(db_session, "ada@example.com")
    identity.is_active = False
    db_session.add(identity)
    db_session.commit()
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert "not registered on this gateway" in response.json()["detail"]
    identities = db_session.execute(select(User).where(col(User.email) == "ada@example.com")).scalars().all()
    assert len(identities) == 1
    assert not identities[0].is_active


def test_an_account_deleted_during_its_sign_in_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """An account deleted while its sign-in is in progress is refused."""
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    original = UserRepository.get_locked

    async def _delete_first(self: UserRepository, user_id: uuid.UUID) -> User | None:
        db_session.execute(delete(User).where(col(User.id) == user_id))
        db_session.commit()
        return await original(self, user_id)

    monkeypatch.setattr(UserRepository, "get_locked", _delete_first)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert SESSION_COOKIE_NAME not in response.cookies


def test_an_account_deactivated_during_its_sign_in_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """An account deactivated while its sign-in is in progress is refused, and nothing is written."""
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    original = UserRepository.get_locked

    async def _deactivate_first(self: UserRepository, user_id: uuid.UUID) -> User | None:
        identity = _identity(db_session, "ada@example.com")
        identity.is_active = False
        db_session.add(identity)
        db_session.commit()
        return await original(self, user_id)

    monkeypatch.setattr(UserRepository, "get_locked", _deactivate_first)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert SESSION_COOKIE_NAME not in response.cookies
    db_session.expire_all()
    refused = _identity(db_session, "ada@example.com")
    assert refused.oauth_provider is None
    assert refused.email_verified_at is None


def test_an_account_that_moves_to_another_address_during_its_sign_in_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """An account whose address changes while its sign-in is in progress is refused."""
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    original = UserRepository.get_locked

    async def _move_first(self: UserRepository, user_id: uuid.UUID) -> User | None:
        identity = _identity(db_session, "ada@example.com")
        identity.email = "grace@example.com"
        db_session.add(identity)
        db_session.commit()
        return await original(self, user_id)

    monkeypatch.setattr(UserRepository, "get_locked", _move_first)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert SESSION_COOKIE_NAME not in response.cookies
    db_session.expire_all()
    moved = _identity(db_session, "grace@example.com")
    assert moved.oauth_provider is None
    assert moved.email_verified_at is None


def test_a_lost_registration_race_signs_in_to_the_winner_rather_than_failing(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """A sign-in that loses the registration race signs in to the account the winner created.

    NOTE: The test runs the race in sequence, because SQLite has no row locks.
    The winning account is committed after the adapter reads the address and before it inserts.
    """
    organization = db_session.execute(select(Organization)).scalars().first()
    assert organization is not None
    original = OrganizationService.provision_signup_tenancy

    async def _lose_the_race(self: OrganizationService, *, email: str, full_name: str | None) -> SignupRegistration:
        db_session.add(User(email=email, is_active=True, active_organization_id=organization.id))
        db_session.commit()
        return await original(self, email=email, full_name=full_name)

    monkeypatch.setattr(OrganizationService, "provision_signup_tenancy", _lose_the_race)
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 200, response.text
    assert SESSION_COOKIE_NAME in response.cookies
    identities = db_session.execute(select(User).where(col(User.email) == "stranger@example.com")).scalars().all()
    assert len(identities) == 1
    assert response.json()["user_id"] == str(identities[0].id)


def test_open_signup_claims_a_rostered_address_rather_than_founding_a_second_tenant(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An existing member signs in to their organization, and no new organization is created.
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == user_id


def test_a_provider_sign_in_after_removal_names_no_organization(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same answer a password sign-in gives, rather than the organization the member was removed from."""
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "ada@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    member_id = added.json()["organization_member_id"]
    stub_exchange(monkeypatch)
    first = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c1", "state": "s"})
    assert first.status_code == 200, first.text
    assert first.json()["active_organization_id"] is not None
    client.cookies.clear()
    removed = client.delete(f"{API_ROOT}/organizations/me/members/{member_id}", headers=master_key_header)
    assert removed.status_code == 200, removed.text

    after_removal = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": "s"})

    assert after_removal.status_code == 200, after_removal.text
    assert after_removal.json()["active_organization_id"] is None


def test_open_signup_gives_a_revoked_invitee_an_organization_of_its_own(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    invitation = _invite(client, master_key_header, email="ada@example.com")
    revoked = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{invitation['invitation_id']}", headers=master_key_header
    )
    assert revoked.status_code == 200, revoked.text
    inviting_organization = _identity(db_session, "ada@example.com").active_organization_id
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 200, response.text
    own = response.json()["active_organization_id"]
    assert own is not None
    assert own != str(inviting_organization)
    db_session.expire_all()
    identity = _identity(db_session, "ada@example.com")
    assert str(identity.default_organization_id) == own
    assert client.get(f"{API_ROOT}/organizations/me").json()["role"] == "owner"


def test_an_unverified_provider_address_is_refused_even_when_it_is_on_the_roster(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, email_verified=EmailAssertion.UNVERIFIED)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert "did not confirm that address is yours" in response.json()["detail"]
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_provider_that_returns_no_address_at_all_is_refused(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_exchange(monkeypatch, email=None)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    # Nothing to verify, so the advice is to add an address, not to verify one.
    assert "Google did not share an email address" in response.json()["detail"]


def test_a_provider_outage_answers_502_with_the_same_message(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _unavailable(*_args: Any, **_kwargs: Any) -> ProviderIdentity:
        raise OAuthProviderUnavailableError("Google")

    monkeypatch.setattr("gateway.api.routes.auth_oauth.exchange_code", _unavailable)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Google did not complete the sign-in. Try again."}


def test_a_deactivated_identity_cannot_sign_in_with_a_provider_either(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # Deactivating somebody has to close every road in, or OAuth becomes the
    # door left open behind them. Flipped in the database because this edition
    # exposes no route that deactivates a tenancy identity; `/api/v1/users` is the
    # request-plane spend identity, which is a different table.
    add_member(client, master_key_header, email="ada@example.com")
    identity = _identity(db_session, "ada@example.com")
    identity.is_active = False
    db_session.add(identity)
    db_session.commit()
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    # Collapsed into the unknown-identity refusal rather than saying "switched
    # off", so somebody an operator shut out cannot keep confirming their
    # account is still on file.
    assert "not registered on this gateway" in response.json()["detail"]


def test_a_differently_cased_provider_address_still_finds_its_roster_row(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, email="Ada@Example.COM")

    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200


def test_the_callback_refuses_an_unconfigured_provider_before_spending_anything(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate is a dependency ahead of the handler, so it holds even with the
    # exchange stubbed out. It was a check inside the exchange first, and this
    # test is what showed that a caller could reach the identity resolution of a
    # provider this deployment never configured.
    spent = stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 503
    assert "oauth_google_client_id" in response.json()["detail"]
    assert spent == []


def test_maintenance_mode_freezes_an_oauth_sign_in_before_the_exchange(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The freeze is on starting a session, not on a credential, so an OAuth
    # sign-in has to answer to it or the switch is bypassable by anybody holding
    # a Google account. Refused before the exchange, so a frozen deployment
    # spends nobody's single-use authorization code.
    add_member(client, master_key_header, email="ada@example.com")
    frozen = client.patch(f"{API_ROOT}/settings/maintenance-mode", json={"enabled": True}, headers=master_key_header)
    assert frozen.status_code == 200, frozen.text
    spent = stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 503
    assert spent == []
    assert SESSION_COOKIE_NAME not in response.cookies


def test_the_callback_body_carries_the_code_and_nothing_else_the_server_trusts(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No redirect_uri: it is derived from public_base_url, so a browser cannot
    # choose what this server sends to a provider. (``state`` is a real field
    # now and is honored; this asserts only that ``redirect_uri`` is not.)
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    response = client.post(
        f"{API_ROOT}/auth/oauth/google/callback",
        json={
            "code": "c",
            "redirect_uri": "https://attacker.example.com/callback",
            "state": "anything",
        },
    )

    assert response.status_code == 200, response.text
    # The extra fields were ignored rather than honored: the URI this deployment
    # would send is still its own, whatever the body asked for.
    assert oauth_service.redirect_uri(test_config, "google") == f"{ORIGIN}/auth/google/callback"


def test_an_oversized_code_is_refused_before_any_outbound_call(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spent = stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "x" * 4096, "state": "s"})

    assert response.status_code == 422
    assert spent == []


def test_a_database_failure_while_staging_the_session_rolls_back_and_says_nothing_more(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging the session row is inside the route's error handling, not beside it.

    Starting a session prunes expired rows with a DELETE before it
    stages anything, so it can fail on its own rather than only at commit time.
    Outside the handled block that failure skipped the rollback and surfaced as
    a bare 500 from the generic handler.
    """
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    async def _explode(*_args: Any, **_kwargs: Any) -> tuple[str, Any]:
        raise SQLAlchemyError("pruning failed")

    monkeypatch.setattr(DashboardSessionRepository, "prune_expired", _explode)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 500
    # The generic wording, not the exception: the error-detail boundary holds on
    # this path the way it does on the commit path beside it.
    assert response.json()["detail"] == "Database error"
    assert "pruning failed" not in response.text
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_sign_in_that_loses_a_race_is_not_counted_as_an_auth_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    failures: list[str] = []
    monkeypatch.setattr("gateway.api.routes.auth_oauth.record_auth_failure", failures.append)

    async def _lose(*_args: Any, **_kwargs: Any) -> Any:
        raise OAuthIdentityLinkConflictError("google")

    monkeypatch.setattr(IdentityService, "sign_in_with_oauth", _lose)

    response = _sign_in(client)

    assert response.status_code == 409, response.text
    assert failures == []


def test_a_session_that_cannot_be_stored_leaves_the_account_and_its_links_as_they_were(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    async def _explode(*_args: Any, **_kwargs: Any) -> tuple[str, Any]:
        raise SQLAlchemyError("pruning failed")

    monkeypatch.setattr(DashboardSessionRepository, "prune_expired", _explode)

    assert _sign_in(client).status_code == 500

    assert _links(db_session, user_id) == []
    identity = _identity(db_session, "ada@example.com")
    assert (identity.email_verified_at, identity.oauth_provider) == (None, None)


@pytest.fixture
def signup_configured(test_config: GatewayConfig, oauth_configured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let signup mail its verification link, on a deployment that also offers both providers."""
    monkeypatch.setattr(test_config, "mail_transport", "console")


def sign_up(client: TestClient, *, email: str, password: str) -> None:
    """Set a password through signup and confirm that it was stored.

    Signup answers 200 whether or not it wrote anything.
    Sign-in answers 403 only for a correct password on an unverified address.
    """
    signed_up = client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": password})
    assert signed_up.status_code == 200, signed_up.text
    locked = client.post(f"{API_ROOT}/auth/session", json={"email": email, "password": password})
    assert locked.status_code == 403, locked.text


@pytest.mark.parametrize("open_signup", [True, False], ids=["open-signup", "admin-added"])
def test_a_provider_sign_in_does_not_unlock_a_password_set_before_it(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    open_signup: bool,
) -> None:
    # The provider confirms who owns the address, not who set the password.
    monkeypatch.setattr(test_config, "open_signup", open_signup)
    if not open_signup:
        add_member(client, master_key_header, email="ada@example.com")
    sign_up(client, email="ada@example.com", password=UNPROVEN_PASSWORD)
    stub_exchange(monkeypatch)

    provider_sign_in = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert provider_sign_in.status_code == 200, provider_sign_in.text
    assert SESSION_COOKIE_NAME in provider_sign_in.cookies
    client.cookies.clear()

    password_sign_in = client.post(
        f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": UNPROVEN_PASSWORD}
    )

    assert password_sign_in.status_code == 401, password_sign_in.text
    identity = _identity(db_session, "ada@example.com")
    assert identity.email_verified_at is not None
    assert identity.hashed_password is None
    assert identity.email_verification_token_hash is None
    assert identity.email_verification_token_expires_at is None


def test_a_password_on_an_already_verified_address_survives_linking_a_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    db_session: Session,
) -> None:
    # The mailed link proved that the password belongs to the owner of the address.
    add_member(client, master_key_header, email="ada@example.com")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        sign_up(client, email="ada@example.com", password=PASSWORD)
    finally:
        gateway_logger.removeHandler(caplog.handler)
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    assert (
        client.post(f"{API_ROOT}/auth/verify-email", json={"token": token.group(1), "password": PASSWORD}).status_code
        == 200
    )
    stub_exchange(monkeypatch)

    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200
    client.cookies.clear()

    assert _identity(db_session, "ada@example.com").oauth_provider == "google"
    password_sign_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert password_sign_in.status_code == 200, password_sign_in.text


def test_a_linked_sign_in_proves_an_address_an_operator_link_left_vouched(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    db_session: Session,
) -> None:
    # A password chosen through a shared operator link never reached the mailbox, so the owner signing in
    # through their existing provider link takes the account back, as a first sign-in would.
    add_member(client, master_key_header, email="ada@example.com")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        sign_up(client, email="ada@example.com", password=PASSWORD)
    finally:
        gateway_logger.removeHandler(caplog.handler)
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token.group(1), "password": PASSWORD})
    assert verified.status_code == 200, verified.text
    stub_exchange(monkeypatch)
    linked = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert linked.status_code == 200, linked.text
    assert linked.json()["password_removed"] is False
    client.cookies.clear()

    monkeypatch.setattr(test_config, "mail_transport", "none")
    identity = _identity(db_session, "ada@example.com")
    issued = client.post(f"{API_ROOT}/admin/users/{identity.id}/password-link", headers=master_key_header)
    assert issued.status_code == 201, issued.text
    shared_token = issued.json()["link"].split("token=", 1)[1]
    redeemed = client.post(
        f"{API_ROOT}/auth/password/reset/confirm", json={"token": shared_token, "new_password": UNPROVEN_PASSWORD}
    )
    assert redeemed.status_code == 204, redeemed.text
    vouched_cookie = _signed_in_cookie(client, email="ada@example.com", password=UNPROVEN_PASSWORD)
    db_session.expire_all()
    _give_a_passkey_and_a_reset_link(db_session, _identity(db_session, "ada@example.com"))

    signed_in = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert signed_in.status_code == 200, signed_in.text
    assert signed_in.json()["password_removed"] is True
    client.cookies.clear()

    client.cookies.set(SESSION_COOKIE_NAME, vouched_cookie)
    assert client.get(f"{API_ROOT}/organizations/me").status_code == 401
    client.cookies.clear()
    password_sign_in = client.post(
        f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": UNPROVEN_PASSWORD}
    )
    assert password_sign_in.status_code == 401
    db_session.expire_all()
    after = _identity(db_session, "ada@example.com")
    assert after.email_vouched_at is None
    assert after.hashed_password is None
    assert after.password_reset_token_hash is None
    assert _passkeys(db_session, after) == []


def test_a_linked_sign_in_keeps_a_proven_password(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        sign_up(client, email="ada@example.com", password=PASSWORD)
    finally:
        gateway_logger.removeHandler(caplog.handler)
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token.group(1), "password": PASSWORD})
    assert verified.status_code == 200, verified.text
    stub_exchange(monkeypatch)
    for _ in range(2):
        signed_in = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
        assert signed_in.status_code == 200, signed_in.text
        assert signed_in.json()["password_removed"] is False
        client.cookies.clear()

    assert _signed_in_cookie(client, email="ada@example.com", password=PASSWORD)


def test_a_second_provider_sign_in_leaves_the_identity_as_the_first_left_it(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # The owner of the address set this password from the session of the first sign-in.
    add_member(client, master_key_header, email="ada@example.com")
    sign_up(client, email="ada@example.com", password=UNPROVEN_PASSWORD)
    stub_exchange(monkeypatch)
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200
    assert client.put(f"{API_ROOT}/auth/password", json={"new_password": PASSWORD}).status_code == 200
    client.cookies.clear()
    first = _identity(db_session, "ada@example.com")
    left = (first.hashed_password, first.email_verified_at, first.oauth_provider)
    assert left[0] is not None
    db_session.expire_all()

    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": "s"}).status_code == 200
    client.cookies.clear()

    second = _identity(db_session, "ada@example.com")
    assert (second.hashed_password, second.email_verified_at, second.oauth_provider) == left
    password_sign_in = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert password_sign_in.status_code == 200, password_sign_in.text


# ---------- the provider's subject names the account, not the address ----------


def test_a_second_identity_at_one_provider_cannot_sign_in_to_a_linked_account(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, subject="ada-subject")
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200
    client.cookies.clear()
    stub_exchange(monkeypatch, subject="another-subject")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": "s"})

    assert response.status_code == 401, response.text
    assert "signs in with a different Google account" in response.json()["detail"]
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_linked_account_still_signs_in_after_the_provider_changes_its_address(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, subject="ada-subject")
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200
    client.cookies.clear()
    stub_exchange(monkeypatch, subject="ada-subject", email="ada@elsewhere.example.com")

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": "s"})

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == user_id


def _links(db_session: Session, user_id: str) -> list[tuple[str, str]]:
    """The ``(provider, subject)`` pairs linked to an account, in provider order."""
    rows = db_session.execute(
        select(OAuthIdentityLink.provider, OAuthIdentityLink.subject)
        .where(OAuthIdentityLink.user_id == uuid.UUID(user_id))
        .order_by(OAuthIdentityLink.provider)
    )
    return [(provider, subject) for provider, subject in rows.all()]


def _sign_in(client: TestClient, provider: str = "google") -> Any:
    """Finish a stubbed sign-in and drop the session cookie it set."""
    response = client.post(f"{API_ROOT}/auth/oauth/{provider}/callback", json={"code": "c", "state": "s"})
    client.cookies.clear()
    return response


def test_a_first_sign_in_links_the_account_to_the_provider_identity(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, subject="ada-subject")

    assert _sign_in(client).status_code == 200

    assert _links(db_session, user_id) == [("google", "ada-subject")]
    link = db_session.execute(select(OAuthIdentityLink)).scalar_one()
    assert (link.email_at_link, link.email_assertion_at_link) == ("ada@example.com", EmailAssertion.VERIFIED)


def test_one_account_holds_a_link_at_each_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, subject="ada-at-google")
    assert _sign_in(client, "google").status_code == 200
    stub_exchange(monkeypatch, subject="ada-at-github")
    assert _sign_in(client, "github").status_code == 200
    stub_exchange(monkeypatch, subject="ada-at-github", email="ada@elsewhere.example.com")

    response = _sign_in(client, "github")

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == user_id
    assert _links(db_session, user_id) == [("github", "ada-at-github"), ("google", "ada-at-google")]


def test_a_linked_identity_is_refused_when_the_provider_denies_its_address(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    assert _sign_in(client).status_code == 200
    stub_exchange(monkeypatch, email_verified=EmailAssertion.UNVERIFIED)

    response = _sign_in(client)

    assert response.status_code == 401
    assert "did not confirm that address is yours" in response.json()["detail"]


def test_a_linked_identity_signs_in_when_the_provider_says_nothing_about_its_address(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    assert _sign_in(client).status_code == 200
    stub_exchange(monkeypatch, email_verified=EmailAssertion.UNASSERTED)

    response = _sign_in(client)

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == user_id


def test_an_address_the_provider_says_nothing_about_links_no_account(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, email_verified=EmailAssertion.UNASSERTED)

    response = _sign_in(client)

    assert response.status_code == 401
    assert _links(db_session, user_id) == []
    assert _identity(db_session, "ada@example.com").email_verified_at is None


def test_a_deactivated_account_is_refused_through_its_link(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    assert _sign_in(client).status_code == 200
    identity = _identity(db_session, "ada@example.com")
    identity.is_active = False
    db_session.add(identity)
    db_session.commit()

    response = _sign_in(client)

    assert response.status_code == 401
    assert "not registered on this gateway" in response.json()["detail"]


def test_a_refused_second_identity_leaves_the_account_and_its_link_as_they_were(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    sign_up(client, email="ada@example.com", password=UNPROVEN_PASSWORD)
    stub_exchange(monkeypatch, subject="ada-subject")
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"}).status_code == 200
    assert client.put(f"{API_ROOT}/auth/password", json={"new_password": PASSWORD}).status_code == 200
    client.cookies.clear()
    before = _identity(db_session, "ada@example.com").hashed_password
    db_session.expire_all()
    stub_exchange(monkeypatch, subject="another-subject")

    assert _sign_in(client).status_code == 401

    assert _identity(db_session, "ada@example.com").hashed_password == before
    assert _links(db_session, user_id) == [("google", "ada-subject")]


def test_a_refused_provider_sign_in_writes_nothing_to_a_deactivated_identity(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    sign_up(client, email="ada@example.com", password=UNPROVEN_PASSWORD)
    identity = _identity(db_session, "ada@example.com")
    identity.is_active = False
    db_session.add(identity)
    db_session.commit()
    stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})

    assert response.status_code == 401
    assert SESSION_COOKIE_NAME not in response.cookies
    db_session.expire_all()
    refused = _identity(db_session, "ada@example.com")
    assert refused.oauth_provider is None
    assert refused.email_verified_at is None
    assert refused.hashed_password is not None
    assert refused.email_verification_token_hash is not None


def _signed_in_cookie(client: TestClient, *, email: str, password: str) -> str:
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": email, "password": password})
    assert signed_in.status_code == 200, signed_in.text
    cookie = signed_in.cookies[SESSION_COOKIE_NAME]
    client.cookies.clear()
    return cookie


def _give_a_passkey_and_a_reset_link(db_session: Session, identity: User) -> None:
    db_session.add(
        WebAuthnCredential(
            user_id=identity.id,
            name="Laptop",
            credential_id=f"credential-{identity.id}",
            public_key="public-key",
            rp_id="testserver",
        )
    )
    identity.password_reset_token_hash = sha256(b"outstanding reset").hexdigest()
    identity.password_reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
    db_session.add(identity)
    db_session.commit()


def _passkeys(db_session: Session, identity: User) -> list[WebAuthnCredential]:
    db_session.expire_all()
    statement = select(WebAuthnCredential).where(col(WebAuthnCredential.user_id) == identity.id)
    return list(db_session.execute(statement).scalars())


def test_a_provider_sign_in_ends_the_access_a_vouched_password_opened(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # The inviter held the link that chose this password, so whatever it opened
    # cannot outlive the owner proving the mailbox.
    invitation = _invite(client, master_key_header, email="ada@example.com")
    token = invitation["accept_link"].split("token=")[1]
    accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": UNPROVEN_PASSWORD})
    assert accepted.status_code == 200, accepted.text
    vouched_cookie = _signed_in_cookie(client, email="ada@example.com", password=UNPROVEN_PASSWORD)
    identity = _identity(db_session, "ada@example.com")
    _give_a_passkey_and_a_reset_link(db_session, identity)
    stub_exchange(monkeypatch)

    signed_in = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert signed_in.status_code == 200, signed_in.text
    # The person is told, so the password they chose does not just stop working.
    assert signed_in.json()["password_removed"] is True
    client.cookies.clear()

    client.cookies.set(SESSION_COOKIE_NAME, vouched_cookie)
    assert client.get(f"{API_ROOT}/organizations/me").status_code == 401
    assert _passkeys(db_session, identity) == []
    assert _identity(db_session, "ada@example.com").password_reset_token_hash is None


def test_a_provider_sign_in_keeps_the_access_a_proven_password_opened(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    db_session: Session,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        sign_up(client, email="ada@example.com", password=PASSWORD)
    finally:
        gateway_logger.removeHandler(caplog.handler)
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token.group(1), "password": PASSWORD})
    assert verified.status_code == 200, verified.text
    proven_cookie = _signed_in_cookie(client, email="ada@example.com", password=PASSWORD)
    identity = _identity(db_session, "ada@example.com")
    _give_a_passkey_and_a_reset_link(db_session, identity)
    stub_exchange(monkeypatch)

    signed_in = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert signed_in.status_code == 200, signed_in.text
    assert signed_in.json()["password_removed"] is False
    client.cookies.clear()

    client.cookies.set(SESSION_COOKIE_NAME, proven_cookie)
    assert client.get(f"{API_ROOT}/organizations/me").status_code == 200
    assert len(_passkeys(db_session, identity)) == 1


@pytest.mark.parametrize(("provider", "label"), [("google", "Google"), ("github", "GitHub")])
def test_a_refusal_names_the_provider_the_way_it_writes_its_name(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch, provider: str, label: str
) -> None:
    stub_exchange(monkeypatch, email="stranger@example.com")
    unknown = client.post(f"{API_ROOT}/auth/oauth/{provider}/callback", json={"code": "c", "state": "s"})
    stub_exchange(monkeypatch, email_verified=EmailAssertion.UNVERIFIED)
    unverified = client.post(f"{API_ROOT}/auth/oauth/{provider}/callback", json={"code": "c", "state": "s"})

    assert unknown.json()["detail"].startswith(f"That {label} account")
    assert unverified.json()["detail"].startswith(f"{label} did not confirm")


# ---------- the terms a deployment publishes ----------

TERMS_URL = "https://example.com/terms"


@pytest.fixture
def terms_published(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish terms on the deployment TestClient serves."""
    monkeypatch.setattr(test_config, "terms_url", TERMS_URL)


def _callback(client: TestClient, *, terms_accepted: bool | None = None) -> Any:
    """Finish a stubbed Google sign-in, sending ``terms_accepted`` only when it is given."""
    body: dict[str, Any] = {"code": "c", "state": "s"}
    if terms_accepted is not None:
        body["terms_accepted"] = terms_accepted
    return client.post(f"{API_ROOT}/auth/oauth/google/callback", json=body)


def _rows(db_session: Session, model: type[Organization] | type[User] | type[OAuthIdentityLink]) -> int:
    return len(db_session.execute(select(model)).scalars().all())


@pytest.mark.parametrize("terms_accepted", [None, False], ids=["absent", "false"])
def test_open_signup_creates_no_account_through_a_provider_without_the_published_terms(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    terms_published: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    terms_accepted: bool | None,
) -> None:
    """The rule the signup form follows, so a provider is not a way around the terms."""
    stub_exchange(monkeypatch, email="stranger@example.com")
    failures: list[str] = []
    monkeypatch.setattr("gateway.api.routes.auth_oauth.record_auth_failure", failures.append)
    before = {model: _rows(db_session, model) for model in (Organization, User, OAuthIdentityLink)}

    response = _callback(client, terms_accepted=terms_accepted)

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "terms_not_accepted"
    assert response.headers["Otari-Error-Code"] == "terms_not_accepted"
    assert SESSION_COOKIE_NAME not in response.cookies
    # The organization provisioned for the account went with it: no orphan tenancy is left behind.
    db_session.expire_all()
    assert {model: _rows(db_session, model) for model in before} == before
    # The person proved who they are, so the refusal is not counted as a failed sign-in.
    assert failures == []


def test_open_signup_records_the_terms_a_provider_sign_up_accepted(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    terms_published: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = _callback(client, terms_accepted=True)

    assert response.status_code == 200, response.text
    assert response.json()["active_organization_id"]
    assert _identity(db_session, "stranger@example.com").terms_accepted_at is not None


def test_a_refused_provider_sign_up_can_be_finished_once_the_terms_are_accepted(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    terms_published: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    stub_exchange(monkeypatch, email="stranger@example.com")
    organizations = _rows(db_session, Organization)

    assert _callback(client).status_code == 400
    accepted = _callback(client, terms_accepted=True)

    assert accepted.status_code == 200, accepted.text
    # One tenant, from the sign-in that kept it.
    assert _rows(db_session, Organization) == organizations + 1


def test_an_invitee_first_claim_through_a_provider_needs_the_published_terms(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    terms_published: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """The first way in to an account an organization added needs the terms, as an invitation's first password does."""
    _invite(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    refused = _callback(client)

    assert refused.status_code == 400, refused.text
    assert refused.json()["code"] == "terms_not_accepted"
    invitee = _identity(db_session, "ada@example.com")
    assert (invitee.email_verified_at, invitee.oauth_provider, invitee.terms_accepted_at) == (None, None, None)
    assert _links(db_session, str(invitee.id)) == []

    accepted = _callback(client, terms_accepted=True)

    assert accepted.status_code == 200, accepted.text
    db_session.expire_all()
    assert _identity(db_session, "ada@example.com").terms_accepted_at is not None


def test_an_invitation_accepted_without_a_password_still_needs_the_terms_at_the_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    terms_published: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """Accepting without a password records no terms, so the provider sign-in after it is still the first way in."""
    link = _invite(client, master_key_header, email="ada@example.com")["accept_link"]
    accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": link.split("token=")[1]})
    assert accepted.status_code == 200, accepted.text
    stub_exchange(monkeypatch)

    assert _callback(client).status_code == 400
    signed_in = _callback(client, terms_accepted=True)

    assert signed_in.status_code == 200, signed_in.text
    assert signed_in.json()["active_organization_id"]
    assert _identity(db_session, "ada@example.com").terms_accepted_at is not None


def test_a_linked_account_signs_in_without_the_terms_published_after_it_was_created(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """Out of scope here: an account that never accepted terms is not asked to on a later sign-in."""
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)
    assert _sign_in(client).status_code == 200
    monkeypatch.setattr(test_config, "terms_url", TERMS_URL)

    again = _callback(client)

    assert again.status_code == 200, again.text
    assert _identity(db_session, "ada@example.com").terms_accepted_at is None


def test_a_claimed_account_links_a_provider_without_the_terms(
    client: TestClient,
    master_key_header: dict[str, str],
    signup_configured: None,
    terms_published: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    db_session: Session,
) -> None:
    """An account someone already claimed with a password is not a new one, whatever provider it adds."""
    add_member(client, master_key_header, email="ada@example.com")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        signed_up = client.post(
            f"{API_ROOT}/auth/signup",
            json={"email": "ada@example.com", "password": PASSWORD, "terms_accepted": True},
        )
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert signed_up.status_code == 200, signed_up.text
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": token.group(1), "password": PASSWORD})
    assert verified.status_code == 200, verified.text
    accepted_at = _identity(db_session, "ada@example.com").terms_accepted_at
    assert accepted_at is not None
    stub_exchange(monkeypatch)

    linked = _callback(client)

    assert linked.status_code == 200, linked.text
    db_session.expire_all()
    identity = _identity(db_session, "ada@example.com")
    assert identity.oauth_provider == "google"
    assert identity.terms_accepted_at == accepted_at


def test_without_published_terms_a_provider_sign_up_records_no_acceptance(
    client: TestClient,
    oauth_configured: None,
    signup_open: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """There is no document to accept, so a ``terms_accepted`` sent anyway records nothing."""
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = _callback(client, terms_accepted=True)

    assert response.status_code == 200, response.text
    assert _identity(db_session, "stranger@example.com").terms_accepted_at is None


# ---------- the state check, with nothing stubbed but the token endpoint ----------


def stub_token_endpoint(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Replace the outbound HTTP only, and return the form each exchange posted.

    Patched at ``_token_request`` rather than at ``exchange_code``, which is the
    point of these tests: consuming the pending state happens *inside*
    ``exchange_code``, so stubbing that method removes the very check being
    asserted on. ``stub_exchange`` above patches even higher, at the route's own
    reference, so nothing using it can say anything about state at all.

    The returned list holds the form fields sent to the token endpoint, which is
    where ``code_verifier`` shows up.
    """
    posted: list[dict[str, str]] = []

    async def _token_request(self: Any, data: dict[str, str]) -> dict[str, Any]:
        posted.append(data)
        return {"access_token": "an-access-token", "token_type": "Bearer"}

    async def _fetch_identity(self: Any, _tokens: Any) -> Any:
        return SimpleNamespace(subject="ada-subject", email="ada@example.com", name="Ada Lovelace", email_verified=True)

    monkeypatch.setattr(OAuthClient, "_token_request", _token_request)
    monkeypatch.setattr(OAuthClient, "fetch_identity", _fetch_identity)
    return posted


def test_a_code_with_no_authorize_behind_it_is_refused(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finding this table exists for: a leaked code, spent by whoever holds it.

    Before the pending-state row, the callback exchanged any well-formed code it
    was handed, so anybody who obtained a victim's unspent code could post it
    here and be handed the victim's session. There is nothing to bind it to
    unless the deployment kept a record of the flow it started.
    """
    add_member(client, master_key_header, email="ada@example.com")
    posted = stub_token_endpoint(monkeypatch)

    response = client.post(
        f"{API_ROOT}/auth/oauth/google/callback",
        json={"code": "a-leaked-code", "state": "never-minted-here"},
    )

    assert response.status_code == 400, response.text
    assert SESSION_COOKIE_NAME not in response.cookies
    # Refused before the code went anywhere: no token request, and a code that
    # is still the victim's to spend.
    assert posted == []


def test_a_state_is_single_use(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_token_endpoint(monkeypatch)
    state = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()["state"]

    first = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    replayed = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})

    assert first.status_code == 200, first.text
    # The row was deleted as it was claimed, so the replay matches nothing.
    assert replayed.status_code == 400


def test_a_state_minted_for_one_provider_does_not_answer_the_other(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_token_endpoint(monkeypatch)
    state = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()["state"]

    response = client.post(f"{API_ROOT}/auth/oauth/github/callback", json={"code": "c", "state": state})

    assert response.status_code == 400
    # The refusal rolled the claim back with the rest of the request, so the
    # callback the state was minted for still has a flow to finish.
    finished = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    assert finished.status_code == 200, finished.text


def test_the_exchange_sends_the_verifier_the_authorize_call_minted(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PKCE end to end: the challenge on the consent URL answers to the verifier sent here."""
    add_member(client, master_key_header, email="ada@example.com")
    posted = stub_token_endpoint(monkeypatch)
    started = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()
    query = parse_qs(urlsplit(started["authorization_url"]).query)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": started["state"]})

    assert response.status_code == 200, response.text
    (form,) = posted
    verifier = form["code_verifier"]
    assert verifier
    # The challenge the provider was given is the SHA-256 of what the exchange
    # sent, which is what makes a code useless to anyone who did not start this.
    expected = urlsafe_b64encode(sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == [expected]
    assert query["code_challenge_method"] == ["S256"]


def test_a_refused_exchange_leaves_the_state_spendable_for_the_retry(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim is staged on the request's transaction, so a rollback restores it.

    Same rule ``webauthn_service.consume_challenge`` documents: only a flow that
    completes retires a nonce. The code is spent at the provider either way, so
    what survives is a state the person's own retry can use, not a replayable
    one.
    """
    add_member(client, master_key_header, email="ada@example.com")
    state = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()["state"]

    async def _boom(self: Any, _data: dict[str, str]) -> Any:
        raise RuntimeError("the provider was unreachable")

    monkeypatch.setattr(OAuthClient, "_token_request", _boom)
    failed = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})

    # An unreachable provider is an upstream failure, not the caller's.
    assert failed.status_code == 502
    stub_token_endpoint(monkeypatch)
    retried = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c2", "state": state})

    assert retried.status_code == 200, retried.text


def test_authorize_sets_the_flow_cookie_the_callback_requires(
    client: TestClient,
    oauth_configured: None,
) -> None:
    started = client.get(f"{API_ROOT}/auth/oauth/google/authorize")

    assert started.status_code == 200, started.text
    cookie = started.headers["set-cookie"]
    assert cookie.startswith(f"{FLOW_COOKIE_NAME}=")
    assert "HttpOnly" in cookie
    assert f"Path={FLOW_COOKIE_PATH}" in cookie
    assert "samesite=lax" in cookie.lower()


def test_a_callback_from_a_browser_that_did_not_start_the_flow_is_refused(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's vector: the redirect URL, code and state both, read out of a log.

    Holding the whole redirect query is not enough, because the flow cookie
    never left the browser that called ``/authorize``. Without it the callback
    is refused before any outbound call; with a different browser's cookie it is
    refused too.
    """
    add_member(client, master_key_header, email="ada@example.com")
    posted = stub_token_endpoint(monkeypatch)
    state = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()["state"]
    victims_cookie = client.cookies[FLOW_COOKIE_NAME]

    client.cookies.clear()
    without = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    assert without.status_code == 400, without.text
    assert posted == []

    client.get(f"{API_ROOT}/auth/oauth/google/authorize")  # a different browser's own cookie
    assert client.cookies[FLOW_COOKIE_NAME] != victims_cookie
    other = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    assert other.status_code == 400, other.text
    assert posted == []

    # The refusals rolled the claim back, so the browser that started it can still finish.
    client.cookies.set(FLOW_COOKIE_NAME, victims_cookie)
    finished = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    assert finished.status_code == 200, finished.text


def test_two_tabs_in_one_browser_share_the_cookie_and_both_finish(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_token_endpoint(monkeypatch)
    first = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()["state"]
    cookie = client.cookies[FLOW_COOKIE_NAME]
    second = client.get(f"{API_ROOT}/auth/oauth/github/authorize").json()["state"]

    # The second call reused the cookie instead of rotating it out from under the first tab.
    assert client.cookies[FLOW_COOKIE_NAME] == cookie
    assert client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c1", "state": first}).status_code == 200
    assert (
        client.post(f"{API_ROOT}/auth/oauth/github/callback", json={"code": "c2", "state": second}).status_code == 200
    )


def test_the_refusal_does_not_say_which_way_the_state_was_wrong(
    client: TestClient,
    oauth_configured: None,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Never minted and already spent answer identically: telling them apart
    # tells an attacker their guess was in the right shape.
    #
    # The sign-in that spends the state has to succeed, which is why this puts
    # an address on the roster first: a refused identity rolls the whole
    # transaction back, and the state it claimed with it.
    add_member(client, master_key_header, email="ada@example.com")
    stub_token_endpoint(monkeypatch)
    state = client.get(f"{API_ROOT}/auth/oauth/google/authorize").json()["state"]
    signed_in = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    assert signed_in.status_code == 200, signed_in.text

    spent = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})
    unknown = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "nope"})
    client.cookies.clear()
    no_cookie = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": state})

    assert spent.status_code == unknown.status_code == no_cookie.status_code
    assert spent.json()["detail"] == unknown.json()["detail"] == no_cookie.json()["detail"]


# ---------- the redirect a provider actually lands on ----------


@pytest.mark.parametrize("provider", ["google", "github"])
def test_the_provider_redirect_path_bounces_into_the_dashboard_hash_route(client: TestClient, provider: str) -> None:
    # A redirect URI may not carry a fragment, so the provider is pointed at an
    # ordinary path and this is what turns it into the hash route the dashboard
    # renders ahead of its auth gate.
    response = client.get(f"/auth/{provider}/callback?code=the-code&state=the-state", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == (f"/#/auth/{provider}/callback?code=the-code&state=the-state")


def test_the_provider_redirect_path_carries_an_error_query_through_too(
    client: TestClient,
) -> None:
    response = client.get("/auth/google/callback?error=access_denied&state=s", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/#/auth/google/callback?error=access_denied&state=s"


def test_the_provider_redirect_path_works_with_no_query_at_all(client: TestClient) -> None:
    response = client.get("/auth/google/callback", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/#/auth/google/callback"


def test_the_provider_redirect_path_needs_no_credential(client: TestClient) -> None:
    # A person is here because a provider sent them, holding nothing.
    assert client.get("/auth/google/callback", follow_redirects=False).status_code == 303
