"""The organization SSO settings surface: role gate, redirect URI, and the domain list.

The service layer (`test_oidc_connection_service.py`) already covers discovery,
encryption, and the conflict/race paths. This file covers what only the HTTP surface
can: the status codes, and the response shape an administrator actually reads,
including the `redirect_uri` the route adds over the service's own schema.
"""

import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col, select

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.identity import (
    EmailAssertion,
    OAuthIdentityLink,
    OIDCConnection,
    oidc_identity_provider,
)
from gateway.models.tenancy import (
    DOMAIN_PROOF_TTL,
    DashboardSession,
    Organization,
    OrganizationDomain,
    OrganizationMember,
    User,
)
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.password_service import hash_password
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client

_ENDPOINT = f"{API_ROOT}/organizations/me/sso"
_ISSUER = "https://login.example.com"
_OTHER_PROVIDER = "oidc:https://other-idp.example"
_SECRET = "s3cr3t-client-value"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving or testing a connection encrypts the client secret, which needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _organization(session: Session, *, slug: str) -> Organization:
    organization = Organization(name=slug.title(), slug=slug)
    session.add(organization)
    session.commit()
    session.refresh(organization)
    return organization


def _identity(session: Session, *, email: str, organization_id: uuid.UUID, role: str, password: bool = False) -> str:
    """Create a member of ``organization_id`` with a live dashboard session, and return its cookie token.

    ``password`` gives them a route in of their own, which an organization's break-glass owner needs.
    """
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
        hashed_password=hash_password("correct-horse-battery-staple") if password else None,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role))
    token = f"otari-sess-{email}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return token


def _member(
    session: Session,
    *,
    organization_id: uuid.UUID,
    role: str = "member",
    status: str = "active",
    email: str | None = None,
) -> User:
    """Create a member of ``organization_id`` with no dashboard session, and return the account."""
    full_name = "SSO Member" if email is None else email
    user = User(email=email, full_name=full_name, active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status=status))
    session.commit()
    return user


def _link(session: Session, *, provider: str, subject: str, user_id: uuid.UUID) -> OAuthIdentityLink:
    link = OAuthIdentityLink(
        provider=provider,
        subject=subject,
        user_id=user_id,
        email_at_link=None,
        email_assertion_at_link=EmailAssertion.UNASSERTED,
    )
    session.add(link)
    session.commit()
    return link


def _connection(organization_id: uuid.UUID, *, first_sign_in_at: datetime | None = None) -> OIDCConnection:
    return OIDCConnection(
        organization_id=organization_id,
        issuer=_ISSUER,
        resolved_issuer=_ISSUER,
        client_id="a-client-id",
        encrypted_client_secret=encrypt_secret(_SECRET),
        client_secret_last4=_SECRET[-4:],
        display_name="Example IdP",
        discovery_document={
            "issuer": _ISSUER,
            "authorize_url": f"{_ISSUER}/authorize",
            "token_url": f"{_ISSUER}/token",
        },
        discovered_at=datetime.now(UTC),
        default_role="member",
        enabled=True,
        first_sign_in_at=first_sign_in_at,
    )


def _connection_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "issuer": _ISSUER,
        "client_id": "a-client-id",
        "client_secret": _SECRET,
        "display_name": "Example IdP",
    }
    body.update(overrides)
    return body


def _stub_discovery(monkeypatch: pytest.MonkeyPatch, *, calls: list[str] | None = None) -> None:
    async def _discover(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        if calls is not None:
            calls.append(issuer)
        return DiscoveredProvider(
            resolved_issuer=_ISSUER,
            document={"issuer": _ISSUER, "authorize_url": f"{_ISSUER}/authorize", "token_url": f"{_ISSUER}/token"},
            discovered_at=datetime.now(UTC),
        )

    monkeypatch.setattr("gateway.services.identity._connection_service.discover_connection_metadata", _discover)


def _call(client: TestClient, token: str, method: str, path: str, json: dict[str, Any] | None = None) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        return client.request(method, path, json=json)
    finally:
        client.cookies.clear()


def test_get_returns_last4_and_never_the_secret(client: TestClient, db_session_factory: Callable[[], Session]) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-get")
        owner_token = _identity(session, email="owner@acme-get.test", organization_id=organization.id, role="owner")
        session.add(_connection(organization.id))
        session.commit()
    finally:
        session.close()

    response = _call(client, owner_token, "GET", _ENDPOINT)
    assert response.status_code == status.HTTP_200_OK, response.text

    body = response.json()
    assert body["connection"]["client_secret_last4"] == _SECRET[-4:]
    assert "client_secret" not in body["connection"]
    assert _SECRET not in response.text


def test_put_refuses_a_caller_who_is_only_a_member(
    client: TestClient, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _stub_discovery(monkeypatch, calls=calls)
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-member")
        member_token = _identity(
            session, email="member@acme-member.test", organization_id=organization.id, role="member"
        )
    finally:
        session.close()

    response = _call(client, member_token, "PUT", _ENDPOINT, json=_connection_body())

    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text
    # A plain member's request is refused before discovery ever dials out.
    assert calls == []


def test_the_response_names_the_redirect_uri_to_register(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-redirect")
        owner_token = _identity(
            session, email="owner@acme-redirect.test", organization_id=organization.id, role="owner"
        )
    finally:
        session.close()

    response = _call(client, owner_token, "GET", _ENDPOINT)

    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    # This client configures no public address, so there is no absolute address to
    # register. A path is not registerable at any provider, so the field is null
    # rather than carrying one.
    assert body["connection"] is None
    assert body["redirect_uri"] is None


@pytest.fixture
def addressed_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    """A client whose deployment has a public address, which is where the redirect URI comes from."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        public_base_url="https://otari.example",
    )
    yield from build_test_client(config)


def test_the_redirect_uri_is_the_deployments_own_address_where_it_has_one(
    addressed_client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    """Asserted on a client of its own, because the address comes from config rather than a row."""
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-addressed")
        owner_token = _identity(
            session, email="owner@acme-addressed.test", organization_id=organization.id, role="owner"
        )
    finally:
        session.close()

    response = _call(addressed_client, owner_token, "GET", _ENDPOINT)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["redirect_uri"] == "https://otari.example/auth/sso/callback"


def test_the_response_lists_each_verified_domain_and_its_expiry(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-domains")
        owner_token = _identity(session, email="owner@acme-domains.test", organization_id=organization.id, role="owner")
        verified_at = datetime.now(UTC) - timedelta(days=1)
        session.add(
            OrganizationDomain(
                organization_id=organization.id,
                domain="acme-domains.test",
                verification_token="a-verification-token",
                verified_at=verified_at,
                enabled=True,
            )
        )
        session.add(
            OrganizationDomain(
                organization_id=organization.id,
                domain="unproven.acme-domains.test",
                verification_token="b-verification-token",
                verified_at=None,
                enabled=True,
            )
        )
        session.add(_connection(organization.id))
        session.commit()
    finally:
        session.close()

    response = _call(client, owner_token, "GET", _ENDPOINT)

    assert response.status_code == status.HTTP_200_OK, response.text
    domains = {entry["domain"]: entry for entry in response.json()["connection"]["domains"]}
    assert set(domains) == {"acme-domains.test", "unproven.acme-domains.test"}

    proven = domains["acme-domains.test"]
    assert proven["enabled"] is True
    proof_expires_at = datetime.fromisoformat(proven["proof_expires_at"])
    assert abs(proof_expires_at - (verified_at + DOMAIN_PROOF_TTL)) < timedelta(seconds=1)

    assert domains["unproven.acme-domains.test"]["proof_expires_at"] is None


def test_the_response_reports_enforcement_as_off_by_default(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforcement-default")
        owner_token = _identity(
            session, email="owner@acme-enforcement-default.test", organization_id=organization.id, role="owner"
        )
    finally:
        session.close()

    response = _call(client, owner_token, "GET", _ENDPOINT)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["enforcement"] == {
        "required": False,
        "break_glass_user_id": None,
        "break_glass_waived": False,
    }


def test_the_response_reflects_enforcement_once_it_is_set(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforcement-reflects")
        owner_email = "owner@acme-enforcement-reflects.test"
        owner_token = _identity(
            session, email=owner_email, organization_id=organization.id, role="owner", password=True
        )
        session.add(_connection(organization.id, first_sign_in_at=datetime.now(UTC)))
        session.commit()
        # Naming the acting owner as the break-glass exception, rather than waiving it, so their
        # own session survives the enforcement this call turns on and the follow-up read can use it.
        owner_id = _owner_id(session, email=owner_email)
    finally:
        session.close()

    put_response = _call(
        client,
        owner_token,
        "PUT",
        f"{_ENDPOINT}/enforcement",
        json=_enforcement_body(break_glass_user_id=str(owner_id), break_glass_waived=False),
    )
    assert put_response.status_code == status.HTTP_204_NO_CONTENT, put_response.text

    get_response = _call(client, owner_token, "GET", _ENDPOINT)
    assert get_response.status_code == status.HTTP_200_OK, get_response.text
    assert get_response.json()["enforcement"] == {
        "required": True,
        "break_glass_user_id": str(owner_id),
        "break_glass_waived": False,
    }


def test_test_and_delete_refuse_404_with_no_connection_saved(
    client: TestClient, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_discovery(monkeypatch)
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-unconfigured")
        owner_token = _identity(
            session, email="owner@acme-unconfigured.test", organization_id=organization.id, role="owner"
        )
    finally:
        session.close()

    assert _call(client, owner_token, "POST", f"{_ENDPOINT}/test").status_code == status.HTTP_404_NOT_FOUND
    assert _call(client, owner_token, "DELETE", _ENDPOINT).status_code == status.HTTP_404_NOT_FOUND


def test_put_then_test_then_delete_round_trip(
    client: TestClient, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_discovery(monkeypatch)
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-roundtrip")
        owner_token = _identity(
            session, email="owner@acme-roundtrip.test", organization_id=organization.id, role="owner"
        )
    finally:
        session.close()

    saved = _call(client, owner_token, "PUT", _ENDPOINT, json=_connection_body())
    assert saved.status_code == status.HTTP_200_OK, saved.text
    assert saved.json()["connection"]["display_name"] == "Example IdP"

    tested = _call(client, owner_token, "POST", f"{_ENDPOINT}/test")
    assert tested.status_code == status.HTTP_200_OK, tested.text

    deleted = _call(client, owner_token, "DELETE", _ENDPOINT)
    assert deleted.status_code == status.HTTP_204_NO_CONTENT, deleted.text

    gone = _call(client, owner_token, "GET", _ENDPOINT)
    assert gone.status_code == status.HTTP_200_OK, gone.text
    assert gone.json()["connection"] is None


def _enforcement_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"required": True, "break_glass_user_id": None, "break_glass_waived": True}
    body.update(overrides)
    return body


def test_enforcement_succeeds_with_a_proven_connection_and_a_waiver(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-on")
        owner_token = _identity(
            session, email="owner@acme-enforce-on.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id, first_sign_in_at=datetime.now(UTC)))
        session.commit()
        organization_id = organization.id
    finally:
        session.close()

    response = _call(client, owner_token, "PUT", f"{_ENDPOINT}/enforcement", json=_enforcement_body())
    assert response.status_code == status.HTTP_204_NO_CONTENT, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is True
        assert stored.sso_break_glass_waived is True
    finally:
        session.close()


def test_enforcement_refuses_404_with_no_connection_saved(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-unconfigured")
        owner_token = _identity(
            session, email="owner@acme-enforce-unconfigured.test", organization_id=organization.id, role="owner"
        )
        organization_id = organization.id
    finally:
        session.close()

    response = _call(client, owner_token, "PUT", f"{_ENDPOINT}/enforcement", json=_enforcement_body())
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is False
    finally:
        session.close()


def test_enforcement_refuses_before_a_sign_in_is_proven(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-unproven")
        owner_token = _identity(
            session, email="owner@acme-enforce-unproven.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id))
        session.commit()
        organization_id = organization.id
    finally:
        session.close()

    response = _call(
        client, owner_token, "PUT", f"{_ENDPOINT}/enforcement", json=_enforcement_body(break_glass_waived=True)
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is False
    finally:
        session.close()


def test_enforcement_refuses_with_no_break_glass_owner_and_no_waiver(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-no-waiver")
        owner_token = _identity(
            session, email="owner@acme-enforce-no-waiver.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id, first_sign_in_at=datetime.now(UTC)))
        session.commit()
        organization_id = organization.id
    finally:
        session.close()

    response = _call(
        client, owner_token, "PUT", f"{_ENDPOINT}/enforcement", json=_enforcement_body(break_glass_waived=False)
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is False
    finally:
        session.close()


def test_enforcement_refuses_a_break_glass_account_that_is_not_an_owner(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-not-owner")
        owner_token = _identity(
            session, email="owner@acme-enforce-not-owner.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id, first_sign_in_at=datetime.now(UTC)))
        session.commit()
        member = _member(session, organization_id=organization.id)
        member_id = member.id
        organization_id = organization.id
    finally:
        session.close()

    response = _call(
        client,
        owner_token,
        "PUT",
        f"{_ENDPOINT}/enforcement",
        json=_enforcement_body(break_glass_user_id=str(member_id), break_glass_waived=False),
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is False
    finally:
        session.close()


def test_enforcement_refuses_a_caller_who_is_only_a_member(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-member")
        member_token = _identity(
            session, email="member@acme-enforce-member.test", organization_id=organization.id, role="member"
        )
        session.add(_connection(organization.id, first_sign_in_at=datetime.now(UTC)))
        session.commit()
        organization_id = organization.id
    finally:
        session.close()

    response = _call(client, member_token, "PUT", f"{_ENDPOINT}/enforcement", json=_enforcement_body())
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is False
    finally:
        session.close()


def test_naming_an_account_that_is_not_an_owner_is_refused_with_enforcement_off(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    """Refused rather than stored: the column is a foreign key, so an unknown id reached the database."""
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-off-stranger")
        owner_token = _identity(
            session, email="owner@acme-off-stranger.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id))
        session.commit()
    finally:
        session.close()

    response = _call(
        client,
        owner_token,
        "PUT",
        f"{_ENDPOINT}/enforcement",
        json=_enforcement_body(required=False, break_glass_user_id=str(uuid.uuid4()), break_glass_waived=False),
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text


def test_enforcement_turns_off_with_no_precondition(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-enforce-off")
        owner_token = _identity(
            session, email="owner@acme-enforce-off.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id))
        session.commit()
        organization_id = organization.id
    finally:
        session.close()

    response = _call(
        client,
        owner_token,
        "PUT",
        f"{_ENDPOINT}/enforcement",
        json=_enforcement_body(required=False, break_glass_user_id=None, break_glass_waived=False),
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT, response.text

    session = db_session_factory()
    try:
        stored = session.get(Organization, organization_id)
        assert stored is not None
        assert stored.sso_required is False
    finally:
        session.close()


def _owner_id(session: Session, *, email: str) -> uuid.UUID:
    return session.scalars(select(User).where(col(User.email) == email)).one().id


def test_preflight_refuses_404_with_no_connection_saved(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-unconfigured")
        owner_token = _identity(
            session, email="owner@acme-preflight-unconfigured.test", organization_id=organization.id, role="owner"
        )
    finally:
        session.close()

    response = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text


def test_preflight_refuses_a_caller_who_is_only_a_member(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-member")
        member_token = _identity(
            session, email="member@acme-preflight-member.test", organization_id=organization.id, role="member"
        )
        session.add(_connection(organization.id))
        session.commit()
    finally:
        session.close()

    response = _call(client, member_token, "GET", f"{_ENDPOINT}/preflight")
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text


def test_the_count_excludes_a_member_already_linked_to_this_connection(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-linked")
        owner_token = _identity(
            session, email="owner@acme-preflight-linked.test", organization_id=organization.id, role="owner"
        )
        owner_id = _owner_id(session, email="owner@acme-preflight-linked.test")
        session.add(_connection(organization.id))
        session.commit()
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="owner-subject", user_id=owner_id)
        linked = _member(session, organization_id=organization.id, email="linked@acme-preflight-linked.test")
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="linked-subject", user_id=linked.id)
        _member(session, organization_id=organization.id, email="unlinked@acme-preflight-linked.test")
    finally:
        session.close()

    response = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight")
    assert response.status_code == status.HTTP_200_OK, response.text

    body = response.json()
    assert body["total_members"] == 3
    assert body["affected_count"] == 1
    assert [entry["email"] for entry in body["affected_members"]] == ["unlinked@acme-preflight-linked.test"]
    assert body["affected_members_truncated"] is False


def test_the_count_excludes_the_named_break_glass_owner(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    """The named exception keeps acting, so counting them as losing access overstates the cost."""
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-exempt")
        owner_token = _identity(
            session, email="owner@acme-preflight-exempt.test", organization_id=organization.id, role="owner"
        )
        owner_id = _owner_id(session, email="owner@acme-preflight-exempt.test")
        organization.sso_break_glass_user_id = owner_id
        session.add(organization)
        session.add(_connection(organization.id))
        _member(session, organization_id=organization.id, email="unlinked@acme-preflight-exempt.test")
        session.commit()
    finally:
        session.close()

    response = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight")
    assert response.status_code == status.HTTP_200_OK, response.text

    body = response.json()
    assert body["total_members"] == 2
    assert body["affected_count"] == 1
    assert [entry["email"] for entry in body["affected_members"]] == ["unlinked@acme-preflight-exempt.test"]


def test_the_count_excludes_a_suspended_member(client: TestClient, db_session_factory: Callable[[], Session]) -> None:
    """A suspended member is not an active membership, so the denominator must not move for one."""
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-suspended")
        owner_token = _identity(
            session, email="owner@acme-preflight-suspended.test", organization_id=organization.id, role="owner"
        )
        owner_id = _owner_id(session, email="owner@acme-preflight-suspended.test")
        session.add(_connection(organization.id))
        session.commit()
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="owner-subject", user_id=owner_id)
        organization_id = organization.id
    finally:
        session.close()

    before = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight").json()

    session = db_session_factory()
    try:
        _member(session, organization_id=organization_id, status="suspended")
    finally:
        session.close()

    after = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight").json()

    assert after["total_members"] == before["total_members"]
    assert after["affected_count"] == before["affected_count"]


def test_the_count_excludes_an_invitee_who_has_not_accepted(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    """An invitation nobody has accepted is not an active membership, so the denominator must not move for one."""
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-invited")
        owner_token = _identity(
            session, email="owner@acme-preflight-invited.test", organization_id=organization.id, role="owner"
        )
        owner_id = _owner_id(session, email="owner@acme-preflight-invited.test")
        session.add(_connection(organization.id))
        session.commit()
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="owner-subject", user_id=owner_id)
        organization_id = organization.id
    finally:
        session.close()

    before = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight").json()

    session = db_session_factory()
    try:
        _member(session, organization_id=organization_id, status="invited")
    finally:
        session.close()

    after = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight").json()

    assert after["total_members"] == before["total_members"]
    assert after["affected_count"] == before["affected_count"]


def test_an_empty_list_is_answered_as_an_empty_list(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-empty")
        owner_token = _identity(
            session, email="owner@acme-preflight-empty.test", organization_id=organization.id, role="owner"
        )
        owner_id = _owner_id(session, email="owner@acme-preflight-empty.test")
        session.add(_connection(organization.id))
        session.commit()
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="owner-subject", user_id=owner_id)
    finally:
        session.close()

    response = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight")
    assert response.status_code == status.HTTP_200_OK, response.text

    body = response.json()
    assert body["affected_count"] == 0
    assert body["affected_members"] == []
    assert body["affected_members_truncated"] is False


def test_the_list_is_capped_while_the_count_stays_exact_past_the_ceiling(
    client: TestClient, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches both a list that ignores the ceiling and a count derived from the truncated list."""
    monkeypatch.setattr("gateway.services.identity._connection_service._PREFLIGHT_SAMPLE_LIMIT", 2)
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-preflight-ceiling")
        owner_token = _identity(
            session, email="owner@acme-preflight-ceiling.test", organization_id=organization.id, role="owner"
        )
        owner_id = _owner_id(session, email="owner@acme-preflight-ceiling.test")
        session.add(_connection(organization.id))
        session.commit()
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="owner-subject", user_id=owner_id)
        for index in range(3):
            _member(session, organization_id=organization.id, email=f"unlinked-{index}@acme-preflight-ceiling.test")
    finally:
        session.close()

    response = _call(client, owner_token, "GET", f"{_ENDPOINT}/preflight")
    assert response.status_code == status.HTTP_200_OK, response.text

    body = response.json()
    assert body["affected_count"] == 3
    assert len(body["affected_members"]) == 2
    assert body["affected_members_truncated"] is True
