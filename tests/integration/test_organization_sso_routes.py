"""The organization SSO settings surface: role gate, redirect URI, and the domain list.

The service layer (`test_oidc_connection_service.py`) already covers discovery,
encryption, and the conflict/race paths. This file covers what only the HTTP surface
can: the status codes, and the response shape an administrator actually reads,
including the `redirect_uri` the route adds over the service's own schema.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.identity import (
    EmailAssertion,
    OAuthIdentityLink,
    OIDCConnection,
    ProviderIdentity,
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
from gateway.services.secret_box import encrypt_secret, generate_secret_key

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


def _identity(session: Session, *, email: str, organization_id: uuid.UUID, role: str) -> str:
    """Create a member of ``organization_id`` with a live dashboard session, and return its cookie token."""
    user = User(email=email, full_name=email.split("@")[0].title(), active_organization_id=organization_id)
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


def _member(session: Session, *, organization_id: uuid.UUID, role: str = "member") -> User:
    """Create an active member of ``organization_id`` with no dashboard session, and return the account."""
    user = User(email=None, full_name="SSO Member", active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role))
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


def _connection(organization_id: uuid.UUID) -> OIDCConnection:
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
    # No connection is saved yet, and the redirect URI is still present: it is
    # what an administrator registers at the provider before the first save.
    assert body["connection"] is None
    assert body["redirect_uri"] == "/auth/sso/callback"


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


def test_unlinking_removes_no_other_provider_link(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-other-provider")
        owner_token = _identity(
            session, email="owner@acme-other-provider.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id))
        session.commit()
        member = _member(session, organization_id=organization.id)
        member_id = member.id
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="sso-subject", user_id=member_id)
        _link(session, provider=_OTHER_PROVIDER, subject="other-subject", user_id=member_id)
    finally:
        session.close()

    response = _call(client, owner_token, "DELETE", f"{_ENDPOINT}/links/{member_id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT, response.text

    session = db_session_factory()
    try:
        assert session.get(OAuthIdentityLink, (oidc_identity_provider(_ISSUER), "sso-subject")) is None
        assert session.get(OAuthIdentityLink, (_OTHER_PROVIDER, "other-subject")) is not None
    finally:
        session.close()


def test_unlinking_deletes_no_account_and_moves_no_membership(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-keeps-account")
        owner_token = _identity(
            session, email="owner@acme-keeps-account.test", organization_id=organization.id, role="owner"
        )
        session.add(_connection(organization.id))
        session.commit()
        member = _member(session, organization_id=organization.id)
        member_id = member.id
        organization_id = organization.id
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="sso-subject", user_id=member_id)
    finally:
        session.close()

    response = _call(client, owner_token, "DELETE", f"{_ENDPOINT}/links/{member_id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT, response.text

    session = db_session_factory()
    try:
        assert session.get(User, member_id) is not None
        membership = session.scalars(
            select(OrganizationMember).where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.user_id) == member_id,
            )
        ).one()
        assert membership.role == "member"
        assert membership.status == "active"
    finally:
        session.close()


def test_an_admin_of_another_organization_cannot_unlink(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    """Two organizations share one issuer, so the link the admin reaches for belongs to the other one."""
    session = db_session_factory()
    try:
        own_organization = _organization(session, slug="acme-own")
        admin_token = _identity(session, email="admin@acme-own.test", organization_id=own_organization.id, role="admin")
        session.add(_connection(own_organization.id))
        other_organization = _organization(session, slug="acme-other")
        session.add(_connection(other_organization.id))
        session.commit()
        foreign_member = _member(session, organization_id=other_organization.id)
        foreign_member_id = foreign_member.id
        _link(
            session,
            provider=oidc_identity_provider(_ISSUER),
            subject="shared-issuer-subject",
            user_id=foreign_member_id,
        )
    finally:
        session.close()

    response = _call(client, admin_token, "DELETE", f"{_ENDPOINT}/links/{foreign_member_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text

    session = db_session_factory()
    try:
        assert session.get(OAuthIdentityLink, (oidc_identity_provider(_ISSUER), "shared-issuer-subject")) is not None
    finally:
        session.close()


def test_unlinking_lets_the_member_link_afresh_on_their_next_sign_in(
    client: TestClient,
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """Nothing here reuses the unlinked account: an SSO sign-in with no link provisions a fresh one."""
    # A real authorization request needs an absolute, https redirect_uri, which only an
    # https public_base_url produces.
    monkeypatch.setattr(test_config, "public_base_url", "https://testserver")
    session = db_session_factory()
    try:
        organization = _organization(session, slug="acme-relink")
        owner_token = _identity(session, email="owner@acme-relink.test", organization_id=organization.id, role="owner")
        connection = _connection(organization.id)
        # The real authorize step builds the full OIDC client before the stubbed exchange below
        # ever runs, and that client refuses metadata with no signing key set.
        connection.discovery_document = {**connection.discovery_document, "jwks_url": f"{_ISSUER}/jwks"}
        session.add(connection)
        session.add(
            OrganizationDomain(
                organization_id=organization.id,
                domain="acme-relink.test",
                verification_token="a-verification-token",
                verified_at=datetime.now(UTC),
                enabled=True,
            )
        )
        session.commit()
        member = _member(session, organization_id=organization.id)
        member_id = member.id
        organization_slug = organization.slug
        _link(session, provider=oidc_identity_provider(_ISSUER), subject="relinking-subject", user_id=member_id)
    finally:
        session.close()

    removed = _call(client, owner_token, "DELETE", f"{_ENDPOINT}/links/{member_id}")
    assert removed.status_code == status.HTTP_204_NO_CONTENT, removed.text

    async def _exchange(_client: Any, *, code: str, **_kwargs: Any) -> ProviderIdentity:
        return ProviderIdentity(
            provider=oidc_identity_provider(_ISSUER),
            subject="relinking-subject",
            email="returning@acme-relink.test",
            full_name="Returning Member",
            email_verified=EmailAssertion.UNASSERTED,
        )

    monkeypatch.setattr("gateway.api.routes.auth_sso._exchange_sso_code", _exchange)

    authorized = client.get(f"{API_ROOT}/auth/sso/{organization_slug}/authorize")
    assert authorized.status_code == status.HTTP_200_OK, authorized.text
    bound = authorized.json()

    callback = client.post(f"{API_ROOT}/auth/sso/callback", json={"code": "a-fresh-code", "state": bound["state"]})

    assert callback.status_code == status.HTTP_200_OK, callback.text
    new_user_id = callback.json()["user_id"]
    assert new_user_id != str(member_id)

    session = db_session_factory()
    try:
        relinked = session.get(OAuthIdentityLink, (oidc_identity_provider(_ISSUER), "relinking-subject"))
        assert relinked is not None
        assert str(relinked.user_id) == new_user_id
    finally:
        session.close()
