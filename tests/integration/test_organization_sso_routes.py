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
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.identity import OIDCConnection
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
