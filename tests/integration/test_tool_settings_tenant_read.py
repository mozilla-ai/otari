"""``GET /v1/tool-settings`` answers a tenant, without the service endpoints in it.

The roles matrix has the Tools pages at View for a member, and mozilla-ai/otari#867
deferred this one read because the fields are deployment infrastructure rather
than anything tenant-scoped (otari-ai#1969). They still are: the decision here is
that a tenant reads *what the built-in tools do to their requests* and never
*where those services live*.

So the three URL fields are withheld from a non-operator rather than masked.
Masking already hides a password, but the host is what the Settings page's
network-safety gates are set against, and it is not a tenant's to see. The write
verbs stay operator-only and have their own tests next door in
``tests/unit/test_tool_settings_endpoint.py``.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import DashboardSession
from gateway.models.tenancy import Organization, OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

_PATH = "/v1/tool-settings"
_URL_KEYS = {"web_search_url", "sandbox_url", "guardrails_url"}


def _identity(session: Session, *, email: str, organization_id: uuid.UUID, is_superuser: bool = False) -> str:
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
        is_superuser=is_superuser,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role="member", status="active"))
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


@pytest.fixture
def sessions(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> dict[str, str]:
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK
    session = db_session_factory()
    try:
        organization = Organization(name="Alpha", slug="alpha")
        session.add(organization)
        session.commit()
        session.refresh(organization)
        return {
            "member": _identity(session, email="member@alpha.test", organization_id=organization.id),
            "operator": _identity(
                session, email="root@alpha.test", organization_id=organization.id, is_superuser=True
            ),
        }
    finally:
        session.close()


def _read_as(client: TestClient, token: str) -> dict[str, Any]:
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        response = client.get(_PATH)
        assert response.status_code == status.HTTP_200_OK, response.text
        return {field["key"]: field for field in response.json()["fields"]}
    finally:
        client.cookies.clear()


def test_a_member_reads_the_settings_without_the_service_endpoints(
    client: TestClient, sessions: dict[str, str]
) -> None:
    fields = _read_as(client, sessions["member"])
    assert fields, "a member gets the read, not an empty body"
    assert _URL_KEYS.isdisjoint(fields)
    # The fields that describe what a request gets are the point of opening it.
    assert "web_search_max_results" in fields
    assert "web_search_intercept" in fields


def test_an_operator_session_still_reads_the_endpoints(client: TestClient, sessions: dict[str, str]) -> None:
    assert _URL_KEYS.issubset(_read_as(client, sessions["operator"]))


def test_the_master_key_still_reads_everything(
    client: TestClient, master_key_header: dict[str, str], sessions: dict[str, str]
) -> None:
    """The control: the deployment credential is not a tenant and is not narrowed."""
    response = client.get(_PATH, headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    assert _URL_KEYS.issubset({field["key"] for field in response.json()["fields"]})


def test_the_write_verbs_still_refuse_a_member(client: TestClient, sessions: dict[str, str]) -> None:
    """Opening the read must not have opened the router."""
    client.cookies.set(SESSION_COOKIE_NAME, sessions["member"])
    try:
        patched = client.patch(_PATH, json={"web_search_max_results": 3})
        assert patched.status_code == status.HTTP_403_FORBIDDEN, patched.text
        probed = client.post(f"{_PATH}/web_search/test", json={"url": "http://searxng:8080"})
        assert probed.status_code == status.HTTP_403_FORBIDDEN, probed.text
    finally:
        client.cookies.clear()


def test_an_unauthenticated_read_is_still_refused(client: TestClient) -> None:
    assert client.get(_PATH).status_code == status.HTTP_401_UNAUTHORIZED
