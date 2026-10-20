"""``DELETE /api/v1/organizations/me``: who may delete an organization, and when it is refused."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.tenancy.provisioning_service import DEFAULT_ORGANIZATION_NAME

_ME = f"{API_ROOT}/organizations/me"


def _create_and_enter(client: TestClient, headers: dict[str, str], name: str = "Research") -> dict[str, Any]:
    created = client.post(f"{API_ROOT}/organizations", json={"name": name}, headers=headers)
    assert created.status_code == 201, created.text
    switched = client.post(f"{_ME}/switch", json={"organization_id": created.json()["id"]}, headers=headers)
    assert switched.status_code == 200, switched.text
    body: dict[str, Any] = created.json()
    return body


def _default_organization_id(client: TestClient, headers: dict[str, str]) -> str:
    """Provision the tenancy root, and return the default organization's id."""
    response = client.get(_ME, headers=headers)
    assert response.status_code == 200, response.text
    organization_id: str = response.json()["organization"]["id"]
    return organization_id


def _identity(
    session_factory: Callable[[], Session],
    *,
    memberships: list[tuple[uuid.UUID, str]],
    email: str,
) -> tuple[uuid.UUID, str]:
    """An identity active in the first organization listed, holding a live dashboard session."""
    session = session_factory()
    try:
        user = User(email=email, full_name=email.split("@")[0].title(), active_organization_id=memberships[0][0])
        session.add(user)
        session.commit()
        session.refresh(user)
        for organization_id, role in memberships:
            session.add(
                OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active")
            )
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
        return user.id, token
    finally:
        session.close()


def _active_organization_id(session_factory: Callable[[], Session], user_id: uuid.UUID) -> uuid.UUID:
    session = session_factory()
    try:
        return session.get_one(User, user_id).active_organization_id
    finally:
        session.close()


def _organization_exists(session_factory: Callable[[], Session], organization_id: str) -> bool:
    session = session_factory()
    try:
        return session.get(Organization, uuid.UUID(organization_id)) is not None
    finally:
        session.close()


def test_an_unused_organization_is_deleted_and_the_caller_moved_out(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    default_id = _default_organization_id(client, master_key_header)
    created = _create_and_enter(client, master_key_header)
    extra = client.post(f"{API_ROOT}/workspaces", json={"name": "Second"}, headers=master_key_header)
    assert extra.status_code == 201, extra.text

    response = client.delete(_ME, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["organization"]["id"] == default_id
    assert client.get(_ME, headers=master_key_header).json()["organization"]["name"] == DEFAULT_ORGANIZATION_NAME
    assert not _organization_exists(db_session_factory, created["id"])
    session = db_session_factory()
    try:
        orphans = session.query(Workspace).filter(col(Workspace.organization_id) == uuid.UUID(created["id"])).count()
    finally:
        session.close()
    assert orphans == 0
    memberships = client.get(f"{_ME}/memberships", headers=master_key_header).json()
    assert [row["organization"]["id"] for row in memberships["data"]] == [default_id]


def test_the_default_organization_cannot_be_deleted(client: TestClient, master_key_header: dict[str, str]) -> None:
    default_id = _default_organization_id(client, master_key_header)
    _create_and_enter(client, master_key_header)
    client.post(f"{_ME}/switch", json={"organization_id": default_id}, headers=master_key_header)

    response = client.delete(_ME, headers=master_key_header)

    assert response.status_code == 400
    assert "default organization" in response.json()["detail"]


def test_an_organization_holding_an_api_key_is_kept(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Keys and usage are ON DELETE RESTRICT, so the refusal is the database's, and nothing is half-deleted."""
    created = _create_and_enter(client, master_key_header)
    key = client.post(f"{_ME}/keys", json={"key_name": "in-use"}, headers=master_key_header)
    assert key.status_code in (200, 201), key.text

    response = client.delete(_ME, headers=master_key_header)

    assert response.status_code == 409
    assert "no usage" in response.json()["detail"]
    assert _organization_exists(db_session_factory, created["id"])
    assert client.get(_ME, headers=master_key_header).json()["organization"]["id"] == created["id"]


def test_the_only_organization_a_caller_belongs_to_is_kept(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    created = _create_and_enter(client, master_key_header)
    _, token = _identity(
        db_session_factory, memberships=[(uuid.UUID(created["id"]), "owner")], email="solo@example.com"
    )
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.delete(_ME)

    assert response.status_code == 400
    assert "only organization" in response.json()["detail"]
    assert _organization_exists(db_session_factory, created["id"])


def test_a_member_cannot_delete_the_organization(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    default_id = uuid.UUID(_default_organization_id(client, master_key_header))
    created = _create_and_enter(client, master_key_header)
    _, token = _identity(
        db_session_factory,
        memberships=[(uuid.UUID(created["id"]), "member"), (default_id, "member")],
        email="member@example.com",
    )
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.delete(_ME)

    assert response.status_code == 403
    assert _organization_exists(db_session_factory, created["id"])


def test_an_admin_can_delete_the_organization(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    default_id = uuid.UUID(_default_organization_id(client, master_key_header))
    created = _create_and_enter(client, master_key_header)
    _, token = _identity(
        db_session_factory,
        memberships=[(uuid.UUID(created["id"]), "admin"), (default_id, "member")],
        email="admin@example.com",
    )
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.delete(_ME)

    assert response.status_code == 200, response.text
    assert response.json()["organization"]["id"] == str(default_id)
    assert not _organization_exists(db_session_factory, created["id"])


def test_everyone_pointed_at_a_deleted_organization_is_moved(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Onto another live membership where there is one, and otherwise onto the default organization."""
    default_id = uuid.UUID(_default_organization_id(client, master_key_header))
    created = uuid.UUID(_create_and_enter(client, master_key_header)["id"])
    elsewhere = uuid.UUID(_create_and_enter(client, master_key_header, name="Elsewhere")["id"])
    client.post(f"{_ME}/switch", json={"organization_id": str(created)}, headers=master_key_header)
    moved, _ = _identity(
        db_session_factory, memberships=[(created, "member"), (elsewhere, "member")], email="moved@example.com"
    )
    stranded, _ = _identity(db_session_factory, memberships=[(created, "member")], email="stranded@example.com")

    response = client.delete(_ME, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert _active_organization_id(db_session_factory, moved) == elsewhere
    assert _active_organization_id(db_session_factory, stranded) == default_id
