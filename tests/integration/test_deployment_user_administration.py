"""Deployment-wide account administration (`/v1/admin`), end to end and at the service.

Split the way `test_tenancy_authorization.py` explains: a master-key request is
always the one bootstrap operator, who is a superuser, so a header-authenticated
route test can only exercise the allowed path and the guards that fire on *self*.
The refusals that depend on being somebody else are reached two ways here. The
one that decides what an outsider sees, the 404, is asserted through the routes
as well, on a dashboard session cookie minted for an identity built for the case,
because the status is the whole point of it and only the HTTP layer reports one.
The rest are asserted at the service, which is where the rule lives.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.models.entities import DashboardSession, RuntimeSetting
from gateway.models.tenancy import (
    DeploymentUserUpdateRequest,
    Organization,
    OrganizationMember,
    User,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
)
from gateway.services.dashboard_session_service import (
    SESSION_COOKIE_NAME,
    create_dashboard_session,
    hash_session_token,
)
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.errors import (
    BootstrapOperatorProtectedError,
    DeploymentAdministrationUnavailableError,
    DeploymentUserNotFoundError,
    DeploymentUserSelfChangeError,
    EmptyDeploymentUserUpdateError,
)
from gateway.services.tenancy.provisioning_service import BOOTSTRAP_IDENTITY_KEY


def _list(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get("/v1/admin/users", headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _row(body: dict[str, Any], user_id: uuid.UUID) -> dict[str, Any]:
    match: list[dict[str, Any]] = [row for row in body["data"] if row["id"] == str(user_id)]
    assert match, f"{user_id} is not in {[row['id'] for row in body['data']]}"
    return match[0]


def _add_identity(
    session_factory: Callable[[], Session],
    *,
    organization_id: uuid.UUID,
    full_name: str,
    email: str,
    role: str = "member",
    status: str = "active",
) -> uuid.UUID:
    """Insert a second identity holding a membership, the way `test_tenancy_api` does."""
    session = session_factory()
    try:
        user = User(email=email, full_name=full_name, active_organization_id=organization_id)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(
            OrganizationMember(
                organization_id=organization_id,
                user_id=user.id,
                role=role,
                status=status,
            )
        )
        session.commit()
        return user.id
    finally:
        session.close()


def _default_organization_id(session_factory: Callable[[], Session]) -> uuid.UUID:
    session = session_factory()
    try:
        organization = session.query(Organization).filter(col(Organization.slug) == "default").one()
        return organization.id
    finally:
        session.close()


def _bootstrap_user_id(session_factory: Callable[[], Session]) -> uuid.UUID:
    session = session_factory()
    try:
        marker = session.get(RuntimeSetting, BOOTSTRAP_IDENTITY_KEY)
        assert marker is not None
        return uuid.UUID(marker.value)
    finally:
        session.close()


# =============================================================================
# The list
# =============================================================================


def test_the_list_carries_every_identity_with_its_organizations(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    # Provision the tenancy root by making one master-key request first.
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    operator_id = _bootstrap_user_id(db_session_factory)
    member_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )

    body = _list(client, master_key_header)

    assert body["count"] == 2
    operator = _row(body, operator_id)
    assert operator["is_bootstrap_operator"] is True
    # The master key resolves to the bootstrap operator, so this row is the
    # caller's own and the page disables its two lockout controls.
    assert operator["is_self"] is True
    assert operator["is_superuser"] is True
    assert operator["is_active"] is True

    member = _row(body, member_id)
    assert member["email"] == "ada@example.com"
    assert member["is_bootstrap_operator"] is False
    assert member["is_self"] is False
    assert member["is_superuser"] is False
    # Never signed in, which is null rather than a timestamp: the column exists
    # so this stays distinguishable from "signed in before the sessions expired".
    assert member["last_sign_in_at"] is None
    assert [organization["organization_id"] for organization in member["organizations"]] == [
        str(organization_id)
    ]
    assert member["organizations"][0]["role"] == "member"
    assert member["organizations"][0]["status"] == "active"


def test_the_list_shows_an_identity_whose_every_membership_is_suspended(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The case the organization roster deliberately hides, and this surface exists for."""
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    stuck_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Grace Hopper",
        email="grace@example.com",
        status="suspended",
    )

    roster = client.get("/v1/organizations/me/members", headers=master_key_header)
    assert roster.status_code == 200, roster.text
    assert str(stuck_id) not in [row["user_id"] for row in roster.json()["data"]]

    stuck = _row(_list(client, master_key_header), stuck_id)
    assert [organization["status"] for organization in stuck["organizations"]] == ["suspended"]


def test_a_session_stamps_the_identity_it_was_minted_for(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    test_config: GatewayConfig,
) -> None:
    """`last_sign_in_at` is written wherever a session is minted, not at each route."""
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    operator_id = _bootstrap_user_id(db_session_factory)
    assert _row(_list(client, master_key_header), operator_id)["last_sign_in_at"] is None

    response = client.post("/v1/auth/session", json={"master_key": test_config.master_key})
    assert response.status_code == 200, response.text

    assert _row(_list(client, master_key_header), operator_id)["last_sign_in_at"] is not None


# =============================================================================
# Changing an account
# =============================================================================


def test_deactivating_an_account_ends_its_dashboard_sessions(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    session = db_session_factory()
    try:
        session.add(
            DashboardSession(
                token_hash=hash_session_token("otari-sess-still-live"),
                user_id=member_id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
    finally:
        session.close()

    response = client.patch(
        f"/v1/admin/users/{member_id}",
        headers=master_key_header,
        json={"is_active": False},
    )

    assert response.status_code == 200, response.text
    assert response.json()["is_active"] is False
    session = db_session_factory()
    try:
        remaining = (
            session.query(DashboardSession).filter(col(DashboardSession.user_id) == member_id).count()
        )
    finally:
        session.close()
    # Ended now rather than refused the next time the cookie is presented, so
    # reactivating does not hand back every cookie the account was holding.
    assert remaining == 0

    revived = client.patch(
        f"/v1/admin/users/{member_id}",
        headers=master_key_header,
        json={"is_active": True},
    )
    assert revived.status_code == 200, revived.text
    assert revived.json()["is_active"] is True


def test_the_superuser_flag_flips_on_its_own(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Omitting `is_active` leaves it alone: the two are separate decisions."""
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )

    response = client.patch(
        f"/v1/admin/users/{member_id}",
        headers=master_key_header,
        json={"is_superuser": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["is_superuser"] is True
    assert response.json()["is_active"] is True


def test_an_operator_cannot_deactivate_or_demote_themselves(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    operator_id = _bootstrap_user_id(db_session_factory)

    for body in ({"is_active": False}, {"is_superuser": False}):
        response = client.patch(f"/v1/admin/users/{operator_id}", headers=master_key_header, json=body)
        assert response.status_code == 400, response.text

    assert _row(_list(client, master_key_header), operator_id)["is_active"] is True
    assert _row(_list(client, master_key_header), operator_id)["is_superuser"] is True


def test_a_change_naming_neither_flag_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )

    response = client.patch(f"/v1/admin/users/{member_id}", headers=master_key_header, json={})

    assert response.status_code == 400, response.text


def test_an_unknown_account_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.patch(
        f"/v1/admin/users/{uuid.uuid4()}",
        headers=master_key_header,
        json={"is_active": False},
    )

    assert response.status_code == 404, response.text


def test_the_operator_probe_answers_for_the_bootstrap_caller(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.get("/v1/admin/access", headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {"granted": True}


def _session_for(
    session_factory: Callable[[], Session],
    *,
    organization_id: uuid.UUID,
    email: str,
    is_superuser: bool = False,
) -> tuple[uuid.UUID, str]:
    """An identity holding a live dashboard session, and the cookie that names it."""
    session = session_factory()
    try:
        user = User(
            email=email,
            full_name=email.split("@")[0].title(),
            active_organization_id=organization_id,
            is_superuser=is_superuser,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(
            OrganizationMember(organization_id=organization_id, user_id=user.id, role="member", status="active")
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


def test_a_non_operator_session_is_refused_with_404_by_every_route(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The status, not only the exception: a 403 here would sign the caller out.

    Reachable through the routes because a session cookie names whoever it was
    minted for, unlike the master key, which is always the bootstrap operator.
    ``/access`` is the one endpoint that answers rather than hides, which is what
    leaves the sidebar something to gate on.
    """
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id, token = _session_for(db_session_factory, organization_id=organization_id, email="ada@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = client.get("/v1/admin/users")
        access = client.get("/v1/admin/access")
        patched = client.patch(f"/v1/admin/users/{member_id}", json={"is_superuser": True})
    finally:
        client.cookies.clear()

    assert listed.status_code == 404, listed.text
    assert patched.status_code == 404, patched.text
    assert access.status_code == 200, access.text
    assert access.json() == {"granted": False}


def test_an_operator_session_that_is_not_the_bootstrap_one_administers(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The other side of the cookie path: `is_self` names the caller, not the marker."""
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    operator_id, token = _session_for(
        db_session_factory,
        organization_id=organization_id,
        email="operator@example.com",
        is_superuser=True,
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = client.get("/v1/admin/users")
        own = client.patch(f"/v1/admin/users/{operator_id}", json={"is_active": False})
        anchor = client.patch(
            f"/v1/admin/users/{_bootstrap_user_id(db_session_factory)}",
            json={"is_active": False},
        )
    finally:
        client.cookies.clear()

    assert listed.status_code == 200, listed.text
    assert _row(listed.json(), operator_id)["is_self"] is True
    assert _row(listed.json(), _bootstrap_user_id(db_session_factory))["is_bootstrap_operator"] is True
    # Both lockout guards, at the status the dashboard renders.
    assert own.status_code == 400, own.text
    assert anchor.status_code == 400, anchor.text


# =============================================================================
# Who may reach the surface
# =============================================================================
#
# No module-level ``pytestmark``: the route cases above are synchronous, so the
# asyncio marker is applied per test rather than to the file.


async def _identity(
    db: AsyncSession,
    *,
    full_name: str,
    is_superuser: bool = False,
) -> tuple[User, Organization]:
    organization = await OrganizationRepository(db).create_organization(
        name=full_name,
        slug=full_name.lower().replace(" ", "-"),
        created_by_user_id=None,
    )
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
        is_superuser=is_superuser,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role="owner",
    )
    await db.commit()
    return user, organization


async def _mark_bootstrap(db: AsyncSession, user: User) -> None:
    db.add(RuntimeSetting(key=BOOTSTRAP_IDENTITY_KEY, value=str(user.id)))
    await db.commit()


@pytest.mark.asyncio
async def test_a_non_operator_is_told_the_surface_does_not_exist(async_db: AsyncSession) -> None:
    """404 and not 403: the dashboard's `apiFetch` drops the session on a 403."""
    outsider, _ = await _identity(async_db, full_name="Ada Lovelace")
    service = DeploymentUserService(async_db)

    assert await service.has_administration_access(outsider) is False
    with pytest.raises(DeploymentAdministrationUnavailableError):
        await service.list_users(actor=outsider)
    with pytest.raises(DeploymentAdministrationUnavailableError):
        await service.update_user(
            actor=outsider,
            user_id=outsider.id,
            request=DeploymentUserUpdateRequest(is_superuser=True),
        )


@pytest.mark.asyncio
async def test_the_bootstrap_marker_admits_an_operator_whose_flag_was_cleared(
    async_db: AsyncSession,
) -> None:
    """The arm that keeps a deployment from being locked out of its own administration."""
    operator, _ = await _identity(async_db, full_name="Operator")
    await _mark_bootstrap(async_db, operator)
    service = DeploymentUserService(async_db)

    assert operator.is_superuser is False
    assert await service.has_administration_access(operator) is True
    assert (await service.list_users(actor=operator)).count == 1


@pytest.mark.asyncio
async def test_a_superuser_reaches_the_surface_with_no_marker_at_all(async_db: AsyncSession) -> None:
    superuser, _ = await _identity(async_db, full_name="Root", is_superuser=True)

    assert await DeploymentUserService(async_db).has_administration_access(superuser) is True


@pytest.mark.asyncio
async def test_another_operator_cannot_deactivate_or_demote_the_bootstrap_identity(
    async_db: AsyncSession,
) -> None:
    anchor, _ = await _identity(async_db, full_name="Operator")
    await _mark_bootstrap(async_db, anchor)
    second, _ = await _identity(async_db, full_name="Root", is_superuser=True)
    service = DeploymentUserService(async_db)

    for request in (
        DeploymentUserUpdateRequest(is_active=False),
        DeploymentUserUpdateRequest(is_superuser=False),
    ):
        with pytest.raises(BootstrapOperatorProtectedError):
            await service.update_user(actor=second, user_id=anchor.id, request=request)

    # Granting is unguarded, which is what lets an operator repair a cleared flag.
    granted = await service.update_user(
        actor=second,
        user_id=anchor.id,
        request=DeploymentUserUpdateRequest(is_superuser=True),
    )
    assert granted.is_superuser is True
    assert granted.is_bootstrap_operator is True


@pytest.mark.asyncio
async def test_the_service_refuses_the_two_shapes_the_routes_also_refuse(
    async_db: AsyncSession,
) -> None:
    operator, _ = await _identity(async_db, full_name="Root", is_superuser=True)
    service = DeploymentUserService(async_db)

    with pytest.raises(EmptyDeploymentUserUpdateError):
        await service.update_user(
            actor=operator,
            user_id=operator.id,
            request=DeploymentUserUpdateRequest(),
        )
    with pytest.raises(DeploymentUserSelfChangeError):
        await service.update_user(
            actor=operator,
            user_id=operator.id,
            request=DeploymentUserUpdateRequest(is_active=False),
        )
    with pytest.raises(DeploymentUserNotFoundError):
        await service.update_user(
            actor=operator,
            user_id=uuid.uuid4(),
            request=DeploymentUserUpdateRequest(is_active=False),
        )


@pytest.mark.asyncio
async def test_a_session_records_the_sign_in_on_the_identity(async_db: AsyncSession) -> None:
    """The stamp lives in `create_dashboard_session`, so every sign-in flow gets it."""
    user, _ = await _identity(async_db, full_name="Ada Lovelace")
    assert user.last_sign_in_at is None

    await create_dashboard_session(async_db, 24, user_id=user.id)
    await async_db.commit()
    await async_db.refresh(user)

    assert user.last_sign_in_at is not None
