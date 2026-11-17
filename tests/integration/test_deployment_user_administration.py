"""Deployment-wide account administration (`/api/v1/admin`), end to end and at the service.

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
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api.deps import get_deployment_user_service
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    BootstrapOperatorProtectedError,
    DeploymentAdministrationUnavailableError,
    DeploymentUserNotFoundError,
    DeploymentUserSelfChangeError,
    EmptyDeploymentUserUpdateError,
)
from gateway.models.platform import RuntimeSetting
from gateway.models.tenancy import (
    DashboardSession,
    DeploymentUserUpdateRequest,
    Organization,
    OrganizationMember,
    User,
)
from gateway.repositories.organizations.bootstrap_marker_repository import BootstrapMarkerRepository
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.password_service import hash_password
from gateway.services.tenancy.provisioning_service import BOOTSTRAP_IDENTITY_KEY


def _list(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get(f"{API_ROOT}/admin/users", headers=headers)
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
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
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
    assert [organization["organization_id"] for organization in member["organizations"]] == [str(organization_id)]
    assert member["organizations"][0]["role"] == "member"
    assert member["organizations"][0]["status"] == "active"


def test_the_list_shows_an_identity_whose_every_membership_is_suspended(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The case the organization roster deliberately hides, and this surface exists for."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    stuck_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Grace Hopper",
        email="grace@example.com",
        status="suspended",
    )

    roster = client.get(f"{API_ROOT}/organizations/me/members", headers=master_key_header)
    assert roster.status_code == 200, roster.text
    assert str(stuck_id) not in [row["user_id"] for row in roster.json()["data"]]

    stuck = _row(_list(client, master_key_header), stuck_id)
    assert [organization["status"] for organization in stuck["organizations"]] == ["suspended"]


def test_search_narrows_the_page_and_the_count_by_name_or_email(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    ada_id = _add_identity(
        db_session_factory, organization_id=organization_id, full_name="Ada Lovelace", email="ada@example.com"
    )
    grace_id = _add_identity(
        db_session_factory, organization_id=organization_id, full_name="Grace Hopper", email="grace@navy.example"
    )

    def search(term: str) -> dict[str, Any]:
        response = client.get(f"{API_ROOT}/admin/users", headers=master_key_header, params={"search": term})
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body

    by_name = search("LOVELACE")
    assert [row["id"] for row in by_name["data"]] == [str(ada_id)]
    assert by_name["count"] == 1

    by_email = search("navy")
    assert [row["id"] for row in by_email["data"]] == [str(grace_id)]
    assert by_email["count"] == 1

    # A wildcard is matched literally rather than matching everyone.
    assert search("%")["count"] == 0
    # A blank term is no filter at all.
    assert search("   ")["count"] == 3


def test_the_list_pages_with_a_total_count(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    for index in range(3):
        _add_identity(
            db_session_factory,
            organization_id=organization_id,
            full_name=f"Member {index}",
            email=f"member{index}@example.com",
        )

    pages = [
        client.get(f"{API_ROOT}/admin/users", headers=master_key_header, params={"skip": skip, "limit": 2}).json()
        for skip in (0, 2)
    ]

    assert [page["count"] for page in pages] == [4, 4]
    assert [len(page["data"]) for page in pages] == [2, 2]
    ids = [row["id"] for page in pages for row in page["data"]]
    assert len(set(ids)) == 4


def test_a_session_stamps_the_identity_it_was_minted_for(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    test_config: GatewayConfig,
) -> None:
    """`last_sign_in_at` is written wherever a session is minted, not at each route."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    operator_id = _bootstrap_user_id(db_session_factory)
    assert _row(_list(client, master_key_header), operator_id)["last_sign_in_at"] is None

    response = client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key})
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
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
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
        f"{API_ROOT}/admin/users/{member_id}",
        headers=master_key_header,
        json={"is_active": False},
    )

    assert response.status_code == 200, response.text
    assert response.json()["is_active"] is False
    session = db_session_factory()
    try:
        remaining = session.query(DashboardSession).filter(col(DashboardSession.user_id) == member_id).count()
    finally:
        session.close()
    # Ended now rather than refused the next time the cookie is presented, so
    # reactivating does not hand back every cookie the account was holding.
    assert remaining == 0

    revived = client.patch(
        f"{API_ROOT}/admin/users/{member_id}",
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
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )

    response = client.patch(
        f"{API_ROOT}/admin/users/{member_id}",
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
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    operator_id = _bootstrap_user_id(db_session_factory)

    for body in ({"is_active": False}, {"is_superuser": False}):
        response = client.patch(f"{API_ROOT}/admin/users/{operator_id}", headers=master_key_header, json=body)
        assert response.status_code == 400, response.text

    assert _row(_list(client, master_key_header), operator_id)["is_active"] is True
    assert _row(_list(client, master_key_header), operator_id)["is_superuser"] is True


def test_a_change_naming_neither_flag_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )

    response = client.patch(f"{API_ROOT}/admin/users/{member_id}", headers=master_key_header, json={})

    assert response.status_code == 400, response.text


def test_an_unknown_account_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.patch(
        f"{API_ROOT}/admin/users/{uuid.uuid4()}",
        headers=master_key_header,
        json={"is_active": False},
    )

    assert response.status_code == 404, response.text


def test_the_operator_probe_answers_for_the_bootstrap_caller(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.get(f"{API_ROOT}/admin/access", headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {"granted": True, "is_bootstrap_operator": True}


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
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    member_id, token = _session_for(db_session_factory, organization_id=organization_id, email="ada@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = client.get(f"{API_ROOT}/admin/users")
        access = client.get(f"{API_ROOT}/admin/access")
        patched = client.patch(f"{API_ROOT}/admin/users/{member_id}", json={"is_superuser": True})
    finally:
        client.cookies.clear()

    assert listed.status_code == 404, listed.text
    assert patched.status_code == 404, patched.text
    assert access.status_code == 200, access.text
    assert access.json() == {"granted": False, "is_bootstrap_operator": False}


def test_an_operator_session_that_is_not_the_bootstrap_one_administers(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The other side of the cookie path: `is_self` names the caller, not the marker."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    organization_id = _default_organization_id(db_session_factory)
    operator_id, token = _session_for(
        db_session_factory,
        organization_id=organization_id,
        email="operator@example.com",
        is_superuser=True,
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = client.get(f"{API_ROOT}/admin/users")
        own = client.patch(f"{API_ROOT}/admin/users/{operator_id}", json={"is_active": False})
        anchor = client.patch(
            f"{API_ROOT}/admin/users/{_bootstrap_user_id(db_session_factory)}",
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
    service = get_deployment_user_service(async_db, UnitOfWork(async_db))

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
    service = get_deployment_user_service(async_db, UnitOfWork(async_db))

    assert operator.is_superuser is False
    assert await service.has_administration_access(operator) is True
    assert (await service.list_users(actor=operator)).count == 1


@pytest.mark.asyncio
async def test_a_superuser_reaches_the_surface_with_no_marker_at_all(async_db: AsyncSession) -> None:
    superuser, _ = await _identity(async_db, full_name="Root", is_superuser=True)

    assert (
        await get_deployment_user_service(async_db, UnitOfWork(async_db)).has_administration_access(superuser) is True
    )


@pytest.mark.asyncio
async def test_another_operator_cannot_deactivate_or_demote_the_bootstrap_identity(
    async_db: AsyncSession,
) -> None:
    anchor, _ = await _identity(async_db, full_name="Operator")
    await _mark_bootstrap(async_db, anchor)
    second, _ = await _identity(async_db, full_name="Root", is_superuser=True)
    service = get_deployment_user_service(async_db, UnitOfWork(async_db))

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
    service = get_deployment_user_service(async_db, UnitOfWork(async_db))

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
async def test_a_deleted_identity_cached_in_the_session_is_not_found(
    async_db: AsyncSession, db_session_factory: Callable[[], Session]
) -> None:
    """A concurrently deleted account must return not-found rather than fail while refreshing it."""
    operator, organization = await _identity(async_db, full_name="Root", is_superuser=True)
    target = User(full_name="Removed account", active_organization_id=organization.id)
    async_db.add(target)
    await async_db.commit()
    service = get_deployment_user_service(async_db, UnitOfWork(async_db))

    with db_session_factory() as other:
        other.execute(delete(User).where(col(User.id) == target.id))
        other.commit()

    # The repository's ordinary primary-key lookup can still return the stale identity-map entry.
    assert await service.users.get(target.id) is target
    with pytest.raises(DeploymentUserNotFoundError):
        await service.update_user(
            actor=operator,
            user_id=target.id,
            request=DeploymentUserUpdateRequest(is_active=False),
        )


# =============================================================================
# Handing over the bootstrap operator role
# =============================================================================

_PASSWORD = "correct horse battery staple"


def _operator(
    session_factory: Callable[[], Session],
    *,
    organization_id: uuid.UUID,
    email: str,
    is_superuser: bool = True,
    is_active: bool = True,
    with_password: bool = True,
    verified: bool = True,
    vouched: bool = False,
) -> uuid.UUID:
    """An identity that can take the role unless a flag says otherwise."""
    session = session_factory()
    try:
        user = User(
            email=email,
            full_name=email.split("@")[0].title(),
            active_organization_id=organization_id,
            is_superuser=is_superuser,
            is_active=is_active,
            hashed_password=hash_password(_PASSWORD) if with_password else None,
            email_verified_at=datetime.now(UTC) if verified else None,
            email_vouched_at=datetime.now(UTC) if vouched else None,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(
            OrganizationMember(organization_id=organization_id, user_id=user.id, role="member", status="active")
        )
        session.commit()
        return user.id
    finally:
        session.close()


def _is_superuser(session_factory: Callable[[], Session], user_id: uuid.UUID) -> bool:
    session = session_factory()
    try:
        user = session.get(User, user_id)
        assert user is not None
        return user.is_superuser
    finally:
        session.close()


def _provisioned(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200


def _claim(client: TestClient, master_key_header: dict[str, str], email: str = "owner@example.com") -> None:
    """Give the bootstrap operator an address and a password, which claims the deployment."""
    claimed = client.put(
        f"{API_ROOT}/auth/password", headers=master_key_header, json={"email": email, "new_password": _PASSWORD}
    )
    assert claimed.status_code == 200, claimed.text


def _hand_over(client: TestClient, user_id: uuid.UUID, headers: dict[str, str] | None = None, **body: str) -> Any:
    return client.post(f"{API_ROOT}/admin/users/{user_id}/bootstrap-operator", headers=headers, json=body or None)


def test_the_master_key_hands_the_role_to_another_operator(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _provisioned(client, master_key_header)
    previous = _bootstrap_user_id(db_session_factory)
    successor = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )

    response = _hand_over(client, successor, master_key_header)

    assert response.status_code == 204, response.text
    assert _bootstrap_user_id(db_session_factory) == successor
    # The previous one stays an operator, now an ordinary one the successor may demote.
    assert _is_superuser(db_session_factory, previous) is True
    listed = _list(client, master_key_header)
    assert _row(listed, previous)["is_bootstrap_operator"] is False
    assert _row(listed, successor)["is_bootstrap_operator"] is True
    # The master key now acts as the successor.
    assert _row(listed, successor)["is_self"] is True
    demoted = client.patch(
        f"{API_ROOT}/admin/users/{previous}", headers=master_key_header, json={"is_superuser": False}
    )
    assert demoted.status_code == 200, demoted.text


def test_the_bootstrap_session_hands_the_role_on_with_its_current_password(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The dashboard path: a session, so a stolen cookie alone is not enough."""
    _provisioned(client, master_key_header)
    _claim(client, master_key_header)
    successor = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "owner@example.com", "password": _PASSWORD})
    assert signed_in.status_code == 200, signed_in.text

    try:
        missing = _hand_over(client, successor)
        wrong = _hand_over(client, successor, current_password="not the password")
        assert _bootstrap_user_id(db_session_factory) != successor
        handed = _hand_over(client, successor, current_password=_PASSWORD)
    finally:
        client.cookies.clear()

    assert missing.status_code == 400, missing.text
    assert "current password" in missing.text
    assert wrong.status_code == 400, wrong.text
    assert handed.status_code == 204, handed.text
    assert _bootstrap_user_id(db_session_factory) == successor


def test_after_a_handover_the_master_key_acts_on_the_successor(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    db_session_factory: Callable[[], Session],
) -> None:
    """Master-key sign-in stays retired, and a master-key password reset now lands on the successor."""
    _provisioned(client, master_key_header)
    _claim(client, master_key_header)
    successor = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )
    assert _hand_over(client, successor, master_key_header).status_code == 204

    sign_in = client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key})
    reset = client.put(f"{API_ROOT}/auth/password", headers=master_key_header, json={"new_password": "a-new-password"})
    client.cookies.clear()
    as_successor = client.post(
        f"{API_ROOT}/auth/session", json={"email": "grace@example.com", "password": "a-new-password"}
    )
    client.cookies.clear()

    assert sign_in.status_code == 403, sign_in.text
    assert reset.status_code == 200, reset.text
    assert as_successor.status_code == 200, as_successor.text


def test_handing_over_an_unclaimed_deployment_claims_it(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    db_session_factory: Callable[[], Session],
) -> None:
    """The successor holds a password, so master-key sign-in retires with the handover."""
    _provisioned(client, master_key_header)
    assert client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key}).status_code == 200
    client.cookies.clear()
    successor = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )

    assert _hand_over(client, successor, master_key_header).status_code == 204

    sign_in = client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key})
    client.cookies.clear()
    assert sign_in.status_code == 403, sign_in.text


def test_the_access_probe_says_who_holds_the_role(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _provisioned(client, master_key_header)
    _, token = _session_for(
        db_session_factory,
        organization_id=_default_organization_id(db_session_factory),
        email="operator@example.com",
        is_superuser=True,
    )

    as_master_key = client.get(f"{API_ROOT}/admin/access", headers=master_key_header)
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        as_operator = client.get(f"{API_ROOT}/admin/access")
    finally:
        client.cookies.clear()

    assert as_master_key.json() == {"granted": True, "is_bootstrap_operator": True}
    assert as_operator.json() == {"granted": True, "is_bootstrap_operator": False}


def test_another_operator_cannot_take_the_role(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """A 400 and not a 403, which would sign an operator in good standing out."""
    _provisioned(client, master_key_header)
    previous = _bootstrap_user_id(db_session_factory)
    operator_id, token = _session_for(
        db_session_factory,
        organization_id=_default_organization_id(db_session_factory),
        email="operator@example.com",
        is_superuser=True,
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        response = _hand_over(client, operator_id)
    finally:
        client.cookies.clear()

    assert response.status_code == 400, response.text
    assert _bootstrap_user_id(db_session_factory) == previous


@pytest.mark.parametrize(
    ("flags", "detail"),
    [
        ({"is_superuser": False}, "Make this account an operator"),
        ({"is_active": False}, "Reactivate this account"),
        ({"with_password": False}, "no password"),
        ({"verified": False}, "not verified"),
        ({"vouched": True}, "only vouched for"),
    ],
)
def test_a_target_that_could_not_hold_the_role_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    flags: dict[str, bool],
    detail: str,
) -> None:
    _provisioned(client, master_key_header)
    previous = _bootstrap_user_id(db_session_factory)
    target = _operator(
        db_session_factory,
        organization_id=_default_organization_id(db_session_factory),
        email="grace@example.com",
        **flags,
    )

    response = _hand_over(client, target, master_key_header)

    assert response.status_code == 400, response.text
    assert detail in response.text
    assert _bootstrap_user_id(db_session_factory) == previous


def test_a_target_whose_address_is_blocked_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Otherwise the role would land on an address the blocklist already refuses sign-in to."""
    _provisioned(client, master_key_header)
    target = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )
    blocked = client.post(f"{API_ROOT}/admin/blocked-emails", headers=master_key_header, json={"value": "example.com"})
    assert blocked.status_code == 201, blocked.text

    response = _hand_over(client, target, master_key_header)

    assert response.status_code == 400, response.text
    assert "blocked" in response.text


def test_handing_the_role_to_itself_or_nobody_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _provisioned(client, master_key_header)
    previous = _bootstrap_user_id(db_session_factory)

    itself = _hand_over(client, previous, master_key_header)
    nobody = _hand_over(client, uuid.uuid4(), master_key_header)

    assert itself.status_code == 400, itself.text
    assert nobody.status_code == 404, nobody.text


def test_another_operator_cannot_get_a_password_link_for_the_bootstrap_operator(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """With no mail the link comes back to the caller, who could sign in as the bootstrap operator."""
    _provisioned(client, master_key_header)
    _claim(client, master_key_header)
    bootstrap_id = _bootstrap_user_id(db_session_factory)
    _, token = _session_for(
        db_session_factory,
        organization_id=_default_organization_id(db_session_factory),
        email="operator@example.com",
        is_superuser=True,
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        refused = client.post(f"{API_ROOT}/admin/users/{bootstrap_id}/password-link")
    finally:
        client.cookies.clear()
    allowed = client.post(f"{API_ROOT}/admin/users/{bootstrap_id}/password-link", headers=master_key_header)

    assert refused.status_code == 409, refused.text
    assert allowed.status_code == 201, allowed.text


def test_a_bootstrap_session_without_a_password_cannot_hand_over(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    db_session_factory: Callable[[], Session],
) -> None:
    """A master-key sign-in session on an unclaimed deployment has no password to step up with."""
    _provisioned(client, master_key_header)
    successor = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )
    assert client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key}).status_code == 200
    try:
        response = _hand_over(client, successor)
    finally:
        client.cookies.clear()

    assert response.status_code == 400, response.text
    assert "Set a password" in response.text
    assert _bootstrap_user_id(db_session_factory) != successor


def test_the_bootstrap_session_cannot_get_a_password_link_for_its_own_account(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """A link handed back would set its password without the current one, getting past the step-up."""
    _provisioned(client, master_key_header)
    _claim(client, master_key_header)
    bootstrap_id = _bootstrap_user_id(db_session_factory)
    assert (
        client.post(f"{API_ROOT}/auth/session", json={"email": "owner@example.com", "password": _PASSWORD}).status_code
        == 200
    )
    try:
        response = client.post(f"{API_ROOT}/admin/users/{bootstrap_id}/password-link")
    finally:
        client.cookies.clear()

    assert response.status_code == 409, response.text


def test_a_handover_voids_a_password_link_issued_before_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Whoever was handed the link must not be able to set the new bootstrap operator's password."""
    _provisioned(client, master_key_header)
    successor = _operator(
        db_session_factory, organization_id=_default_organization_id(db_session_factory), email="grace@example.com"
    )
    link = client.post(f"{API_ROOT}/admin/users/{successor}/password-link", headers=master_key_header)
    assert link.status_code == 201, link.text
    token = link.json()["link"].rsplit("token=", 1)[-1]

    assert _hand_over(client, successor, master_key_header).status_code == 204
    redeemed = client.post(
        f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": "a-new-password"}
    )

    assert redeemed.status_code == 400, redeemed.text


@pytest.mark.asyncio
async def test_the_marker_moves_only_from_the_identity_it_still_names(async_db: AsyncSession) -> None:
    """Two handovers racing from the same operator: the second finds the marker moved."""
    anchor, _ = await _identity(async_db, full_name="Operator")
    await _mark_bootstrap(async_db, anchor)
    first, _ = await _identity(async_db, full_name="First", is_superuser=True)
    second, _ = await _identity(async_db, full_name="Second", is_superuser=True)
    markers = BootstrapMarkerRepository(async_db)

    assert await markers.move(from_user_id=anchor.id, to_user_id=first.id) is True
    assert await markers.move(from_user_id=anchor.id, to_user_id=second.id) is False
    assert await markers.user_id() == first.id
