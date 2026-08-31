"""The organization-scoped routing-policy read sees the caller's own workspaces and no more.

``/v1/routing/policies`` is deployment-wide and operator-only, and stays that
way: its ``workspace_id`` parameter is a filter the client supplies, so nothing
but the operator gate stands between a signed-in member and another
organization's routing. ``/v1/organizations/me/routing-policies`` is the
tenant's View half of it (otari-ai#1942), shaped like the usage scope
(otari#837), and this suite is that file's shape over the ``routing_policies``
table: two organizations with policies in both, and every assertion names the
rows that must be absent rather than counting the ones that came back.

The last test is the control: the deployment-wide route must still refuse an
organization owner. A change that made these pass by loosening that gate would
have reopened otari-ai#1880.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import DashboardSession, RoutingPolicy
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

_SCOPED_PATH = "/v1/organizations/me/routing-policies"


@dataclass
class _World:
    """Two organizations, their workspaces, and one session cookie per identity."""

    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)


# Policy names double as row identity: every assertion below is "these names and
# no others", which is what makes a leak visible rather than merely miscounted.
_ALPHA_ONE_POLICIES = ("alpha-one-fast", "alpha-one-cheap")
_ALPHA_TWO_POLICIES = ("alpha-two-fast",)
_BETA_POLICIES = ("beta-one-fast",)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    is_superuser: bool = False,
    workspace_ids: tuple[uuid.UUID, ...] = (),
    membership: bool = True,
) -> str:
    """Create an identity with a live dashboard session, and return its cookie."""
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
        is_superuser=is_superuser,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    if membership:
        session.add(
            OrganizationMember(
                organization_id=organization_id,
                user_id=user.id,
                role=role,
                status="active",
            )
        )
    for workspace_id in workspace_ids:
        session.add(
            WorkspaceMember(
                workspace_id=workspace_id,
                user_id=user.id,
                role="member",
                status="active",
            )
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
    return token


def _policy_rows(session: Session, workspace_id: uuid.UUID, names: tuple[str, ...]) -> None:
    for name in names:
        session.add(
            RoutingPolicy(
                id=str(uuid.uuid4()),
                name=name,
                spec={"select": [{"default": "openai:gpt-4o-mini"}]},
                workspace_id=workspace_id,
            )
        )
    session.commit()


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """Two tenants with stored policies in both, and the identities that read them."""
    # One master-key call provisions the tenancy root, so the organizations built
    # below sit beside a real default rather than replacing it.
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK

    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha")
        beta = Organization(name="Beta", slug="beta")
        session.add_all([alpha, beta])
        session.commit()
        session.refresh(alpha)
        session.refresh(beta)

        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        alpha_two = Workspace(name="Alpha two", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, alpha_two, beta_one])
        session.commit()
        for workspace in (alpha_one, alpha_two, beta_one):
            session.refresh(workspace)

        _policy_rows(session, alpha_one.id, _ALPHA_ONE_POLICIES)
        _policy_rows(session, alpha_two.id, _ALPHA_TWO_POLICIES)
        _policy_rows(session, beta_one.id, _BETA_POLICIES)

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "alpha_two": alpha_two.id, "beta_one": beta_one.id}
        built.sessions = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            "alpha_newcomer": _identity(session, email="new@alpha.test", organization_id=alpha.id),
            "beta_owner": _identity(session, email="owner@beta.test", organization_id=beta.id, role="owner"),
            "impostor": _identity(
                session,
                email="impostor@nowhere.test",
                organization_id=alpha.id,
                membership=False,
            ),
            "superuser": _identity(
                session,
                email="root@beta.test",
                organization_id=beta.id,
                role="owner",
                is_superuser=True,
            ),
        }
        return built
    finally:
        session.close()


def _as(client: TestClient, world: _World, who: str) -> tuple[int, object]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.get(_SCOPED_PATH)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
        return response.status_code, body
    finally:
        client.cookies.clear()


def _stored_names(client: TestClient, world: _World, who: str) -> set[str]:
    """The stored policy names this caller is shown.

    Filtered to ``source == "stored"`` so a config-file policy the test
    deployment happens to define cannot pad or break an assertion: config
    policies are deployment-wide by design and outside what this suite pins.
    """
    code, body = _as(client, world, who)
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, list)
    return {row["name"] for row in body if row["source"] == "stored"}


def test_an_owner_reads_every_workspace_in_their_own_organization(client: TestClient, world: _World) -> None:
    assert _stored_names(client, world, "alpha_owner") == set(_ALPHA_ONE_POLICIES) | set(_ALPHA_TWO_POLICIES)


def test_an_owner_reads_no_other_organizations_rows(client: TestClient, world: _World) -> None:
    """Stated on its own, because it is the claim the whole route rests on."""
    assert _stored_names(client, world, "alpha_owner").isdisjoint(_BETA_POLICIES)
    assert _stored_names(client, world, "beta_owner") == set(_BETA_POLICIES)


def test_a_member_reads_only_the_workspaces_they_belong_to(client: TestClient, world: _World) -> None:
    """``alpha_member`` belongs to Alpha one and not Alpha two.

    Both are their organization's, so an implementation that scoped to the
    organization alone would pass every other test in this file and fail here.
    """
    listed = _stored_names(client, world, "alpha_member")
    assert listed == set(_ALPHA_ONE_POLICIES)
    assert listed.isdisjoint(_ALPHA_TWO_POLICIES)
    assert listed.isdisjoint(_BETA_POLICIES)


def test_a_member_of_no_workspace_reads_no_stored_rows_rather_than_a_refusal(
    client: TestClient, world: _World
) -> None:
    """Nothing was refused; no stored policy is theirs to see yet."""
    code, body = _as(client, world, "alpha_newcomer")
    assert code == status.HTTP_200_OK, body
    assert _stored_names(client, world, "alpha_newcomer") == set()


def test_a_superuser_reads_their_active_organization_and_not_every_tenant(client: TestClient, world: _World) -> None:
    """This route is scoped even for an operator; ``/v1/routing/policies`` is where they read across tenants."""
    listed = _stored_names(client, world, "superuser")
    assert listed == set(_BETA_POLICIES)
    assert listed.isdisjoint(_ALPHA_ONE_POLICIES)


def test_an_active_organization_pointer_with_no_membership_behind_it_reads_nothing(
    client: TestClient, world: _World
) -> None:
    """The pointer is not the authority; the membership is."""
    code, body = _as(client, world, "impostor")
    assert code in {status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND}, body


def test_the_deployment_wide_route_still_refuses_a_tenant(client: TestClient, world: _World) -> None:
    """The control: this route exists so that gate does not have to loosen."""
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_owner"])
    try:
        refused = client.get("/v1/routing/policies")
        assert refused.status_code == status.HTTP_403_FORBIDDEN, refused.text
    finally:
        client.cookies.clear()
