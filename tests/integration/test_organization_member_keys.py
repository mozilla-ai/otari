"""The member-scoped key surface touches the caller's own keys and no more.

``/v1/keys`` is deployment-wide and operator-only (otari-ai#1880), which left a
hosted organization member with no way to mint a key (mozilla-ai/otari-ai#1941).
``/v1/organizations/me/keys`` is the tenant's half of it, and the whole of its
correctness is that ownership and workspace scope are decided by the caller's
identity and memberships rather than by anything the request carries. So the
suite is written against a world holding two organizations, two members sharing
a workspace, and keys owned by each, and the assertions name the rows that must
be absent or refused rather than counting the ones that came back.

The control group at the bottom matters as much as the rest: the deployment-wide
router must still refuse a plain member, and a member-minted key must never be
budget-exempt. A change that made the rest pass by loosening either would have
reopened otari-ai#1880 or opened a budget bypass.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import APIKey, DashboardSession
from gateway.models.entities import User as BillingUser
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

_PREFIX = "/v1/organizations/me/keys"


@dataclass
class _World:
    """Two organizations, their workspaces, and one session cookie per identity."""

    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    users: dict[str, uuid.UUID] = field(default_factory=dict)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    workspace_ids: tuple[uuid.UUID, ...] = (),
    membership: bool = True,
) -> tuple[uuid.UUID, str]:
    """Create an identity with a live dashboard session, and return its cookie."""
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
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
    return user.id, token


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """Two tenants and the identities that act in them."""
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

        # Alpha one is created first, so with no workspace named "Default" it is
        # what ``organization_default_workspace_id`` resolves, which the
        # omitted-workspace tests below lean on.
        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        session.add(alpha_one)
        session.commit()
        alpha_two = Workspace(name="Alpha two", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_two, beta_one])
        session.commit()
        for workspace in (alpha_one, alpha_two, beta_one):
            session.refresh(workspace)

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "alpha_two": alpha_two.id, "beta_one": beta_one.id}
        people = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            # Belongs to one of alpha's two workspaces, which is the case the
            # surface exists for.
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            # A second member of the same workspace, so "may see the workspace"
            # and "owns the key" can be told apart.
            "alpha_colleague": _identity(
                session,
                email="colleague@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            # In the organization, in none of its workspaces.
            "alpha_newcomer": _identity(session, email="new@alpha.test", organization_id=alpha.id),
            "beta_owner": _identity(session, email="owner@beta.test", organization_id=beta.id, role="owner"),
            # Points at alpha, belongs to nothing: the stale-pointer shape.
            "impostor": _identity(
                session,
                email="impostor@nowhere.test",
                organization_id=alpha.id,
                membership=False,
            ),
        }
        built.users = {name: user_id for name, (user_id, _) in people.items()}
        built.sessions = {name: token for name, (_, token) in people.items()}
        return built
    finally:
        session.close()


def _request(
    client: TestClient,
    world: _World,
    who: str,
    method: str,
    path: str,
    json: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.request(method, path, json=json)
        is_json = response.headers.get("content-type", "").startswith("application/json")
        body = response.json() if is_json and response.content else None
        return response.status_code, body
    finally:
        client.cookies.clear()


def _create(
    client: TestClient,
    world: _World,
    who: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return _request(client, world, who, "POST", _PREFIX, json=body or {"key_name": f"{who}-key"})


# =============================================================================
# Creating: ownership derived, workspace membership enforced
# =============================================================================


def test_a_member_creates_a_key_in_a_workspace_they_belong_to(client: TestClient, world: _World) -> None:
    code, body = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "mine", "workspace_id": str(world.workspaces["alpha_one"])},
    )
    assert code == status.HTTP_200_OK, body
    assert body["key"].strip()
    # Ownership is derived: the key bills to the caller's own attribution row.
    assert body["user_id"] == str(world.users["alpha_member"])
    # And is always budget-enforced; there is no field to say otherwise.
    assert body["exclude_from_budget"] is False

    # ``CreateKeyResponse`` carries no workspace, so where the key landed is
    # read back off the list.
    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK
    row = next(row for row in listed if row["id"] == body["id"])
    assert row["workspace_id"] == str(world.workspaces["alpha_one"])


def test_a_member_cannot_mint_into_a_sibling_workspace_they_do_not_belong_to(
    client: TestClient, world: _World
) -> None:
    """Alpha two is their organization's, so scoping to the organization alone would pass everything but this."""
    code, body = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "sneaky", "workspace_id": str(world.workspaces["alpha_two"])},
    )
    assert code == status.HTTP_404_NOT_FOUND, body


def test_a_member_cannot_mint_into_another_organizations_workspace(client: TestClient, world: _World) -> None:
    code, body = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "cross-tenant", "workspace_id": str(world.workspaces["beta_one"])},
    )
    assert code == status.HTTP_404_NOT_FOUND, body


def test_an_owner_may_mint_into_any_workspace_of_their_organization(client: TestClient, world: _World) -> None:
    """The management arm of the visibility rule, same as every other tenant-scoped surface."""
    code, body = _create(
        client,
        world,
        "alpha_owner",
        {"key_name": "owner-key", "workspace_id": str(world.workspaces["alpha_two"])},
    )
    assert code == status.HTTP_200_OK, body
    assert body["user_id"] == str(world.users["alpha_owner"])


def test_an_omitted_workspace_means_the_default_one_the_caller_belongs_to(
    client: TestClient, world: _World
) -> None:
    code, body = _create(client, world, "alpha_member", {"key_name": "defaulted"})
    assert code == status.HTTP_200_OK, body

    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK
    row = next(row for row in listed if row["id"] == body["id"])
    assert row["workspace_id"] == str(world.workspaces["alpha_one"])


def test_an_omitted_workspace_refuses_a_caller_outside_the_default_one(client: TestClient, world: _World) -> None:
    """A member of no workspace gets told what to do, not a 404 about a parameter they did not send."""
    code, body = _create(client, world, "alpha_newcomer", {"key_name": "nowhere"})
    assert code == status.HTTP_409_CONFLICT, body
    assert "workspace" in body["detail"]


def test_a_stale_organization_pointer_grants_nothing(client: TestClient, world: _World) -> None:
    """``active_organization_id`` without a live membership refuses, as everywhere on this surface."""
    code, body = _create(client, world, "impostor", {"key_name": "stolen"})
    assert code in {status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND}, body


def test_a_member_key_cannot_exceed_their_own_model_default(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The narrow-only allowed_models rule binds against the caller's own user row."""
    session = db_session_factory()
    try:
        session.add(BillingUser(user_id=str(world.users["alpha_member"]), allowed_models=["openai:gpt-4o"]))
        session.commit()
    finally:
        session.close()

    code, body = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "too-wide", "allowed_models": ["openai:*"]},
    )
    assert code == status.HTTP_400_BAD_REQUEST, body

    code, body = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "narrow-enough", "allowed_models": ["openai:gpt-4o"]},
    )
    assert code == status.HTTP_200_OK, body


# =============================================================================
# Listing: own keys, and nobody else's
# =============================================================================


def test_a_member_lists_their_own_keys_and_not_a_colleagues(client: TestClient, world: _World) -> None:
    """Both share Alpha one, so a workspace-scoped list would leak the colleague's key."""
    code, mine = _create(client, world, "alpha_member", {"key_name": "mine"})
    assert code == status.HTTP_200_OK
    code, theirs = _create(client, world, "alpha_colleague", {"key_name": "theirs"})
    assert code == status.HTTP_200_OK

    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK, listed
    ids = {row["id"] for row in listed}
    assert mine["id"] in ids
    assert theirs["id"] not in ids


def test_a_key_assigned_to_the_member_in_their_organization_is_listed(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """Ownership is the billing row, however the key was minted.

    Inserted directly rather than over ``POST /v1/keys``, because the operator
    acts in the deployment's default organization and cannot mint into alpha;
    what a handed-over key looks like is a row in the member's workspace billed
    to their attribution user. Dropping the owner predicate from the list query
    would still pass the colleague test (a different owner), so this is the
    positive half that pins the predicate itself.
    """
    handed_id = str(uuid.uuid4())
    session = db_session_factory()
    try:
        # The attribution row first: api_keys.user_id is a foreign key, and the
        # member has minted nothing yet in this test's fresh database.
        session.add(BillingUser(user_id=str(world.users["alpha_member"]), alias="member@alpha.test"))
        session.commit()
        session.add(
            APIKey(
                id=handed_id,
                workspace_id=world.workspaces["alpha_one"],
                key_hash=f"hash-{handed_id}",
                key_name="handed-over",
                user_id=str(world.users["alpha_member"]),
            )
        )
        session.commit()
    finally:
        session.close()

    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK
    assert handed_id in {row["id"] for row in listed}


def test_an_operator_minted_key_outside_the_organization_stays_off_the_list(
    client: TestClient, world: _World, master_key_header: dict[str, str]
) -> None:
    """The organization predicate holds even when the owner matches.

    The operator acts in the deployment's default organization, so a key they
    mint with the member's attribution id lands in a workspace outside alpha;
    the member's list is confined to their active organization and must not
    show it.
    """
    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK
    before = {row["id"] for row in listed}

    response = client.post(
        "/v1/keys",
        headers=master_key_header,
        json={"key_name": "handed-over", "user_id": str(world.users["alpha_member"])},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    handed = response.json()

    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK
    after = {row["id"] for row in listed}
    assert after == before
    assert handed["id"] not in after


def test_the_workspace_filter_narrows_and_never_widens(client: TestClient, world: _World) -> None:
    code, created = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "filtered", "workspace_id": str(world.workspaces["alpha_one"])},
    )
    assert code == status.HTTP_200_OK

    code, listed = _request(
        client, world, "alpha_member", "GET", f"{_PREFIX}?workspace_id={world.workspaces['alpha_one']}"
    )
    assert code == status.HTTP_200_OK
    assert created["id"] in {row["id"] for row in listed}

    # A workspace in another organization lists nothing rather than refusing,
    # matching the operator surface's filter semantics.
    code, listed = _request(
        client, world, "alpha_member", "GET", f"{_PREFIX}?workspace_id={world.workspaces['beta_one']}"
    )
    assert code == status.HTTP_200_OK
    assert listed == []


# =============================================================================
# Acting on a key: the owner predicate on every load
# =============================================================================


def test_a_member_updates_rotates_and_revokes_their_own_key(client: TestClient, world: _World) -> None:
    code, created = _create(client, world, "alpha_member", {"key_name": "lifecycle"})
    assert code == status.HTTP_200_OK

    code, updated = _request(
        client, world, "alpha_member", "PATCH", f"{_PREFIX}/{created['id']}", json={"key_name": "renamed"}
    )
    assert code == status.HTTP_200_OK, updated
    assert updated["key_name"] == "renamed"

    code, rotated = _request(client, world, "alpha_member", "POST", f"{_PREFIX}/{created['id']}/rotate")
    assert code == status.HTTP_200_OK, rotated
    assert rotated["id"] == created["id"]
    assert rotated["key"] != created["key"]

    code, _ = _request(client, world, "alpha_member", "DELETE", f"{_PREFIX}/{created['id']}")
    assert code == status.HTTP_204_NO_CONTENT

    code, listed = _request(client, world, "alpha_member", "GET", _PREFIX)
    assert code == status.HTTP_200_OK
    assert created["id"] not in {row["id"] for row in listed}


def test_an_update_with_null_clears_the_name_and_the_expiry(client: TestClient, world: _World) -> None:
    """Absent means unchanged and null means clear, the tri-state the dashboard's edit form sends."""
    code, created = _create(
        client,
        world,
        "alpha_member",
        {"key_name": "named", "expires_at": "2030-01-01T00:00:00+00:00"},
    )
    assert code == status.HTTP_200_OK

    # A body that names neither field changes neither.
    code, updated = _request(
        client, world, "alpha_member", "PATCH", f"{_PREFIX}/{created['id']}", json={"is_active": True}
    )
    assert code == status.HTTP_200_OK, updated
    assert updated["key_name"] == "named"
    assert updated["expires_at"] is not None

    code, updated = _request(
        client,
        world,
        "alpha_member",
        "PATCH",
        f"{_PREFIX}/{created['id']}",
        json={"key_name": None, "expires_at": None},
    )
    assert code == status.HTTP_200_OK, updated
    assert updated["key_name"] is None
    assert updated["expires_at"] is None


@pytest.mark.parametrize(
    ("method", "suffix"),
    [("PATCH", ""), ("POST", "/rotate"), ("DELETE", "")],
)
def test_a_colleagues_key_answers_the_404_a_missing_one_does(
    client: TestClient, world: _World, method: str, suffix: str
) -> None:
    """Same workspace, different owner: the id must be indistinguishable from one that does not exist."""
    code, theirs = _create(client, world, "alpha_colleague", {"key_name": "not-yours"})
    assert code == status.HTTP_200_OK

    json = {"key_name": "hijacked"} if method == "PATCH" else None
    code, body = _request(client, world, "alpha_member", method, f"{_PREFIX}/{theirs['id']}{suffix}", json=json)
    assert code == status.HTTP_404_NOT_FOUND, body


def test_an_update_cannot_exempt_the_key_from_budget(client: TestClient, world: _World) -> None:
    """The member body has no such field, and sending one anyway changes nothing."""
    code, created = _create(client, world, "alpha_member", {"key_name": "enforced"})
    assert code == status.HTTP_200_OK
    assert created["exclude_from_budget"] is False

    code, updated = _request(
        client,
        world,
        "alpha_member",
        "PATCH",
        f"{_PREFIX}/{created['id']}",
        json={"exclude_from_budget": True},
    )
    assert code == status.HTTP_200_OK, updated
    assert updated["exclude_from_budget"] is False


def test_a_create_cannot_name_an_owner_or_exempt_itself(client: TestClient, world: _World) -> None:
    """The escalation fields of the operator body are inert here, not honored."""
    code, body = _create(
        client,
        world,
        "alpha_member",
        {
            "key_name": "escalation",
            "user_id": "somebody-else",
            "exclude_from_budget": True,
        },
    )
    assert code == status.HTTP_200_OK, body
    assert body["user_id"] == str(world.users["alpha_member"])
    assert body["exclude_from_budget"] is False


# =============================================================================
# The control group: what must not have moved
# =============================================================================


def test_the_deployment_wide_router_still_refuses_a_member(client: TestClient, world: _World) -> None:
    """The gate otari-ai#1880 added stays; this surface is a sibling, not a loosening."""
    code, body = _request(client, world, "alpha_member", "GET", "/v1/keys")
    assert code == status.HTTP_403_FORBIDDEN, body
    code, body = _request(client, world, "alpha_member", "POST", "/v1/keys", json={"key_name": "nope"})
    assert code == status.HTTP_403_FORBIDDEN, body


def test_a_member_minted_key_authenticates_on_the_data_plane(client: TestClient, world: _World) -> None:
    """The key is a real credential, not a dashboard artifact.

    An unknown model on it earns the model resolver's own 400, which only a
    request that authenticated can reach; a rejected credential would have
    stopped at 401.
    """
    code, created = _create(client, world, "alpha_member", {"key_name": "live"})
    assert code == status.HTTP_200_OK

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created['key']}"},
        json={"model": "does-not-exist:nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert "Unknown or unsupported model" in response.json()["detail"]
