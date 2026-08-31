"""The organization-scoped usage reads see the caller's own organization and no more.

``/v1/usage`` is deployment-wide and operator-only since #821. These routes are
the tenant's half of it (otari#837), and the whole of their correctness is that
the row set is decided by the caller's membership rather than by anything the
request carries. So the suite is written against a world holding *two*
organizations with traffic in both, and every assertion names the rows that must
be absent rather than counting the ones that came back: a count can match by
accident, and "beta's model is not in this response" cannot.

Four things are asserted for each of the four routes:

- an owner reads every workspace in their organization, and none outside it;
- a member reads the workspaces they belong to, and not the sibling workspace
  they do not;
- ``workspace_id`` narrows and never widens, answering 404 outside the scope the
  way a workspace that does not exist does;
- the scope follows the *membership*, not the ``active_organization_id`` pointer,
  so pointing an identity at an organization it does not belong to grants
  nothing.

The last group is the control: the deployment-wide route must still refuse an
organization owner. A change that made these pass by loosening that gate would
have reopened otari-ai#1880, so it is asserted here as well as in
``test_deployment_operator_gate.py``.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.models.entities import DashboardSession, UsageLog
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

# Every read this router serves. Parametrized rather than asserted once, because
# a scope applied to the list and forgotten on the summary would leak an
# organization's spend through its aggregates while the log looked correct.
_SCOPED_PATHS = [
    "/v1/organizations/me/usage",
    "/v1/organizations/me/usage/count",
    "/v1/organizations/me/usage/summary",
    "/v1/organizations/me/usage/series?group_by=model",
]


@dataclass
class _World:
    """Two organizations, their workspaces, and one session cookie per identity."""

    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    users: dict[str, uuid.UUID] = field(default_factory=dict)


# Model names double as row identity: every assertion below is "these models and
# no others", which is what makes a leak visible rather than merely miscounted.
_ALPHA_ONE_MODELS = ("alpha-one-a", "alpha-one-b")
_ALPHA_TWO_MODELS = ("alpha-two-a",)
_BETA_MODELS = ("beta-one-a", "beta-one-b")


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    is_superuser: bool = False,
    workspace_ids: tuple[uuid.UUID, ...] = (),
    membership: bool = True,
) -> tuple[uuid.UUID, str]:
    """Create an identity with a live dashboard session, and return its cookie.

    ``membership=False`` builds the case the pointer test needs: an identity
    whose ``active_organization_id`` names an organization it holds no membership
    in, which is the shape a stolen or stale pointer would have.
    """
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
    return user.id, token


def _usage_rows(session: Session, workspace_id: uuid.UUID, models: tuple[str, ...]) -> None:
    for model in models:
        session.add(
            UsageLog(
                id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                model=model,
                provider="p",
                endpoint="/v1/chat/completions",
                source="gateway",
                status="success",
                total_tokens=10,
                timestamp=datetime.now(UTC),
            )
        )
    session.commit()


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """Two tenants with traffic in both, and the identities that read them."""
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

        _usage_rows(session, alpha_one.id, _ALPHA_ONE_MODELS)
        _usage_rows(session, alpha_two.id, _ALPHA_TWO_MODELS)
        _usage_rows(session, beta_one.id, _BETA_MODELS)

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "alpha_two": alpha_two.id, "beta_one": beta_one.id}
        people = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            "alpha_admin": _identity(session, email="admin@alpha.test", organization_id=alpha.id, role="admin"),
            # Belongs to one of alpha's two workspaces, which is the case the
            # member arm exists for.
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            # The fourth role, and outside MANAGEMENT_ROLES like ``member``, so
            # it must take the workspace branch rather than the whole-tenant one.
            "alpha_viewer": _identity(
                session,
                email="viewer@alpha.test",
                organization_id=alpha.id,
                role="viewer",
                workspace_ids=(alpha_one.id,),
            ),
            # In the organization, in none of its workspaces.
            "alpha_newcomer": _identity(session, email="new@alpha.test", organization_id=alpha.id),
            "beta_owner": _identity(session, email="owner@beta.test", organization_id=beta.id, role="owner"),
            # Points at alpha, belongs to nothing.
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
        built.users = {name: user_id for name, (user_id, _) in people.items()}
        built.sessions = {name: token for name, (_, token) in people.items()}
        return built
    finally:
        session.close()


def _as(client: TestClient, world: _World, who: str, path: str) -> tuple[int, object]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.get(path)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
        return response.status_code, body
    finally:
        client.cookies.clear()


def _models_listed(client: TestClient, world: _World, who: str, query: str = "") -> set[str]:
    code, body = _as(client, world, who, f"/v1/organizations/me/usage{query}")
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, list)
    return {row["model"] for row in body}


def _models_summarized(client: TestClient, world: _World, who: str, query: str = "") -> set[str]:
    code, body = _as(client, world, who, f"/v1/organizations/me/usage/summary{query}")
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, dict)
    return {row["key"] for row in body["by_model"] if not row.get("is_other")}


# =============================================================================
# How much of the organization each role reads
# =============================================================================


@pytest.mark.parametrize("who", ["alpha_owner", "alpha_admin"])
def test_an_owner_or_admin_reads_every_workspace_in_their_own_organization(
    client: TestClient, world: _World, who: str
) -> None:
    """The management arm: the whole tenant, and nothing outside it."""
    assert _models_listed(client, world, who) == set(_ALPHA_ONE_MODELS) | set(_ALPHA_TWO_MODELS)


@pytest.mark.parametrize("who", ["alpha_owner", "alpha_admin"])
def test_an_owner_or_admin_reads_no_other_organizations_rows(client: TestClient, world: _World, who: str) -> None:
    """Stated on its own, because it is the claim the whole change rests on."""
    assert _models_listed(client, world, who).isdisjoint(_BETA_MODELS)
    assert _models_summarized(client, world, who).isdisjoint(_BETA_MODELS)


def test_a_member_reads_only_the_workspaces_they_belong_to(client: TestClient, world: _World) -> None:
    """The member arm, and the sibling workspace in the same organization is the point.

    ``alpha_member`` belongs to Alpha one and not Alpha two. Both are their
    organization's, so an implementation that scoped to the organization alone
    would pass every other test in this file and fail this one.
    """
    listed = _models_listed(client, world, "alpha_member")
    assert listed == set(_ALPHA_ONE_MODELS)
    assert listed.isdisjoint(_ALPHA_TWO_MODELS)
    assert listed.isdisjoint(_BETA_MODELS)


def test_a_viewer_is_scoped_like_a_member_and_not_like_an_admin(client: TestClient, world: _World) -> None:
    """``viewer`` is the fourth role and is outside ``MANAGEMENT_ROLES``.

    Pinned separately from ``member`` because the two reach the workspace branch
    through different values, and a scope that tested for ``member`` by name
    rather than for management would hand a viewer the whole organization.
    """
    listed = _models_listed(client, world, "alpha_viewer")
    assert listed == set(_ALPHA_ONE_MODELS)
    assert listed.isdisjoint(_ALPHA_TWO_MODELS)


def test_a_member_of_no_workspace_reads_an_empty_page_rather_than_a_refusal(client: TestClient, world: _World) -> None:
    """Nothing was refused; there is simply nothing here yet."""
    code, body = _as(client, world, "alpha_newcomer", "/v1/organizations/me/usage")
    assert code == status.HTTP_200_OK, body
    assert body == []

    code, body = _as(client, world, "alpha_newcomer", "/v1/organizations/me/usage/count")
    assert code == status.HTTP_200_OK, body
    assert body == {"total": 0}


def test_a_member_of_no_workspace_gets_empty_aggregates_rather_than_an_error(client: TestClient, world: _World) -> None:
    """The empty scope reaches a ``GROUP BY`` rather than a row filter here.

    ``/summary`` and ``/series`` fold their breakdowns over the same conditions,
    so a scope of "nothing" has to produce zeroed totals and empty groups rather
    than a division by zero or a fold over an empty top-N set.
    """
    code, body = _as(client, world, "alpha_newcomer", "/v1/organizations/me/usage/summary")
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, dict)
    assert body["totals"]["request_count"] == 0
    assert body["totals"]["cost"] == 0
    assert body["by_model"] == []
    assert body["series"] == []

    for group_by in ("model", "user_id", "api_key_id", "source"):
        code, body = _as(client, world, "alpha_newcomer", f"/v1/organizations/me/usage/series?group_by={group_by}")
        assert code == status.HTTP_200_OK, (group_by, body)
        assert isinstance(body, dict)
        assert body["groups"] == [], group_by
        assert body["points"] == [], group_by


def test_a_suspended_workspace_membership_stops_granting_the_workspace(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The scope reads active memberships only, so suspending one takes the rows back.

    Distinct from removing the organization membership, which would refuse the
    whole surface: this caller is still a member of Alpha and still reads it,
    with one workspace fewer in it.
    """
    assert _models_listed(client, world, "alpha_member") == set(_ALPHA_ONE_MODELS)

    session = db_session_factory()
    try:
        membership = (
            session.query(WorkspaceMember)
            .filter(col(WorkspaceMember.workspace_id) == world.workspaces["alpha_one"])
            .filter(col(WorkspaceMember.user_id) == world.users["alpha_member"])
            .one()
        )
        membership.status = "suspended"
        session.add(membership)
        session.commit()
    finally:
        session.close()

    assert _models_listed(client, world, "alpha_member") == set()
    # Still their organization: an empty page, not a refusal.
    code, _ = _as(client, world, "alpha_member", "/v1/organizations/me/usage")
    assert code == status.HTTP_200_OK


def test_a_superuser_reads_their_active_organization_and_not_every_tenant(client: TestClient, world: _World) -> None:
    """This router is scoped even for an operator; ``/v1/usage`` is where they read across tenants."""
    listed = _models_listed(client, world, "superuser")
    assert listed == set(_BETA_MODELS)
    assert listed.isdisjoint(_ALPHA_ONE_MODELS)


# =============================================================================
# The workspace filter narrows and cannot widen
# =============================================================================


def test_a_workspace_filter_inside_the_scope_narrows_the_read(client: TestClient, world: _World) -> None:
    query = f"?workspace_id={world.workspaces['alpha_two']}"
    assert _models_listed(client, world, "alpha_owner", query) == set(_ALPHA_TWO_MODELS)


@pytest.mark.parametrize("path", _SCOPED_PATHS)
def test_another_organizations_workspace_is_not_found_on_every_route(
    client: TestClient, world: _World, path: str
) -> None:
    """404 rather than 403 or an empty 200: it must read like a workspace that does not exist."""
    joiner = "&" if "?" in path else "?"
    code, _ = _as(client, world, "alpha_owner", f"{path}{joiner}workspace_id={world.workspaces['beta_one']}")
    assert code == status.HTTP_404_NOT_FOUND


@pytest.mark.parametrize("path", _SCOPED_PATHS)
def test_a_workspace_the_member_does_not_belong_to_is_not_found(client: TestClient, world: _World, path: str) -> None:
    """Alpha two is the caller's own organization's, and still not theirs to name."""
    joiner = "&" if "?" in path else "?"
    code, _ = _as(client, world, "alpha_member", f"{path}{joiner}workspace_id={world.workspaces['alpha_two']}")
    assert code == status.HTTP_404_NOT_FOUND


# =============================================================================
# The scope follows the membership, not the pointer
# =============================================================================


@pytest.mark.parametrize("path", _SCOPED_PATHS)
def test_an_active_organization_pointer_with_no_membership_behind_it_reads_nothing(
    client: TestClient, world: _World, path: str
) -> None:
    """The pointer is not the authority; the membership is.

    ``impostor`` has ``active_organization_id`` set to Alpha and no
    ``OrganizationMember`` row anywhere. If the scope were derived from the
    pointer alone this would return Alpha's traffic.
    """
    code, body = _as(client, world, "impostor", path)
    assert code in {status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND}, body


def test_switching_organizations_is_what_moves_the_scope(client: TestClient, world: _World) -> None:
    """And it refuses an organization the caller does not belong to, so it is not a way in."""
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_owner"])
    try:
        refused = client.post("/v1/organizations/me/switch", json={"organization_id": str(world.beta)})
        assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text
        # And the read is unmoved by the attempt.
        listed = {row["model"] for row in client.get("/v1/organizations/me/usage").json()}
        assert listed.isdisjoint(_BETA_MODELS)
    finally:
        client.cookies.clear()


# =============================================================================
# The aggregates carry the same scope as the log
# =============================================================================


def test_the_count_never_describes_more_rows_than_the_list_returns(client: TestClient, world: _World) -> None:
    for who, expected in (
        ("alpha_owner", len(_ALPHA_ONE_MODELS) + len(_ALPHA_TWO_MODELS)),
        ("alpha_member", len(_ALPHA_ONE_MODELS)),
        ("beta_owner", len(_BETA_MODELS)),
    ):
        code, body = _as(client, world, who, "/v1/organizations/me/usage/count")
        assert code == status.HTTP_200_OK, body
        assert body == {"total": expected}, who


def test_the_summary_totals_count_only_the_callers_own_rows(client: TestClient, world: _World) -> None:
    code, body = _as(client, world, "alpha_member", "/v1/organizations/me/usage/summary")
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, dict)
    assert body["totals"]["request_count"] == len(_ALPHA_ONE_MODELS)
    assert _models_summarized(client, world, "alpha_member") == set(_ALPHA_ONE_MODELS)


def test_the_series_splits_only_the_callers_own_rows(client: TestClient, world: _World) -> None:
    code, body = _as(client, world, "alpha_member", "/v1/organizations/me/usage/series?group_by=model")
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, dict)
    keys = {group["key"] for group in body["groups"] if not group.get("is_other")}
    assert keys == set(_ALPHA_ONE_MODELS)


# =============================================================================
# Controls: the deployment-wide route is untouched
# =============================================================================


def test_an_organization_owner_is_still_refused_by_the_deployment_wide_usage_route(
    client: TestClient, world: _World
) -> None:
    """The gate #821 added stays where it is. This is the regression that would matter most."""
    code, _ = _as(client, world, "alpha_owner", "/v1/usage")
    assert code == status.HTTP_403_FORBIDDEN


def test_the_master_key_still_reads_every_tenant_through_the_deployment_wide_route(
    client: TestClient, world: _World, master_key_header: dict[str, str]
) -> None:
    listed = {row["model"] for row in client.get("/v1/usage", headers=master_key_header).json()}
    assert set(_ALPHA_ONE_MODELS) | set(_BETA_MODELS) <= listed


@pytest.mark.parametrize("path", _SCOPED_PATHS)
def test_an_unauthenticated_request_is_refused(client: TestClient, world: _World, path: str) -> None:
    """No credential is a 401, so the scope is never reached with nobody behind it."""
    assert client.get(path).status_code == status.HTTP_401_UNAUTHORIZED


# =============================================================================
# The organization-context contract two sibling issues consume
# =============================================================================


def test_the_context_reports_whether_the_caller_operates_the_deployment(client: TestClient, world: _World) -> None:
    """`deployment_operator` is the answer `GET /v1/admin/access` gives, on a read the shell already makes.

    Carried here so the sidebar has no window in which it shows a row it is about
    to retract (otari#836), and so the roster can say which authority its role
    picker sets (otari#838). Asserted against both answers, because a field that
    is always true is indistinguishable from one nothing computes.
    """
    for who, expected in (("alpha_owner", False), ("superuser", True)):
        code, body = _as(client, world, who, "/v1/organizations/me")
        assert code == status.HTTP_200_OK, body
        assert isinstance(body, dict)
        assert body["deployment_operator"] is expected, who


def test_the_context_agrees_with_the_admin_access_endpoint(client: TestClient, world: _World) -> None:
    """Two publishers of one predicate, which is only safe while they cannot disagree."""
    for who in ("alpha_owner", "superuser"):
        _, context = _as(client, world, who, "/v1/organizations/me")
        _, access = _as(client, world, who, "/v1/admin/access")
        assert isinstance(context, dict)
        assert isinstance(access, dict)
        assert context["deployment_operator"] is access["granted"], who


def test_every_response_carrying_the_context_carries_the_operator_answer(
    client: TestClient, world: _World
) -> None:
    """`POST /me/switch` and `PATCH /me` return the shape too, not just `GET /me`.

    The dashboard keeps whichever context it saw last, so a write answering a
    stale `false` would strip an operator's sidebar until the next reload, and a
    stale `true` would leave a member looking at rows the server refuses. Both
    identities are asserted for the reason the read above gives: a field that is
    always true is indistinguishable from one nothing computes.

    Each caller switches to the organization it is already in, which the service
    admits on purpose ("a switcher that re-sent the current row should not have
    to be told off for it") and which keeps this test out of the membership
    graph the rest of the file is about.
    """
    for who, organization, expected in (
        ("alpha_owner", world.alpha, False),
        ("superuser", world.beta, True),
    ):
        client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
        try:
            switched = client.post(
                "/v1/organizations/me/switch",
                json={"organization_id": str(organization)},
            )
            renamed = client.patch("/v1/organizations/me", json={"name": f"Renamed by {who}"})
        finally:
            client.cookies.clear()

        assert switched.status_code == status.HTTP_200_OK, switched.text
        assert renamed.status_code == status.HTTP_200_OK, renamed.text
        assert switched.json()["deployment_operator"] is expected, who
        assert renamed.json()["deployment_operator"] is expected, who


def test_the_context_names_the_caller_it_describes(client: TestClient, world: _World) -> None:
    """The identity behind the standing, which is what a shell draws a person from.

    Nothing else on this contract says *who* is signed in, so the dashboard's
    account control had no name to show and showed a role instead (otari#832).
    Asserted per identity rather than for shape alone: a field filled from the
    wrong identity is the failure that matters here, and one filled from a
    constant would satisfy a shape check.
    """
    for who, email in (
        ("alpha_owner", "owner@alpha.test"),
        ("alpha_member", "member@alpha.test"),
        ("superuser", "root@beta.test"),
    ):
        code, body = _as(client, world, who, "/v1/organizations/me")
        assert code == status.HTTP_200_OK, body
        assert isinstance(body, dict)
        caller = body["caller"]
        assert caller["user_id"] == str(world.users[who]), who
        assert caller["email"] == email, who
        # What `_identity` gives every one of them: the address's local part.
        assert caller["full_name"] == email.split("@")[0].title(), who


def test_the_context_names_an_identity_that_holds_no_address(
    client: TestClient, world: _World, master_key_header: dict[str, str]
) -> None:
    """The local operator, which is the identity a standalone first boot leaves behind.

    It has a name and no email, so a shell that assumed an address would have
    nothing to draw for the one caller every standalone deployment has.
    """
    response = client.get("/v1/organizations/me", headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    caller = response.json()["caller"]
    assert caller["email"] is None
    assert caller["full_name"]


def test_the_context_reports_whether_provider_keys_can_be_encrypted(client: TestClient, world: _World) -> None:
    """A deployment fact a tenant is allowed to know, because it decides whether their own write works.

    The only endpoint that reported it is operator-gated, so the provider-keys
    page inferred it from a refused request and told an owner to go set an
    environment variable that was already set (otari#839). Readable by a member
    here, which is the whole point.
    """
    code, body = _as(client, world, "alpha_member", "/v1/organizations/me")
    assert code == status.HTTP_200_OK, body
    assert isinstance(body, dict)
    assert isinstance(body["provider_key_encryption_available"], bool)
    # Says whether a key is configured, never anything about its value.
    assert "secret" not in str(body).lower()
