"""``/api/v1/users`` answers for the caller's organization and no other.

The router is deployment-wide and operator-only, which read as "so it may serve
every row" and did: ``list_users`` was an unfiltered select and the by-id routes
looked a user up by primary key alone, so an operator acting in one organization
read, re-budgeted and soft-deleted another's people, and ``POST /api/v1/keys``
took whatever ``user_id`` it was handed. ``/api/v1/keys`` had already settled the
same question the other way (otari#817), so the gap was between two routers
rather than in either one (otari-ai#2108).

``users`` carries no organization column, and could not: one identity's
attribution row serves every organization that person belongs to, and membership
is many-to-many. So the scope is derived, and this suite is written against the
derivation rather than against one route: a user is in reach through a key,
through usage, or through a roster row, and a user in reach of nothing anywhere
is shared.

Every assertion names the row that must be absent rather than counting what came
back, for the reason ``test_organization_usage_scope`` gives: a count matches by
accident, and "beta's user is not in this response" does not.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.entities import APIKey, DashboardSession, UsageLog, User
from gateway.models.tenancy import Organization, OrganizationMember, Workspace
from gateway.models.tenancy import User as TenancyUser
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

# The request-plane ids this suite reasons about. Each names the join that puts
# it in reach, so a failure says which half of the derivation broke.
ALPHA_KEYED = "alpha-ci-bot"
ALPHA_SPENDER = "alpha-retired-bot"
BETA_KEYED = "beta-ci-bot"
UNATTACHED = "just-created-bot"


@dataclass
class _World:
    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    # The attribution row each identity bills through: its UUID as a string,
    # which is the only thing joining the roster to ``users``.
    attribution: dict[str, str] = field(default_factory=dict)


def _operator(
    session: Session,
    *,
    email: str,
    organization_ids: tuple[uuid.UUID, ...],
) -> tuple[str, str]:
    """A superuser with a live session, on the roster of each organization given.

    Superuser because every route here is behind ``require_deployment_operator``:
    the question this suite asks is what a deployment operator sees, not whether
    a tenant may reach the router at all.

    Several organizations rather than one is not a curiosity: it is the case a
    ``users.organization_id`` column could not have represented, so the roster
    join has to answer it in each of them.
    """
    person = TenancyUser(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_ids[0],
        is_superuser=True,
    )
    session.add(person)
    session.commit()
    session.refresh(person)

    for organization_id in organization_ids:
        session.add(
            OrganizationMember(
                organization_id=organization_id,
                user_id=person.id,
                role="owner",
                status="active",
            )
        )
    # The attribution row the roster names. Written here rather than through the
    # service, so the suite states the join it is testing.
    session.add(User(user_id=str(person.id), alias=email))

    token = f"otari-sess-{email}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=person.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return str(person.id), token


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """Two tenants, a user reachable by each join, and one reachable by none."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK

    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha")
        beta = Organization(name="Beta", slug="beta")
        session.add_all([alpha, beta])
        session.commit()
        session.refresh(alpha)
        session.refresh(beta)

        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, beta_one])
        session.commit()
        session.refresh(alpha_one)
        session.refresh(beta_one)

        session.add_all(
            [
                User(user_id=ALPHA_KEYED, alias="Alpha CI"),
                User(user_id=ALPHA_SPENDER, alias="Alpha retired"),
                User(user_id=BETA_KEYED, alias="Beta CI"),
                User(user_id=UNATTACHED, alias="Just created"),
            ]
        )
        session.commit()

        # Reach by key.
        session.add_all(
            [
                APIKey(
                    id=str(uuid.uuid4()),
                    workspace_id=alpha_one.id,
                    key_hash=f"hash-{ALPHA_KEYED}",
                    key_prefix="gw-alpha",
                    user_id=ALPHA_KEYED,
                ),
                APIKey(
                    id=str(uuid.uuid4()),
                    workspace_id=beta_one.id,
                    key_hash=f"hash-{BETA_KEYED}",
                    key_prefix="gw-beta",
                    user_id=BETA_KEYED,
                ),
            ]
        )
        # Reach by usage alone: every key this owner had is gone, and the spend
        # a budget is enforcing against is not.
        session.add(
            UsageLog(
                id=str(uuid.uuid4()),
                workspace_id=alpha_one.id,
                user_id=ALPHA_SPENDER,
                model="m",
                provider="p",
                endpoint="/v1/chat/completions",
                source="gateway",
                status="success",
                total_tokens=10,
                timestamp=datetime.now(UTC),
            )
        )
        session.commit()

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "beta_one": beta_one.id}
        people = {
            "alpha_operator": _operator(session, email="op@alpha.test", organization_ids=(alpha.id,)),
            "beta_operator": _operator(session, email="op@beta.test", organization_ids=(beta.id,)),
            # The multi-org case: one identity, one attribution row, two rosters.
            "consultant": _operator(session, email="both@example.test", organization_ids=(alpha.id, beta.id)),
        }
        built.attribution = {name: attribution for name, (attribution, _) in people.items()}
        built.sessions = {name: token for name, (_, token) in people.items()}
        return built
    finally:
        session.close()


def _listed(client: TestClient, world: _World, who: str) -> set[str]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.get(f"{API_ROOT}/users?limit=1000")
        assert response.status_code == status.HTTP_200_OK, response.text
        return {row["user_id"] for row in response.json()}
    finally:
        client.cookies.clear()


def _request(client: TestClient, world: _World, who: str, method: str, path: str, **kwargs: object) -> int:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        return client.request(method, path, **kwargs).status_code  # type: ignore[arg-type]
    finally:
        client.cookies.clear()


# =============================================================================
# What the list answers
# =============================================================================


def test_an_operator_lists_the_users_their_own_organization_can_name(client: TestClient, world: _World) -> None:
    listed = _listed(client, world, "alpha_operator")
    assert ALPHA_KEYED in listed
    assert ALPHA_SPENDER in listed, "usage is a join too: a revoked key must not hide a ledger"
    assert world.attribution["alpha_operator"] in listed, "a member with no key is still their organization's"


def test_an_operator_lists_no_other_organizations_people(client: TestClient, world: _World) -> None:
    """The claim the whole change rests on, stated on its own."""
    listed = _listed(client, world, "alpha_operator")
    assert BETA_KEYED not in listed
    assert world.attribution["beta_operator"] not in listed


def test_a_user_reachable_from_nowhere_is_shared_rather_than_hidden(client: TestClient, world: _World) -> None:
    """Otherwise a just-created user is invisible on the page that budgets it.

    It is also what keeps the shared ``default`` owner listed, which every key
    minted with no owner bills through.
    """
    assert UNATTACHED in _listed(client, world, "alpha_operator")
    assert UNATTACHED in _listed(client, world, "beta_operator")


def test_one_identity_is_listed_in_every_organization_it_belongs_to(client: TestClient, world: _World) -> None:
    """The case a single ``users.organization_id`` column could not have held.

    One attribution row serves the consultant in both tenants, so a scope that
    picked one organization for the row would have hidden them in the other.
    """
    consultant = world.attribution["consultant"]
    assert consultant in _listed(client, world, "alpha_operator")
    assert consultant in _listed(client, world, "beta_operator")


def test_a_header_master_key_acts_in_the_default_organization(
    client: TestClient, world: _World, master_key_header: dict[str, str]
) -> None:
    """The deployment credential is scoped too, which is a real narrowing.

    A header master key names nobody, so it resolves the bootstrap operator and
    acts in the *default* organization rather than in all of them. That is the
    rule ``/api/v1/keys`` already follows (otari#817) and the reason this router
    now matches it, but it means the master key alone no longer administers a
    user in a tenant it did not provision: that is a session's to do, after
    switching into the organization.
    """
    response = client.get(f"{API_ROOT}/users?limit=1000", headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    listed = {row["user_id"] for row in response.json()}

    assert ALPHA_KEYED not in listed
    assert BETA_KEYED not in listed
    # Shared, so the credential that mints the first key can still see its owner.
    assert UNATTACHED in listed


# =============================================================================
# The by-id routes answer 404 outside the scope
# =============================================================================


@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [
        ("GET", "", None),
        ("PATCH", "", {"alias": "renamed"}),
        ("DELETE", "", None),
        ("GET", "/usage", None),
    ],
)
def test_every_by_id_route_refuses_another_organizations_user(
    client: TestClient, world: _World, method: str, suffix: str, body: dict[str, str] | None
) -> None:
    """Parametrized because a scope applied to the read and forgotten on the
    write would leave an operator able to block or re-budget a tenant they
    cannot see."""
    kwargs = {"json": body} if body is not None else {}
    assert (
        _request(client, world, "alpha_operator", method, f"{API_ROOT}/users/{BETA_KEYED}{suffix}", **kwargs)
        == status.HTTP_404_NOT_FOUND
    )


def test_the_same_routes_still_serve_the_callers_own_user(client: TestClient, world: _World) -> None:
    """The control: 404 has to mean out of scope, not broken."""
    assert (
        _request(client, world, "alpha_operator", "GET", f"{API_ROOT}/users/{ALPHA_KEYED}") == status.HTTP_200_OK
    )
    assert (
        _request(client, world, "beta_operator", "GET", f"{API_ROOT}/users/{BETA_KEYED}") == status.HTTP_200_OK
    )


# =============================================================================
# The write side
# =============================================================================


def test_a_key_cannot_be_minted_for_another_organizations_owner(client: TestClient, world: _World) -> None:
    """Scoping the read alone would leave the owner nameable by a client.

    The refusal is the 404 an unknown id gets, so it reports no more than the
    list does about which ids the other tenant holds.
    """
    assert (
        _request(
            client,
            world,
            "alpha_operator",
            "POST",
            f"{API_ROOT}/keys",
            json={"user_id": BETA_KEYED, "workspace_id": str(world.workspaces["alpha_one"])},
        )
        == status.HTTP_404_NOT_FOUND
    )


def test_a_key_for_an_owner_in_scope_or_a_new_id_still_mints(client: TestClient, world: _World) -> None:
    for owner in (ALPHA_KEYED, UNATTACHED, "brand-new-bot"):
        assert (
            _request(
                client,
                world,
                "alpha_operator",
                "POST",
                f"{API_ROOT}/keys",
                json={"user_id": owner, "workspace_id": str(world.workspaces["alpha_one"])},
            )
            == status.HTTP_200_OK
        ), owner
