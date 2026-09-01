"""``GET /v1/models`` shows a tenant only the providers their organization reaches.

The roles matrix wants a member's model list narrowed to the providers they have
access to (otari-ai#1969). The narrowing reuses the allow-list machinery an API
key already goes through (``services/model_access``), so the assertions here are
about *which* allow-list a caller is answered by rather than about a second
matcher:

* a header master key is the deployment credential and is unrestricted;
* a session that operates the deployment is unrestricted;
* any other session is answered by its membership, which is every
  ``config.providers`` instance (deployment-wide, so every tenant reaches them)
  plus the organization's own BYO providers.

The test deployment configures no ``providers:`` block, so every entry a caller
is shown here comes from a BYO key. That is the sharp case: a deployment with
config-file providers gives every tenant those on top, which is why opening this
filter changes nothing for a single-tenant install.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import DashboardSession
from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderModelRestriction
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

# Priced but undiscovered models: phase 2 of the listing publishes them, so the
# catalog is deterministic without dialing a provider.
_OPENAI_MODEL = "openai:gpt-4o-mini"
_OPENAI_OTHER = "openai:gpt-4o"
_ANTHROPIC_MODEL = "anthropic:claude-3-5-haiku-latest"
_MISTRAL_MODEL = "mistral:mistral-small-latest"
_ALL_MODELS = (_OPENAI_MODEL, _OPENAI_OTHER, _ANTHROPIC_MODEL, _MISTRAL_MODEL)


@dataclass
class _World:
    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    keys: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    is_superuser: bool = False,
    workspace_ids: tuple[uuid.UUID, ...] = (),
) -> str:
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
        OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active")
    )
    for workspace_id in workspace_ids:
        session.add(
            WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role="member", status="active")
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


def _unmembered_identity(session: Session, *, email: str, organization_id: uuid.UUID) -> str:
    """A signed-in identity pointed at a real organization it holds no membership in.

    ``users.active_organization_id`` is a foreign key, so a pointer at nothing is
    not a state the database can hold; a pointer with no membership behind it is,
    and it is the one the scope resolver refuses.
    """
    user = User(email=email, full_name="Orphan", active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
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


def _byo_key(session: Session, *, organization_id: uuid.UUID, provider: str) -> uuid.UUID:
    key = OrgProviderKey(
        organization_id=organization_id,
        provider=provider,
        name=f"{provider}-primary",
        encrypted_api_key="encrypted",
        last4="1234",
        is_org_default=True,
    )
    session.add(key)
    session.commit()
    session.refresh(key)
    return key.id


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """Two tenants with different BYO providers, and a priced catalog spanning three."""
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK
    for model_key in _ALL_MODELS:
        priced = client.post(
            "/v1/pricing",
            json={"model_key": model_key, "input_price_per_million": 1.0, "output_price_per_million": 2.0},
            headers=master_key_header,
        )
        assert priced.status_code == status.HTTP_200_OK, priced.text

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

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "alpha_two": alpha_two.id, "beta_one": beta_one.id}
        built.keys = {
            "alpha_openai": _byo_key(session, organization_id=alpha.id, provider="openai"),
            "beta_anthropic": _byo_key(session, organization_id=beta.id, provider="anthropic"),
        }
        built.sessions = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            "alpha_newcomer": _identity(session, email="new@alpha.test", organization_id=alpha.id),
            "beta_member": _identity(
                session,
                email="member@beta.test",
                organization_id=beta.id,
                workspace_ids=(beta_one.id,),
            ),
            "orphan": _unmembered_identity(
                session, email="orphan@nowhere.test", organization_id=beta.id
            ),
            "superuser": _identity(
                session,
                email="root@alpha.test",
                organization_id=alpha.id,
                role="owner",
                is_superuser=True,
            ),
        }
        return built
    finally:
        session.close()


def _catalog_as(client: TestClient, world: _World, who: str) -> set[str]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.get("/v1/models")
        assert response.status_code == status.HTTP_200_OK, response.text
        return {model["id"] for model in response.json()["data"]}
    finally:
        client.cookies.clear()


def test_the_master_key_still_sees_every_priced_model(
    client: TestClient, master_key_header: dict[str, str], world: _World
) -> None:
    """The control: the deployment credential is not narrowed by anyone's membership."""
    response = client.get("/v1/models", headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    assert {model["id"] for model in response.json()["data"]} == set(_ALL_MODELS)


def test_a_member_sees_only_their_organizations_providers(client: TestClient, world: _World) -> None:
    listed = _catalog_as(client, world, "alpha_member")
    assert listed == {_OPENAI_MODEL, _OPENAI_OTHER}
    assert _ANTHROPIC_MODEL not in listed
    assert _MISTRAL_MODEL not in listed


def test_two_tenants_are_shown_disjoint_catalogs(client: TestClient, world: _World) -> None:
    """Stated on its own, because it is the claim the narrowing rests on."""
    assert _catalog_as(client, world, "alpha_member").isdisjoint(_catalog_as(client, world, "beta_member"))


def test_an_admin_is_answered_from_the_organizations_providers_not_one_workspaces(
    client: TestClient, world: _World
) -> None:
    """An owner belongs to no workspace here, and still reads the organization's providers.

    An implementation that walked the caller's workspace memberships would show
    them nothing, which is the failure this pins.
    """
    assert _catalog_as(client, world, "alpha_owner") == {_OPENAI_MODEL, _OPENAI_OTHER}


def test_a_member_of_no_workspace_sees_no_byo_models_rather_than_a_refusal(
    client: TestClient, world: _World
) -> None:
    """Nothing was refused; no workspace of theirs holds a key yet."""
    assert _catalog_as(client, world, "alpha_newcomer") == set()


def test_a_deployment_operator_session_is_not_narrowed(client: TestClient, world: _World) -> None:
    """A superuser operates the deployment, so the catalog is the deployment's."""
    assert _catalog_as(client, world, "superuser") == set(_ALL_MODELS)


def test_a_workspace_model_restriction_narrows_a_members_catalog(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The allow-list a workspace already enforces at dispatch decides the listing too."""
    session = db_session_factory()
    try:
        session.add(
            WorkspaceProviderModelRestriction(
                workspace_id=world.workspaces["alpha_one"],
                organization_id=world.alpha,
                org_provider_key_id=world.keys["alpha_openai"],
                model="gpt-4o-mini",
            )
        )
        session.commit()
    finally:
        session.close()

    assert _catalog_as(client, world, "alpha_member") == {_OPENAI_MODEL}
    # The admin still reads the organization's whole provider, because lifting one
    # workspace's restriction is theirs to do.
    assert _catalog_as(client, world, "alpha_owner") == {_OPENAI_MODEL, _OPENAI_OTHER}


def test_a_single_model_read_agrees_with_the_listing(client: TestClient, world: _World) -> None:
    """A model withheld from the listing is 404 by id, never a different answer."""
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    try:
        assert client.get(f"/v1/models/{_OPENAI_MODEL}").status_code == status.HTTP_200_OK
        assert client.get(f"/v1/models/{_ANTHROPIC_MODEL}").status_code == status.HTTP_404_NOT_FOUND
    finally:
        client.cookies.clear()


def test_an_identity_with_no_live_membership_is_answered_rather_than_refused(
    client: TestClient, world: _World
) -> None:
    """A catalog read is not one of the routes whose whole question is "which organization".

    The pointer is not the authority, so this caller reaches no organization's
    keys, which is an empty list here (the test deployment configures no
    ``providers:`` block) rather than a 403 over a page that used to render. In
    particular it must not be shown Beta's models just because it points there.
    """
    assert _catalog_as(client, world, "orphan") == set()
