"""An organization admin edits their own routing entries, and nobody else's.

The View half of the Build pages landed in mozilla-ai/otari#867; this is the
Edit half the roles matrix asks for (otari-ai#1969). ``routing_policies`` and
``model_aliases`` both carry a non-nullable ``workspace_id`` already, so the
tenant-scoped writers here need no owner column: a workspace belongs to exactly
one organization, and that is the row's tenant.

Four refusals carry the whole surface, and each has a test that fails on its own
if the check is dropped: a member may not write at all, another tenant's
workspace is not found rather than forbidden, a write must name a workspace, and
a target the organization cannot already reach is refused. The last is the one
that makes opening these verbs safe: a policy decides which real model a name
resolves to, so an unconstrained write would be a way to name a model the tenant
holds no provider key for.

The control at the end is the same as the read suite's: the deployment-wide
routers must still refuse a tenant, or a change here would have reopened
otari-ai#1880.
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

from gateway.models.entities import DashboardSession
from gateway.models.provider_keys import OrgProviderKey
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

_POLICIES = "/v1/organizations/me/routing-policies"
_ALIASES = "/v1/organizations/me/aliases"

# Alpha holds an OpenAI key and Beta an Anthropic one, so each tenant has a
# target the other cannot reach: that asymmetry is what the target guard is
# tested against.
_ALPHA_TARGET = "openai:gpt-4o-mini"
_BETA_TARGET = "anthropic:claude-3-5-haiku-latest"


@dataclass
class _World:
    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    workspace_ids: tuple[uuid.UUID, ...] = (),
) -> str:
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
    for workspace_id in workspace_ids:
        session.add(WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role="member", status="active"))
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
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
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
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, beta_one])
        session.commit()
        session.refresh(alpha_one)
        session.refresh(beta_one)

        for organization_id, provider in ((alpha.id, "openai"), (beta.id, "anthropic")):
            session.add(
                OrgProviderKey(
                    organization_id=organization_id,
                    provider=provider,
                    name=f"{provider}-primary",
                    encrypted_api_key="encrypted",
                    last4="1234",
                    is_org_default=True,
                )
            )
        session.commit()

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "beta_one": beta_one.id}
        built.sessions = {
            "alpha_admin": _identity(session, email="admin@alpha.test", organization_id=alpha.id, role="admin"),
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            "beta_owner": _identity(session, email="owner@beta.test", organization_id=beta.id, role="owner"),
        }
        return built
    finally:
        session.close()


def _post(client: TestClient, world: _World, who: str, path: str, body: dict[str, Any]) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        return client.post(path, json=body)
    finally:
        client.cookies.clear()


def _get(client: TestClient, world: _World, who: str, path: str) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        return client.get(path)
    finally:
        client.cookies.clear()


def _delete(client: TestClient, world: _World, who: str, path: str) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        return client.delete(path)
    finally:
        client.cookies.clear()


def _policy_body(world: _World, *, name: str, target: str = _ALPHA_TARGET, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "spec": {"select": [{"default": target}]},
        "workspace_id": str(world.workspaces["alpha_one"]),
    }
    body.update(extra)
    return body


def test_an_admin_writes_a_policy_into_their_own_workspace(client: TestClient, world: _World) -> None:
    created = _post(client, world, "alpha_admin", _POLICIES, _policy_body(world, name="tenant-fast"))
    assert created.status_code == status.HTTP_200_OK, created.text
    assert created.json()["workspace_id"] == str(world.workspaces["alpha_one"])

    listed = _get(client, world, "alpha_admin", _POLICIES)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    assert "tenant-fast" in {row["name"] for row in listed.json()}


def test_a_member_may_read_but_not_write(client: TestClient, world: _World) -> None:
    """The matrix's whole distinction for these pages, in one test."""
    assert _get(client, world, "alpha_member", _POLICIES).status_code == status.HTTP_200_OK
    refused = _post(client, world, "alpha_member", _POLICIES, _policy_body(world, name="member-fast"))
    assert refused.status_code == status.HTTP_403_FORBIDDEN, refused.text


def test_another_tenants_workspace_is_not_found_rather_than_forbidden(client: TestClient, world: _World) -> None:
    """A bare workspace id says nothing about whose it is, so a 403 would be an oracle."""
    body = _policy_body(world, name="cross-tenant", target=_BETA_TARGET)
    body["workspace_id"] = str(world.workspaces["beta_one"])
    refused = _post(client, world, "alpha_admin", _POLICIES, body)
    assert refused.status_code == status.HTTP_404_NOT_FOUND, refused.text


def test_a_write_must_name_a_workspace(client: TestClient, world: _World) -> None:
    """No default-workspace fallback here: the deployment's default is not the tenant's."""
    body = _policy_body(world, name="unscoped")
    del body["workspace_id"]
    refused = _post(client, world, "alpha_admin", _POLICIES, body)
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


def test_a_user_scoped_write_is_refused(client: TestClient, world: _World) -> None:
    """``user_id`` is deployment-wide, so accepting one would make this a cross-tenant oracle."""
    refused = _post(client, world, "alpha_admin", _POLICIES, _policy_body(world, name="scoped", user_id="someone"))
    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, refused.text


def test_a_target_the_organization_cannot_reach_is_refused(client: TestClient, world: _World) -> None:
    """The escalation guard: a policy is a name for a model, so writing one must not widen access."""
    refused = _post(
        client,
        world,
        "alpha_admin",
        _POLICIES,
        _policy_body(world, name="borrowed", target=_BETA_TARGET),
    )
    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
    assert _BETA_TARGET in refused.json()["detail"]


def test_every_static_candidate_is_checked_not_only_the_default(client: TestClient, world: _World) -> None:
    """An on_failure entry dispatches too, so it is as much a target as the default."""
    body = _policy_body(world, name="fallback-borrowed")
    body["spec"] = {"select": [{"default": _ALPHA_TARGET}], "on_failure": [_BETA_TARGET]}
    refused = _post(client, world, "alpha_admin", _POLICIES, body)
    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text


def test_an_admin_deletes_their_own_policy_and_a_member_cannot(client: TestClient, world: _World) -> None:
    assert _post(client, world, "alpha_admin", _POLICIES, _policy_body(world, name="doomed")).status_code == 200
    path = f"{_POLICIES}/doomed?workspace_id={world.workspaces['alpha_one']}"
    assert _delete(client, world, "alpha_member", path).status_code == status.HTTP_403_FORBIDDEN
    assert _delete(client, world, "alpha_admin", path).status_code == status.HTTP_204_NO_CONTENT
    assert "doomed" not in {row["name"] for row in _get(client, world, "alpha_admin", _POLICIES).json()}


def test_the_alias_sibling_follows_the_same_rules(client: TestClient, world: _World) -> None:
    """One surface, two tables: whatever the policy writer refuses, the alias writer refuses."""
    body = {
        "name": "tenant-alias",
        "target": _ALPHA_TARGET,
        "workspace_id": str(world.workspaces["alpha_one"]),
    }
    created = _post(client, world, "alpha_admin", _ALIASES, body)
    assert created.status_code == status.HTTP_200_OK, created.text
    assert "tenant-alias" in {row["name"] for row in _get(client, world, "alpha_admin", _ALIASES).json()}

    assert _post(client, world, "alpha_member", _ALIASES, body).status_code == status.HTTP_403_FORBIDDEN
    borrowed = _post(client, world, "alpha_admin", _ALIASES, {**body, "name": "borrowed", "target": _BETA_TARGET})
    assert borrowed.status_code == status.HTTP_400_BAD_REQUEST, borrowed.text

    path = f"{_ALIASES}/tenant-alias?workspace_id={world.workspaces['alpha_one']}"
    assert _delete(client, world, "alpha_admin", path).status_code == status.HTTP_204_NO_CONTENT


def test_an_alias_list_shows_no_other_tenants_rows(client: TestClient, world: _World) -> None:
    body = {"name": "beta-alias", "target": _BETA_TARGET, "workspace_id": str(world.workspaces["beta_one"])}
    assert _post(client, world, "beta_owner", _ALIASES, body).status_code == status.HTTP_200_OK
    assert "beta-alias" not in {row["name"] for row in _get(client, world, "alpha_admin", _ALIASES).json()}


def test_the_deployment_wide_writers_still_refuse_a_tenant(client: TestClient, world: _World) -> None:
    """The control: these routers exist so those gates do not have to loosen."""
    assert _post(client, world, "alpha_admin", "/v1/routing/policies", _policy_body(world, name="x")).status_code == (
        status.HTTP_403_FORBIDDEN
    )
    assert _post(
        client,
        world,
        "alpha_admin",
        "/v1/aliases",
        {"name": "x", "target": _ALPHA_TARGET},
    ).status_code == status.HTTP_403_FORBIDDEN
