"""Only a deployment operator reaches the hosted-providers surface, on every route it has.

The surface holds the deployment's own upstream credentials and the price it
charges for them, so nothing about it may change for anyone but the operator:
the master key, the bootstrap operator's session, or a superuser's session.
Everyone else is refused before a handler runs. A caller with no credential
and an API key alike get the 401 the master-key check gives, and a signed-in
non-operator gets the 404 ``/api/v1/admin`` gives, whatever their role in
whichever organization.

The routes are read off the mounted app rather than listed, so a route added
later is walked by every principal here without anyone remembering to add it.
Each refusal is followed by a read of every table the surface writes, so a
gate that answered with a refusal after letting a write through still fails.
"""

import uuid
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.providers import ProviderCredential
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.provisioning_service import BOOTSTRAP_IDENTITY_KEY

from .conftest import build_test_client

PATH = f"{API_ROOT}/hosted-providers"
KEY = "sk-live-hosted-1234"
SECRET_ARG = "shh-client-secret"
# The provider the walks address. Not the one the create example names, so an
# operator's create lands rather than colliding with it.
PROVIDER = "anthropic"
# What discovery lists, kept off the model the add-model example names.
DISCOVERED = "listed-model"
# Every table a route on this surface writes: the providers, their rosters,
# the deployment price list the offers seed, and the organization overrides
# the catalog sweep deletes.
WRITTEN_TABLES = ("hosted_providers", "hosted_provider_models", "model_pricing", "organization_model_pricing")
# What the operator gate answers a signed-in non-operator, word for word, and
# what ``/api/v1/admin`` answers the same caller: it says nothing about the
# surface, not even that this one holds credentials.
NOT_THERE = {"detail": "Not found"}

REFUSED_UNAUTHENTICATED = {
    "anonymous",
    "member_api_key",
    "owner_api_key",
}
REFUSED_SIGNED_IN = {
    "member_session",
    "operator_org_owner_session",
    "own_org_admin_session",
    "own_org_owner_session",
}
OPERATORS = {"bootstrap_operator_session", "superuser_session", "master_key"}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=DISCOVERED, object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


@pytest.fixture(params=["standalone", "hosted"])
def mode_client(
    request: pytest.FixtureRequest, test_config: GatewayConfig, clean_database: None
) -> Generator[TestClient]:
    """A client on each topology that mounts the surface, so the gate is pinned on both."""
    yield from build_test_client(test_config.model_copy(update={"mode": request.param}))


@dataclass
class _World:
    """The principals, each as the request headers and cookie that carry it."""

    headers: dict[str, dict[str, str]] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)


def _hosted_routes(client: TestClient) -> list[tuple[str, str]]:
    """Every (method, path template) the mounted app serves under the surface, in declaration order."""
    routes: list[tuple[str, str]] = []
    for route in iter_route_contexts(client.app.routes):  # type: ignore[attr-defined]
        path = route.path or ""
        if isinstance(route.original_route, APIRoute) and (path == PATH or path.startswith(f"{PATH}/")):
            routes.extend((method, path) for method in sorted(route.methods or ()))
    return routes


def _example_body(client: TestClient, method: str, template: str) -> dict[str, Any] | None:
    """The request example the route's body schema declares, so a refused write is a well-formed one.

    A body with no example is sent empty: still refused at the gate, and a 422
    in the operator walk, which is the prompt to give it one.
    """
    for route in iter_route_contexts(client.app.routes):  # type: ignore[attr-defined]
        if route.path == template and method in (route.methods or ()):
            body_field = route.original_route.body_field  # type: ignore[attr-defined]
            if body_field is None:
                return None
            model = body_field.field_info.annotation
            extra = model.model_config.get("json_schema_extra") or {}
            example: dict[str, Any] = dict(extra.get("example", {}))
            return example
    raise AssertionError(f"{method} {template} is not mounted")


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    workspace_id: uuid.UUID,
    role: str = "member",
    is_superuser: bool = False,
) -> str:
    """A member of one organization and its workspace, holding a live dashboard session; returns the cookie."""
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
        is_superuser=is_superuser,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
    session.add(WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role="member", status="active"))
    return _session_cookie(session, user.id, email)


def _session_cookie(session: Session, user_id: uuid.UUID, label: str) -> str:
    token = f"otari-sess-{label}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user_id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return token


def _api_key(client: TestClient, cookie: str, workspace_id: uuid.UUID, name: str) -> str:
    """A virtual key the identity behind ``cookie`` mints for itself."""
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    try:
        response = client.post(
            f"{API_ROOT}/organizations/me/keys", json={"key_name": name, "workspace_id": str(workspace_id)}
        )
    finally:
        client.cookies.clear()
    assert response.status_code in (200, 201), response.text
    key: str = response.json()["key"]
    return key


def _build_world(client: TestClient, master_key: str, session_factory: Callable[[], Session]) -> _World:
    """The operator's organization, one somebody else owns, and every principal the matrix walks."""
    master = {"Authorization": f"Bearer {master_key}"}
    # One master-key call provisions the tenancy root and the bootstrap operator.
    assert client.get(f"{API_ROOT}/organizations/me", headers=master).status_code == 200

    session = session_factory()
    try:
        default = session.query(Organization).filter(col(Organization.slug) == "default").one()
        default_workspace = session.query(Workspace).filter(col(Workspace.organization_id) == default.id).first()
        assert default_workspace is not None
        acme = Organization(name="Acme", slug="acme")
        session.add(acme)
        session.commit()
        session.refresh(acme)
        acme_workspace = Workspace(name="Acme one", organization_id=acme.id)
        session.add(acme_workspace)
        session.commit()
        session.refresh(acme_workspace)

        marker = session.execute(
            text("SELECT value FROM runtime_settings WHERE key = :key"), {"key": BOOTSTRAP_IDENTITY_KEY}
        ).scalar_one()
        cookies = {
            "bootstrap_operator_session": _session_cookie(session, uuid.UUID(marker), "bootstrap"),
            "superuser_session": _identity(
                session,
                email="root@example.com",
                organization_id=default.id,
                workspace_id=default_workspace.id,
                is_superuser=True,
            ),
            "member_session": _identity(
                session, email="member@example.com", organization_id=default.id, workspace_id=default_workspace.id
            ),
            "operator_org_owner_session": _identity(
                session,
                email="owner@example.com",
                organization_id=default.id,
                workspace_id=default_workspace.id,
                role="owner",
            ),
            "own_org_owner_session": _identity(
                session, email="owner@acme.test", organization_id=acme.id, workspace_id=acme_workspace.id, role="owner"
            ),
            "own_org_admin_session": _identity(
                session, email="admin@acme.test", organization_id=acme.id, workspace_id=acme_workspace.id, role="admin"
            ),
        }
        default_workspace_id, acme_workspace_id = default_workspace.id, acme_workspace.id
    finally:
        session.close()

    member_key = _api_key(client, cookies["member_session"], default_workspace_id, "member-key")
    owner_key = _api_key(client, cookies["own_org_owner_session"], acme_workspace_id, "owner-key")
    return _World(
        headers={
            "anonymous": {},
            "member_api_key": {"Authorization": f"Bearer {member_key}"},
            "owner_api_key": {"Authorization": f"Bearer {owner_key}"},
            "master_key": master,
        },
        cookies=cookies,
    )


def _seed(client: TestClient, master: dict[str, str]) -> uuid.UUID:
    """A provider with a priced model on it, through the surface; returns that model's id."""
    created = client.post(
        PATH,
        json={"provider": PROVIDER, "api_key": KEY, "client_args": {"aws_secret_access_key": SECRET_ARG}},
        headers=master,
    )
    assert created.status_code == 201, created.text
    added = client.post(
        f"{PATH}/{PROVIDER}/models",
        json={"model": "priced-model", "input_price_per_million": 1, "output_price_per_million": 2},
        headers=master,
    )
    assert added.status_code == 201, added.text
    return uuid.UUID(added.json()["id"])


def _state(session_factory: Callable[[], Session]) -> dict[str, list[str]]:
    session = session_factory()
    try:
        return {
            table: sorted(str(tuple(row)) for row in session.execute(text(f"SELECT * FROM {table}")))  # noqa: S608
            for table in WRITTEN_TABLES
        }
    finally:
        session.close()


# A deployment provider the move cannot take (named after another provider), so
# the discard route has a target after the move route has run.
LEFTOVER = "groq"


def _seed_leftover(session_factory: Callable[[], Session]) -> None:
    with session_factory() as db:
        db.add(ProviderCredential(instance=LEFTOVER, provider_type="openai", client_args={}))
        db.commit()


def _concrete(template: str, model_id: uuid.UUID) -> str:
    return template.replace("{provider}", PROVIDER).replace("{model_id}", str(model_id)).replace("{instance}", LEFTOVER)


def _request(client: TestClient, world: _World, principal: str, method: str, path: str, body: Any) -> Any:
    cookie = world.cookies.get(principal)
    if cookie is not None:
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
    try:
        return client.request(method, path, json=body, headers=world.headers.get(principal, {}))
    finally:
        client.cookies.clear()


def test_the_surface_has_every_route_the_matrix_assumes(client: TestClient) -> None:
    """A floor under the derivation: an empty route list would pass every walk below vacuously."""
    routes = _hosted_routes(client)

    assert len(routes) >= 12
    assert ("POST", PATH) in routes
    assert ("DELETE", f"{PATH}/{{provider}}/models/{{model_id}}") in routes


@pytest.mark.parametrize("principal", sorted(REFUSED_UNAUTHENTICATED | REFUSED_SIGNED_IN))
def test_a_non_operator_is_refused_by_every_route_and_changes_nothing(
    mode_client: TestClient,
    test_config: GatewayConfig,
    db_session_factory: Callable[[], Session],
    principal: str,
) -> None:
    client = mode_client
    world = _build_world(client, test_config.master_key or "", db_session_factory)
    model_id = _seed(client, world.headers["master_key"])
    before = _state(db_session_factory)
    expected = 401 if principal in REFUSED_UNAUTHENTICATED else 404
    if expected == 404:
        admin = _request(client, world, principal, "GET", f"{API_ROOT}/admin/users", None)
        assert (admin.status_code, admin.json()) == (404, NOT_THERE)

    for method, template in _hosted_routes(client):
        response = _request(
            client, world, principal, method, _concrete(template, model_id), _example_body(client, method, template)
        )

        call = f"{principal}: {method} {template}"
        assert response.status_code == expected, (call, response.text)
        if expected == 404:
            assert response.json() == NOT_THERE, call
        assert KEY not in response.text, call
        assert SECRET_ARG not in response.text, call
        assert _state(db_session_factory) == before, f"{call} was refused but wrote"


@pytest.mark.parametrize("principal", sorted(OPERATORS))
def test_an_operator_reaches_every_route(
    mode_client: TestClient,
    test_config: GatewayConfig,
    db_session_factory: Callable[[], Session],
    principal: str,
) -> None:
    """Each route answers its own success, the deletes last so every earlier route has a target."""
    client = mode_client
    world = _build_world(client, test_config.master_key or "", db_session_factory)
    model_id = _seed(client, world.headers["master_key"])
    _seed_leftover(db_session_factory)
    routes = sorted(
        _hosted_routes(client),
        key=lambda route: (route[0] == "DELETE", -len(route[1]) if route[0] == "DELETE" else 0),
    )

    for method, template in routes:
        response = _request(
            client, world, principal, method, _concrete(template, model_id), _example_body(client, method, template)
        )

        assert 200 <= response.status_code < 300, (principal, method, template, response.text)


def test_hybrid_mode_mounts_none_of_the_surface(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hybrid gateway has no local control plane, so the routes are absent rather than gated."""
    routes = _hosted_routes(client)
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    config = GatewayConfig(
        mode="hybrid", master_key="test-master-key", platform={"base_url": "http://localhost:8100/api/v1"}
    )

    try:
        with TestClient(create_app(config)) as hybrid:
            assert _hosted_routes(hybrid) == []
            for method, template in routes:
                path = _concrete(template, uuid.uuid4())
                for headers in ({}, {"Authorization": "Bearer test-master-key"}):
                    response = hybrid.request(
                        method, path, json=_example_body(client, method, template), headers=headers
                    )
                    assert response.status_code == 404, (method, template, response.text)
    finally:
        reset_config()
        reset_db()
