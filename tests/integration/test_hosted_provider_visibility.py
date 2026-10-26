"""Who is shown a hosted provider's models and rates, across organizations, on the core adapter.

The deployment hosts openai for everyone and deepseek for Alpha alone. Each
read that names models or rates (the public catalog, the per-organization
catalogs, the price list) is asked by a visitor, by members and owners of
both organizations, and by an organization API key: Alpha's restricted provider
must reach nobody outside Alpha, and a model the operator switched off must
reach nobody the deployment would serve it to.
"""

import uuid
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services import model_catalog_service as mcs
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
OPEN_MODEL = "openai:gpt-4o"
OFF_MODEL = "openai:gpt-4o-mini"
RESTRICTED_MODEL = "deepseek:restricted-secret-model"
# The flat listing names each model by its catalog id rather than by selector.
OPEN_ID = "openai/gpt-4o"
OFF_ID = "openai/gpt-4o-mini"
RESTRICTED_ID = "restricted-secret-model"


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No ambient provider key, which would serve a provider on its own, and a real secret box."""
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _no_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(provider=impl_name, models=[])

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


@pytest.fixture
def client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    """A deployment with a public catalog, no configured instance and no discovery: the catalog is its prices."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        default_pricing=False,
        model_discovery=False,
        public_catalog=True,
    )
    mcs.clear_catalog_cache()
    try:
        yield from build_test_client(config)
    finally:
        mcs.clear_catalog_cache()


@dataclass
class _World:
    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)


def _identity(
    session: Session, *, email: str, organization_id: uuid.UUID, role: str, workspace_id: uuid.UUID | None = None
) -> str:
    user = User(email=email, full_name=email, active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
    if workspace_id is not None:
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


def _ok(response: Any, expected: int = status.HTTP_200_OK) -> Any:
    assert response.status_code == expected, response.text
    return response.json() if response.content else None


@pytest.fixture
def world(client: TestClient, db_session_factory: Callable[[], Session]) -> _World:
    # The deployment provisions its own organization before a second one exists beside it.
    _ok(client.get(f"{API_ROOT}/organizations/me", headers=MASTER))
    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha")
        beta = Organization(name="Beta", slug="beta")
        session.add_all([alpha, beta])
        session.commit()
        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, beta_one])
        session.commit()
        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "beta_one": beta_one.id}
        built.sessions = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            "alpha_member": _identity(
                session, email="member@alpha.test", organization_id=alpha.id, role="member", workspace_id=alpha_one.id
            ),
            "beta_owner": _identity(session, email="owner@beta.test", organization_id=beta.id, role="owner"),
            "beta_member": _identity(
                session, email="member@beta.test", organization_id=beta.id, role="member", workspace_id=beta_one.id
            ),
        }
    finally:
        session.close()

    _ok(
        client.post(
            f"{API_ROOT}/hosted-providers", json={"provider": "openai", "api_key": "sk-hosted"}, headers=MASTER
        ),
        status.HTTP_201_CREATED,
    )
    _ok(
        client.post(
            f"{API_ROOT}/hosted-providers",
            json={"provider": "deepseek", "api_key": "sk-hosted-ds", "restricted_to_organization_id": str(built.alpha)},
            headers=MASTER,
        ),
        status.HTTP_201_CREATED,
    )
    for model_key, rates in ((OPEN_MODEL, (2.5, 10)), (OFF_MODEL, (0.15, 0.6)), (RESTRICTED_MODEL, (3, 4))):
        provider, model = model_key.split(":", 1)
        _ok(
            client.post(
                f"{API_ROOT}/hosted-providers/{provider}/models",
                json={"model": model, "input_price_per_million": rates[0], "output_price_per_million": rates[1]},
                headers=MASTER,
            ),
            status.HTTP_201_CREATED,
        )
    mini = next(
        row
        for row in _ok(client.get(f"{API_ROOT}/hosted-providers/openai/models", headers=MASTER))["data"]
        if row["model"] == "gpt-4o-mini"
    )
    _ok(
        client.patch(f"{API_ROOT}/hosted-providers/openai/models/{mini['id']}", json={"enabled": False}, headers=MASTER)
    )
    return built


def _as(client: TestClient, world: _World, who: str | None, path: str) -> Any:
    """GET ``path`` as a named session, or as a visitor when ``who`` is None."""
    if who is not None:
        client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        return client.get(f"{API_ROOT}{path}")
    finally:
        client.cookies.clear()


def _catalog_selectors(response: Any) -> set[str]:
    return {selector for model in _ok(response)["models"] for selector in model["selectors"]}


def test_a_visitor_to_the_public_catalog_never_sees_a_provider_restricted_to_one_organization(
    client: TestClient, world: _World
) -> None:
    selectors = _catalog_selectors(_as(client, world, None, "/catalog/models"))

    assert OPEN_MODEL in selectors
    assert RESTRICTED_MODEL not in selectors
    assert not any(selector.startswith("deepseek:") for selector in selectors)
    by_id = _as(client, world, None, "/catalog/models/restricted-secret-model")
    assert by_id.status_code == status.HTTP_404_NOT_FOUND, by_id.text
    # The organization it is restricted to still sees it.
    assert RESTRICTED_MODEL in _catalog_selectors(_as(client, world, "alpha_member", "/catalog/models"))


def _api_key(client: TestClient, world: _World, owner: str) -> dict[str, str]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[owner])
    try:
        minted = _ok(client.post(f"{API_ROOT}/organizations/me/keys", json={"key_name": f"{owner} key"}))
    finally:
        client.cookies.clear()
    return {API_KEY_HEADER: f"Bearer {minted['key']}"}


def _price_list_reads(client: TestClient, get: Callable[[str], Any]) -> tuple[set[str], set[str], int, int]:
    """Every price-list read for ``RESTRICTED_MODEL``: the two listings' keys and the two single reads' statuses."""
    versions = {row["model_key"] for row in _ok(get("/pricing"))}
    current = {row["model_key"] for row in _ok(get("/pricing/current"))["data"]}
    single = get(f"/pricing/{RESTRICTED_MODEL}").status_code
    history = get(f"/pricing/{RESTRICTED_MODEL}/history").status_code
    return versions, current, single, history


@pytest.mark.parametrize("who", ["beta_member", "beta_owner", "beta_key"])
def test_the_price_list_withholds_a_restricted_provider_from_another_organization(
    client: TestClient, world: _World, who: str
) -> None:
    if who == "beta_key":
        headers = _api_key(client, world, "beta_owner")

        def get(path: str) -> Any:
            return client.get(f"{API_ROOT}{path}", headers=headers)
    else:

        def get(path: str) -> Any:
            return _as(client, world, who, path)

    versions, current, single, history = _price_list_reads(client, get)

    assert OPEN_MODEL in versions
    assert OPEN_MODEL in current
    assert RESTRICTED_MODEL not in versions
    assert RESTRICTED_MODEL not in current
    assert (single, history) == (status.HTTP_404_NOT_FOUND, status.HTTP_404_NOT_FOUND)
    count = _ok(get("/pricing/current"))["count"]
    assert count == len(current), "the count leaves the withheld key out too"


@pytest.mark.parametrize("who", ["alpha_member", "master"])
def test_the_price_list_still_shows_a_restricted_provider_to_its_organization_and_the_operator(
    client: TestClient, world: _World, who: str
) -> None:
    def get(path: str) -> Any:
        if who == "master":
            return client.get(f"{API_ROOT}{path}", headers=MASTER)
        return _as(client, world, who, path)

    versions, current, single, history = _price_list_reads(client, get)

    assert RESTRICTED_MODEL in versions
    assert RESTRICTED_MODEL in current
    assert (single, history) == (status.HTTP_200_OK, status.HTTP_200_OK)


def _models(client: TestClient, headers: dict[str, str]) -> set[str]:
    return {model["id"] for model in _ok(client.get(f"{API_ROOT}/models", headers=headers))["data"]}


@pytest.mark.parametrize("how", ["switched_off", "deleted"])
def test_an_organization_key_stops_listing_a_hosted_provider_the_operator_took_away(
    client: TestClient, world: _World, how: str
) -> None:
    """Its rates stay on the price list, and dispatch refuses its models, so the catalog says so too."""
    beta_key = _api_key(client, world, "beta_owner")
    assert OPEN_ID in _models(client, beta_key)
    assert OFF_ID not in _models(client, beta_key)

    if how == "switched_off":
        _ok(client.patch(f"{API_ROOT}/hosted-providers/openai", json={"enabled": False}, headers=MASTER))
    else:
        _ok(client.delete(f"{API_ROOT}/hosted-providers/openai", headers=MASTER))

    listed = _models(client, beta_key)
    assert OPEN_ID not in listed
    assert OFF_ID not in listed
    assert client.get(f"{API_ROOT}/models/{OPEN_MODEL}", headers=beta_key).status_code == status.HTTP_404_NOT_FOUND
    # The member's session agrees, and the deployment's own view keeps its price list.
    assert OPEN_ID not in {m["id"] for m in _ok(_as(client, world, "beta_member", "/models"))["data"]}
    assert OPEN_ID in _models(client, MASTER)


def test_an_organization_key_still_lists_a_model_it_reaches_on_its_own_key(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    from gateway.models.provider_keys import OrgProviderKey
    from gateway.services.secret_box import encrypt_secret

    session = db_session_factory()
    try:
        session.add(
            OrgProviderKey(
                organization_id=world.beta,
                provider="openai",
                name="openai-own",
                encrypted_api_key=encrypt_secret("sk-beta-own"),
                last4="1234",
                is_org_default=True,
            )
        )
        session.commit()
    finally:
        session.close()
    _ok(client.patch(f"{API_ROOT}/hosted-providers/openai", json={"enabled": False}, headers=MASTER))

    assert OPEN_ID in _models(client, _api_key(client, world, "beta_owner"))


_ALPHA = ("alpha_owner", "alpha_member")
_BETA = ("beta_owner", "beta_member")


@pytest.mark.parametrize("who", [*_ALPHA, *_BETA])
def test_each_member_of_each_organization_is_shown_what_its_organization_reaches(
    client: TestClient, world: _World, who: str
) -> None:
    """An owner and a workspace member of both organizations, on the flat list, the grouped one and a single read."""
    listed = {model["id"] for model in _ok(_as(client, world, who, "/models"))["data"]}
    grouped = _catalog_selectors(_as(client, world, who, "/catalog/models"))
    single = _as(client, world, who, f"/models/{RESTRICTED_MODEL}").status_code

    for shown, (open_model, off_model, restricted) in (
        (listed, (OPEN_ID, OFF_ID, RESTRICTED_ID)),
        (grouped, (OPEN_MODEL, OFF_MODEL, RESTRICTED_MODEL)),
    ):
        assert open_model in shown
        assert off_model not in shown, "switched off for everyone"
        assert (restricted in shown) is (who in _ALPHA)
    assert single == (status.HTTP_200_OK if who in _ALPHA else status.HTTP_404_NOT_FOUND)


@pytest.mark.parametrize("owner", ["alpha_owner", "beta_owner"])
def test_each_organizations_api_key_is_shown_what_its_organization_reaches(
    client: TestClient, world: _World, owner: str
) -> None:
    headers = _api_key(client, world, owner)
    listed = _models(client, headers)
    grouped = _catalog_selectors(client.get(f"{API_ROOT}/catalog/models", headers=headers))

    for shown, (open_model, off_model, restricted) in (
        (listed, (OPEN_ID, OFF_ID, RESTRICTED_ID)),
        (grouped, (OPEN_MODEL, OFF_MODEL, RESTRICTED_MODEL)),
    ):
        assert open_model in shown
        assert off_model not in shown
        assert (restricted in shown) is (owner == "alpha_owner")
