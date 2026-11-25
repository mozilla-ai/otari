"""The catalog's data-location filter and facet, drawn from what the hosted providers declare.

Locations are discovery only: a provider's declaration, or a region's own, is
read as dispatch reads it (``HostedSource.region_credentials``), and a viewer
sees only the providers its organization may use.
"""

import uuid
from collections.abc import Callable, Generator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.services import model_catalog_service as mcs
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client
from .hosted_port_helpers import HostedModelProvider, bind_model_provider
from .test_catalog_endpoint import _NEBIUS_GLM, _NEBIUS_KIMI, _config, _get, _price

_MASTER = {"Authorization": "Bearer test-master-key"}
_CATALOG = f"{API_ROOT}/catalog/models"
_PROVIDERS = f"{API_ROOT}/hosted-providers"


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="glm-5.3", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    yield


def _ok(response: Any) -> Any:
    assert response.status_code in (200, 201), response.text
    return response.json()


def _create(client: TestClient, name: str, **fields: Any) -> Any:
    return _ok(
        client.post(
            _PROVIDERS,
            json={
                "provider": "openai",
                "name": name,
                "api_key": "sk-fleet",
                "api_base": "https://eu.hosted.example/v1",
                **fields,
            },
            headers=_MASTER,
        )
    )


def _region(client: TestClient, provider: str, slug: str, **fields: Any) -> Any:
    return _ok(
        client.post(
            f"{_PROVIDERS}/{provider}/regions",
            json={"slug": slug, "label": slug.upper(), "api_key": f"sk-{slug}", **fields},
            headers=_MASTER,
        )
    )


def _locations(client: TestClient, headers: dict[str, str] | None = _MASTER) -> tuple[list[str], dict[str, list[str]]]:
    body = _get(client, _CATALOG, headers=headers, params={"include_facets": True})
    return body["facets"]["data_locations"], {model["id"]: model["data_locations"] for model in body["models"]}


@pytest.fixture
def client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    """Nebius (a ``providers:`` instance, so no location) serves both models; hosted providers serve them elsewhere."""
    mcs.clear_catalog_cache()
    try:
        for client in build_test_client(_config(postgres_url)):
            bind_model_provider(
                client,
                HostedModelProvider(
                    "berlin",
                    "oregon",
                    "nowhere",
                    models={"berlin": ["glm-5.3"], "oregon": ["glm-5.3"], "nowhere": ["glm-5.3"]},
                ),
            )
            for key in (_NEBIUS_GLM, _NEBIUS_KIMI, "berlin:glm-5.3", "oregon:glm-5.3", "nowhere:glm-5.3"):
                _price(client, _MASTER, key, 0.5, 2.0)
            yield client
    finally:
        mcs.clear_catalog_cache()


def test_a_hosted_providers_declaration_is_its_models_location(client: TestClient) -> None:
    _create(client, "berlin", data_location="de")
    _create(client, "oregon", data_location="US")
    _create(client, "nowhere")

    facet, by_model = _locations(client)

    assert facet == ["DE", "US"]
    assert by_model == {"z-ai/glm-5.3": ["DE", "US"], "moonshotai/kimi-k2.6": []}


def test_the_filter_matches_as_a_workspaces_allowed_locations_do(client: TestClient) -> None:
    _create(client, "berlin", data_location="DE")
    _create(client, "oregon", data_location="US")

    def ids(**params: Any) -> list[str]:
        return [model["id"] for model in _get(client, _CATALOG, headers=_MASTER, params=params)["models"]]

    assert ids(data_location=["EU"]) == ["z-ai/glm-5.3"]
    assert ids(data_location=["DE"]) == ["z-ai/glm-5.3"]
    assert ids(data_location=["FR"]) == []
    assert ids(data_location=["GLOBAL"]) == []
    # With nothing selected, a model no provider declares a location for is still listed.
    assert set(ids()) == {"z-ai/glm-5.3", "moonshotai/kimi-k2.6"}


def test_each_enabled_region_is_a_location_and_a_disabled_one_is_not(client: TestClient) -> None:
    _create(client, "berlin", data_location="DE")
    _region(client, "berlin", "frankfurt", data_location="DE")
    _region(client, "berlin", "virginia", data_location="US")
    _region(client, "berlin", "tokyo", data_location="JP")
    _region(client, "berlin", "silent")
    _ok(client.patch(f"{_PROVIDERS}/berlin/regions/tokyo", json={"enabled": False}, headers=_MASTER))

    facet, by_model = _locations(client)

    # The provider's own connection stays a region, the region left out declares nothing, the disabled one is gone.
    assert facet == ["DE", "US"]
    assert by_model["z-ai/glm-5.3"] == ["DE", "US"]


def test_a_provider_restricted_to_one_organization_does_not_lend_its_location_to_another(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    # The deployment provisions its own organization before a second one exists beside it.
    _ok(client.get(f"{API_ROOT}/organizations/me", headers=_MASTER))
    session = db_session_factory()
    try:
        tokens = {}
        organizations = {}
        for slug in ("acme", "globex"):
            organization = Organization(name=slug, slug=slug)
            session.add(organization)
            session.commit()
            session.refresh(organization)
            user = User(email=f"admin@{slug}.test", full_name=slug, active_organization_id=organization.id)
            session.add(user)
            session.commit()
            session.refresh(user)
            session.add(
                OrganizationMember(organization_id=organization.id, user_id=user.id, role="admin", status="active")
            )
            tokens[slug] = f"otari-sess-{slug}"
            session.add(
                DashboardSession(
                    token_hash=hash_session_token(tokens[slug]),
                    user_id=user.id,
                    created_at=datetime.now(UTC),
                    expires_at=datetime.now(UTC) + timedelta(hours=1),
                )
            )
            organizations[slug] = organization.id
        session.commit()
    finally:
        session.close()
    restricted_to: uuid.UUID = organizations["acme"]

    _create(client, "oregon", data_location="US")
    _create(client, "berlin", data_location="DE", restricted_to_organization_id=str(restricted_to))

    seen = {}
    for slug, token in tokens.items():
        client.cookies.set(SESSION_COOKIE_NAME, token)
        try:
            seen[slug] = _locations(client, headers=None)
        finally:
            client.cookies.clear()
    # The master key acts in the deployment's own organization, which is neither of them.
    assert _locations(client)[0] == ["US"]

    assert seen["acme"][0] == ["DE", "US"]
    assert seen["globex"][0] == ["US"]
