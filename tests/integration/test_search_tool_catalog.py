"""``GET /api/v1/search-tools/catalog``: the search tools any caller may name, priced for its organization.

The search counterpart of ``GET /api/v1/models``. An API key reads it, sees only
the tools a direct search would serve, and never their keys, endpoints or options.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.tools import reset_search_tool_cache

from .conftest import build_test_client

CATALOG = f"{API_ROOT}/search-tools/catalog"


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """Two search tools any-search serves, one priced and the default, and one it cannot run."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        search_tools={
            "fake-search": {
                "provider": "fake",
                "api_base": "https://search.internal.example",
                "options": {"hits": []},
                "price_per_call": 0.004,
            },
            "fake-fast": {"provider": "fake"},
            "legacy": {"provider": "tavily", "api_key": "tvly-secret-key"},
        },
        web_search_default_tool="fake-search",
    )


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    reset_search_tool_cache()
    yield
    reset_search_tool_cache()


@pytest.fixture
def search_off_client(test_config: GatewayConfig, clean_database: None) -> Iterator[TestClient]:
    """A deployment whose workspaces search only once they turn it on."""
    yield from build_test_client(test_config.model_copy(update={"web_search_workspace_default": False}))


def _entries(client: TestClient, headers: dict[str, str]) -> list[dict[str, Any]]:
    resp = client.get(CATALOG, headers=headers)
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    assert body["object"] == "list"
    entries: list[dict[str, Any]] = body["data"]
    return entries


def test_an_api_key_lists_the_tools_a_direct_search_serves_with_their_price_and_the_default(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    resp = client.get(CATALOG, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == [
        {
            "id": "fake-fast",
            "object": "search_tool",
            "provider": "fake",
            "label": "Fake",
            "price_per_call": None,
            "default": False,
        },
        {
            "id": "fake-search",
            "object": "search_tool",
            "provider": "fake",
            "label": "Fake",
            "price_per_call": 0.004,
            "default": True,
        },
    ]
    # Nothing of a tool's configuration reaches the caller, and a tool a direct search cannot run is not listed.
    assert "search.internal.example" not in resp.text
    assert "tvly-secret-key" not in resp.text
    assert "legacy" not in resp.text


def test_the_callers_organization_override_prices_its_listing(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    override = {
        "kind": "search",
        "model_key": "fake-search",
        "input_price_per_million": 1_000,
        "output_price_per_million": 0,
        "unit": "request",
    }
    created = client.post(f"{API_ROOT}/organizations/me/pricing", json=override, headers=master_key_header)
    assert created.status_code == 201, created.text

    prices = {entry["id"]: entry["price_per_call"] for entry in _entries(client, api_key_header)}

    assert prices == {"fake-fast": None, "fake-search": pytest.approx(0.001)}


def test_a_workspace_with_web_search_off_lists_nothing_to_its_key(
    search_off_client: TestClient, master_key_header: dict[str, str]
) -> None:
    client = search_off_client
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "off"}, headers=master_key_header)
    assert key.status_code == 200, key.text

    assert _entries(client, {API_KEY_HEADER: f"Bearer {key.json()['key']}"}) == []
    # The master key belongs to no one workspace, so it gets the deployment's list.
    assert [entry["id"] for entry in _entries(client, master_key_header)] == ["fake-fast", "fake-search"]


def test_a_key_sees_only_the_tools_its_allowed_tools_grants(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = client.post(
        f"{API_ROOT}/keys", json={"key_name": "narrow", "allowed_tools": ["fake-search"]}, headers=master_key_header
    )
    assert created.status_code == 200, created.text
    narrow = {API_KEY_HEADER: f"Bearer {created.json()['key']}"}

    assert [entry["id"] for entry in _entries(client, narrow)] == ["fake-search"]

    denied = client.patch(
        f"{API_ROOT}/keys/{created.json()['id']}", json={"allowed_tools": []}, headers=master_key_header
    )
    assert denied.status_code == 200, denied.text
    assert _entries(client, narrow) == []


def test_the_catalog_pages_with_skip_and_limit(client: TestClient, api_key_header: dict[str, str]) -> None:
    first = client.get(CATALOG, params={"limit": 1}, headers=api_key_header)
    second = client.get(CATALOG, params={"skip": 1, "limit": 1}, headers=api_key_header)

    assert [entry["id"] for entry in first.json()["data"]] == ["fake-fast"]
    assert [entry["id"] for entry in second.json()["data"]] == ["fake-search"]
    assert client.get(CATALOG, params={"limit": 0}, headers=api_key_header).status_code == 422
    assert client.get(CATALOG, params={"limit": 1001}, headers=api_key_header).status_code == 422
    assert client.get(CATALOG, params={"skip": -1}, headers=api_key_header).status_code == 422


def test_the_catalog_needs_a_credential_and_the_operator_listing_stays_operator_only(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    assert client.get(CATALOG).status_code == 401
    assert client.get(CATALOG, headers={API_KEY_HEADER: "Bearer not-a-key"}).status_code == 401
    assert client.get(f"{API_ROOT}/search-tools", headers=api_key_header).status_code == 401


@pytest.fixture
def legacy_client(test_config: GatewayConfig, clean_database: None) -> Iterator[TestClient]:
    """A deployment with no search tool, searching only through the legacy settings' Tavily key."""
    legacy = {"search_tools": {}, "web_search_provider": "tavily", "web_search_provider_api_key": "tvly-secret-key"}
    yield from build_test_client(test_config.model_copy(update=legacy))


def test_a_deployment_with_only_the_legacy_settings_lists_their_engine_at_the_web_search_price(
    legacy_client: TestClient, master_key_header: dict[str, str]
) -> None:
    client = legacy_client
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "legacy"}, headers=master_key_header)
    assert created.status_code == 200, created.text
    api_key_header = {API_KEY_HEADER: f"Bearer {created.json()['key']}"}
    assert _entries(client, api_key_header) == [
        {
            "id": "tavily",
            "object": "search_tool",
            "provider": "tavily",
            "label": "Tavily",
            "price_per_call": None,
            "default": True,
        }
    ]
    priced = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "otari:web_search", "input_price_per_million": 3000.0, "output_price_per_million": 0.0},
        headers=master_key_header,
    )
    assert priced.status_code in (200, 201), priced.text

    [entry] = _entries(client, api_key_header)
    assert entry["price_per_call"] == pytest.approx(0.003)
    assert "tvly-secret-key" not in client.get(CATALOG, headers=api_key_header).text


def test_a_key_lists_the_legacy_engine_only_when_its_allowed_tools_grants_it_by_name(
    legacy_client: TestClient, master_key_header: dict[str, str]
) -> None:
    client = legacy_client
    listed: dict[str, list[str]] = {}
    for allowed in (["tavily"], ["builtin_fetch"]):
        created = client.post(
            f"{API_ROOT}/keys", json={"key_name": allowed[0], "allowed_tools": allowed}, headers=master_key_header
        )
        assert created.status_code == 200, created.text
        headers = {API_KEY_HEADER: f"Bearer {created.json()['key']}"}
        listed[allowed[0]] = [entry["id"] for entry in _entries(client, headers)]

    assert listed == {"tavily": ["tavily"], "builtin_fetch": []}


def test_a_search_tool_hides_the_legacy_engine_from_the_catalog(
    test_config: GatewayConfig, clean_database: None, master_key_header: dict[str, str]
) -> None:
    legacy = {"web_search_provider": "tavily", "web_search_provider_api_key": "tvly-secret-key"}
    for client in build_test_client(test_config.model_copy(update=legacy)):
        assert [entry["id"] for entry in _entries(client, master_key_header)] == ["fake-fast", "fake-search"]
